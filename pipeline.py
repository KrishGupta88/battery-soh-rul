"""
pipeline.py
-----------
Streamlit-free orchestration of the whole modelling stage so it can be unit-tested and so app.py stays thin.

For both targets (SOH and RUL) three predictors are produced on the same chronological hold-out:

  * "Trend"   : extrapolation-safe empirical baseline (SOH: fade-curve ensemble; RUL: causal EOL-hat)
  * "ML"      : Random Forest hybrid  = trend + w * RF(residual)
  * "DL"      : LSTM hybrid           = trend + w * LSTM(residual)

plus a rolling-origin EOL validation (predict the EOL cycle from progressively longer histories).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import models as md
import rul_core as rc
import trend as tr


@dataclass
class TargetResult:
    trend: md.EvalResult
    ml: md.EvalResult
    dl: Optional[md.EvalResult]
    baseline: md.EvalResult                      # straight-line extrapolation (what you'd get with no modelling)
    ml_w: float = 0.0
    dl_w: float = 0.0
    latest_trend: float = float("nan")
    latest_ml: float = float("nan")
    latest_dl: Optional[float] = None


@dataclass
class PipelineResult:
    tr_idx: np.ndarray
    te_idx: np.ndarray
    cycles: np.ndarray
    X_rul: pd.DataFrame
    X_soh: pd.DataFrame
    soh: TargetResult
    rul: TargetResult
    rolling: tr.RollingEOL
    ml_rul_model: object
    ml_soh_model: object
    dl_rul_model: Optional[object]
    dl_soh_model: Optional[object]
    trend_names: str
    loss_rul: list = field(default_factory=list)
    loss_soh: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _linear_forecast(x_tr, y_tr, x_te, tail_frac=0.5):
    """Naive baseline: straight line fitted on the later part of the training data, extended."""
    k = max(5, int(round(tail_frac * len(x_tr))))
    s, i = np.polyfit(x_tr[-k:], y_tr[-k:], 1)
    return i + s * x_te


def run_pipeline(labeled: pd.DataFrame, eol_soh: float, test_frac: float = 0.2, n_estimators: int = 300,
                 lstm_window: int = 10, lstm_hidden: int = 32, lstm_epochs: int = 150, lstm_lr: float = 5e-3,
                 use_dl: bool = True, dl_models: int = 3, true_eol: Optional[float] = None) -> PipelineResult:
    lab = rc.add_causal_rul_features(labeled, eol_soh)
    cycles = lab["cycle"].to_numpy(dtype=float)
    n = len(lab)
    tr_idx, te_idx = md.chronological_split(n, test_frac)
    notes: list[str] = []
    dl_ok = use_dl and md.TORCH_AVAILABLE and len(tr_idx) >= lstm_window + 5

    # ------------------------------------------------------------------ SOH ----
    X_soh, _ = rc.build_feature_matrix(lab, exclude_soh=True)
    raw = lab["soh"].to_numpy(dtype=float)
    # TRAIN target is denoised using TRAIN rows only (no look-ahead into the test block);
    # the evaluation target is the usual full-file smoothed SOH.
    y_soh = lab["soh_smooth"].to_numpy(dtype=float).copy()
    y_soh_train = tr.monotone_smooth(raw[tr_idx])
    y_soh[tr_idx] = y_soh_train

    fit_tr = tr.fit_trend(cycles[tr_idx], y_soh_train)
    base_soh = np.empty(n)
    base_soh[tr_idx] = fit_tr.predict_insample(cycles[tr_idx])
    base_soh[te_idx] = fit_tr.predict(cycles[te_idx])
    lin_soh = _linear_forecast(cycles[tr_idx], y_soh_train, cycles[te_idx])
    y_te = lab["soh_smooth"].to_numpy(dtype=float)[te_idx]
    pre = y_te > eol_soh
    baseline_soh = md.make_eval(y_te, lin_soh, cycles[te_idx], pre_eol_mask=pre)
    trend_soh = md.make_eval(y_te, base_soh[te_idx], cycles[te_idx], lin_soh, pre)

    ml_soh = md.ResidualHybrid(lambda: md.MLModel(n_estimators=n_estimators))
    ml_soh.fit(X_soh.iloc[tr_idx], base_soh[tr_idx], y_soh[tr_idx], cycles[tr_idx])
    p = ml_soh.predict(X_soh, base_soh, cycles)
    ml_soh_eval = md.make_eval(y_te, p[te_idx], cycles[te_idx], lin_soh, pre)
    ml_soh_latest = float(p[-1])

    dl_soh = dl_soh_eval = None
    dl_soh_latest = None
    loss_soh: list = []
    if dl_ok:
        dl_soh = md.ResidualHybrid(lambda: md.DLModel(window=lstm_window, hidden_size=lstm_hidden,
                                                       epochs=lstm_epochs, lr=lstm_lr, n_models=dl_models))
        dl_soh.fit(X_soh.iloc[tr_idx], base_soh[tr_idx], y_soh[tr_idx], cycles[tr_idx])
        p = dl_soh.predict(X_soh, base_soh, cycles)
        dl_soh_eval = md.make_eval(y_te, p[te_idx], cycles[te_idx], lin_soh, pre)
        dl_soh_latest = float(p[-1])
        loss_soh = dl_soh.inner.loss_history

    soh_res = TargetResult(trend_soh, ml_soh_eval, dl_soh_eval, baseline_soh, ml_soh.w,
                           dl_soh.w if dl_soh else 0.0, float(base_soh[-1]), ml_soh_latest, dl_soh_latest)

    # ------------------------------------------------------------------ RUL ----
    X_rul, _ = rc.build_feature_matrix(lab)
    y_rul = lab["rul_label"].to_numpy(dtype=float)
    base_rul = lab["rul_hat"].to_numpy(dtype=float)           # causal at every cycle -> usable for train AND test
    sc_rul = base_rul + 20.0                                   # relative-residual scale: correction shrinks near EOL
    y_te_r = y_rul[te_idx]
    pre_r = y_te_r > 0
    lin_soh_all = _linear_forecast(cycles[tr_idx], y_soh_train, cycles[te_idx])
    # RUL implied by the straight-line SOH baseline
    lin_fit = np.polyfit(cycles[tr_idx][-max(5, len(tr_idx) // 2):], y_soh_train[-max(5, len(tr_idx) // 2):], 1)
    lin_eol = (eol_soh - lin_fit[1]) / lin_fit[0] if lin_fit[0] < 0 else np.nan
    lin_rul = np.clip(lin_eol - cycles[te_idx], 0, None) if np.isfinite(lin_eol) else np.full(len(te_idx), np.nan)
    ok_lin = np.isfinite(lin_rul).all()
    baseline_rul = md.make_eval(y_te_r, lin_rul if ok_lin else base_rul[te_idx], cycles[te_idx], pre_eol_mask=pre_r)
    trend_rul = md.make_eval(y_te_r, base_rul[te_idx], cycles[te_idx], lin_rul if ok_lin else None, pre_r)

    def _rul_post(pred):
        pred = np.clip(pred, 0, None)
        return np.where(base_rul <= 0, 0.0, pred)             # observed EOL crossing -> RUL is exactly 0

    ml_rul = md.ResidualHybrid(lambda: md.MLModel(n_estimators=n_estimators), lower=0.0)
    ml_rul.fit(X_rul.iloc[tr_idx], base_rul[tr_idx], y_rul[tr_idx], cycles[tr_idx], scale=sc_rul[tr_idx])
    p = _rul_post(ml_rul.predict(X_rul, base_rul, cycles, scale=sc_rul))
    ml_rul_eval = md.make_eval(y_te_r, p[te_idx], cycles[te_idx], lin_rul if ok_lin else None, pre_r)
    ml_rul_latest = float(p[-1])

    dl_rul = dl_rul_eval = None
    dl_rul_latest = None
    loss_rul: list = []
    if dl_ok:
        dl_rul = md.ResidualHybrid(lambda: md.DLModel(window=lstm_window, hidden_size=lstm_hidden,
                                                       epochs=lstm_epochs, lr=lstm_lr, n_models=dl_models), lower=0.0)
        dl_rul.fit(X_rul.iloc[tr_idx], base_rul[tr_idx], y_rul[tr_idx], cycles[tr_idx], scale=sc_rul[tr_idx])
        p = _rul_post(dl_rul.predict(X_rul, base_rul, cycles, scale=sc_rul))
        dl_rul_eval = md.make_eval(y_te_r, p[te_idx], cycles[te_idx], lin_rul if ok_lin else None, pre_r)
        dl_rul_latest = float(p[-1])
        loss_rul = dl_rul.inner.loss_history

    rul_res = TargetResult(trend_rul, ml_rul_eval, dl_rul_eval, baseline_rul, ml_rul.w,
                           dl_rul.w if dl_rul else 0.0, float(base_rul[-1]), ml_rul_latest, dl_rul_latest)

    # ------------------------------------------------------- rolling-origin ----
    if true_eol is None:
        hit = np.where(lab["soh_smooth"].to_numpy() <= eol_soh)[0]
        true_eol = float(cycles[hit[0]]) if len(hit) else float("nan")
    rolling = tr.rolling_origin_eol(cycles, lab["soh_smooth"].to_numpy(dtype=float), eol_soh, true_eol)

    if not dl_ok and use_dl:
        notes.append("LSTM was skipped (PyTorch missing or too few training cycles for the chosen window).")

    return PipelineResult(
        tr_idx=tr_idx, te_idx=te_idx, cycles=cycles, X_rul=X_rul, X_soh=X_soh, soh=soh_res, rul=rul_res,
        rolling=rolling, ml_rul_model=ml_rul, ml_soh_model=ml_soh, dl_rul_model=dl_rul, dl_soh_model=dl_soh,
        trend_names=", ".join(fit_tr.names), loss_rul=loss_rul, loss_soh=loss_soh, notes=notes,
    )
