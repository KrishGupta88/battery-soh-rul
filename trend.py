"""
trend.py
--------
Extrapolation-safe degradation modelling used by rul_core.py and models.py.

Why this module exists
~~~~~~~~~~~~~~~~~~~~~~
* Tree models (Random Forest) cannot predict outside the range they were trained on, and a plain
  rolling mean shrinks its window at the ends of the series (biasing the last points upward on a
  falling curve). Both produced misleading numbers in the reports.
* Capacity fade is smooth and monotone, so it is modelled first with a small family of parametric
  curves (selected by *pseudo-extrapolation*: fit on the early part of the history, score on the
  later part). ML / DL models then only learn the *residual* on top of that trend.

Everything here is pure numpy / scipy / sklearn (no Streamlit, no torch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression


# --------------------------------------------------------------------------
# 1. EDGE-SAFE SMOOTHING
# --------------------------------------------------------------------------

def _odd(n: int, lo: int = 5) -> int:
    n = max(int(n), lo)
    return n if n % 2 == 1 else n + 1


def smooth_series(y: np.ndarray, frac: float = 0.05, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing that fits a local polynomial *at the edges too* (mode='interp'),
    so the first/last points are not biased the way a shrinking-window rolling mean is."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 7:
        return y.copy()
    w = _odd(frac * n)
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= polyorder:
        return y.copy()
    return savgol_filter(y, window_length=w, polyorder=polyorder, mode="interp")


def smooth_derivative(x: np.ndarray, y: np.ndarray, frac: float = 0.08, polyorder: int = 2) -> np.ndarray:
    """dy/dx using a Savitzky-Golay derivative (edge-safe). Assumes (nearly) uniform x spacing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 7:
        return np.gradient(y, x) if n > 1 else np.zeros(n)
    w = min(_odd(frac * n, lo=7), n if n % 2 == 1 else n - 1)
    dx = float(np.median(np.diff(x))) or 1.0
    return savgol_filter(y, window_length=w, polyorder=min(polyorder, w - 1), deriv=1, delta=dx, mode="interp")


def monotone_smooth(y: np.ndarray, frac: float = 0.03) -> np.ndarray:
    """Robust denoise for a capacity / SOH curve: median filter (spikes) -> edge-safe
    Savitzky-Golay -> non-increasing isotonic projection (capacity fade is monotone)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 7:
        return y.copy()
    med = np.array(pd.Series(y).rolling(window=5, center=True, min_periods=1).median(), dtype=float)
    # fix the 2 edge points of the median filter by keeping the raw value there
    med[:2], med[-2:] = y[:2], y[-2:]
    sg = smooth_series(med, frac=frac)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    return iso.fit_transform(np.arange(n, dtype=float), sg)


# --------------------------------------------------------------------------
# 2. DEGRADATION-TREND FAMILIES  (all linear in their amplitude params for a fixed shape param)
# --------------------------------------------------------------------------

def _basis(kind: str, x: np.ndarray, x0: float, xs: float, shape: float) -> np.ndarray:
    t = (np.asarray(x, dtype=float) - x0) / xs
    one = np.ones_like(t)
    if kind == "linear":
        return np.column_stack([one, t])
    if kind == "quadratic":
        return np.column_stack([one, t, t * t])
    if kind == "power":             # y = a + b * t^p
        return np.column_stack([one, np.power(np.clip(t, 0, None), shape)])
    if kind == "exp":               # y = a + b * (exp(c t) - 1)
        return np.column_stack([one, np.expm1(shape * t)])
    raise ValueError(kind)


_SHAPES = {
    "linear": [0.0],
    "quadratic": [0.0],
    "power": [1.2, 1.5, 2.0, 2.5, 3.0, 0.7, 0.85],
    "exp": [0.5, 1.0, 2.0, 3.0, 4.5],
    "local_linear": [0.0],
    "local_quadratic": [0.0],
}


@dataclass
class _Cand:
    kind: str
    shape: float
    coef: np.ndarray
    x0: float
    xs: float
    fit_from: float            # first x used for the fit (local models)
    anchor_x: float            # last observed x
    anchor_y: float            # fitted value at last observed x
    holdout_mae: float = np.inf

    def predict(self, xq: np.ndarray) -> np.ndarray:
        base = "quadratic" if self.kind == "local_quadratic" else (
            "linear" if self.kind == "local_linear" else self.kind)
        return _basis(base, xq, self.x0, self.xs, self.shape) @ self.coef


def _fit_cand(kind: str, shape: float, x: np.ndarray, y: np.ndarray, frac_local: float) -> Optional[_Cand]:
    n = len(x)
    if kind.startswith("local"):
        k = max(8, int(round(frac_local * n)) if kind == "local_linear" else int(round(0.5 * n)))
        xf, yf = x[-k:], y[-k:]
    else:
        xf, yf = x, y
    if len(xf) < (4 if "quadratic" in kind else 3):
        return None
    base = "quadratic" if kind == "local_quadratic" else ("linear" if kind == "local_linear" else kind)
    x0, xs = float(x[0]), float(max(x[-1] - x[0], 1.0))
    B = _basis(base, xf, x0, xs, shape)
    coef, *_ = np.linalg.lstsq(B, yf, rcond=None)
    if base in ("power", "exp") and coef[1] > 0:       # fade must lower SOH
        return None
    cand = _Cand(kind, shape, coef, x0, xs, float(xf[0]), float(x[-1]), 0.0)
    cand.anchor_y = float(cand.predict(np.array([x[-1]]))[0])
    return cand


def _forecast(cand: _Cand, xq: np.ndarray, y_last: float, x_last: float) -> np.ndarray:
    """Candidate forecast, re-anchored to the last observed (smoothed) value so there is no jump,
    and forced to be non-increasing (a cell cannot regain capacity)."""
    p = cand.predict(xq)
    p = p + (y_last - float(cand.predict(np.array([x_last]))[0]))
    return np.minimum.accumulate(p)


@dataclass
class TrendFit:
    """Ensemble of the best few trend candidates, selected by hold-out extrapolation error."""
    cands: list[_Cand]
    x_last: float
    y_last: float
    names: list[str] = field(default_factory=list)
    holdout_mae: float = np.nan

    def predict(self, xq) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        preds = np.stack([_forecast(c, xq, self.y_last, self.x_last) for c in self.cands])
        out = preds.mean(axis=0)
        # in-sample part: use the un-anchored fit shape but still monotone
        return out

    def predict_insample(self, xq) -> np.ndarray:
        """Plain (un-anchored) ensemble mean, used for fitted values on the training range."""
        xq = np.asarray(xq, dtype=float)
        return np.stack([c.predict(xq) for c in self.cands]).mean(axis=0)

    def eol_cycle(self, eol: float, horizon_mult: float = 4.0, x_start: Optional[float] = None) -> float:
        """First cycle at which the forecast crosses `eol` (nan if it never does within the horizon)."""
        lo = self.x_last if x_start is None else x_start
        span = max(self.x_last - self.cands[0].x0, 1.0)
        grid = np.arange(lo, self.x_last + horizon_mult * span + 1.0, max(span / 2000.0, 0.25))
        p = self.predict(grid)
        hit = np.where(p <= eol)[0]
        if len(hit) == 0:
            return float("nan")
        i = hit[0]
        if i == 0:
            return float(grid[0])
        # linear interpolation inside the crossing step
        x0g, x1g, p0, p1 = grid[i - 1], grid[i], p[i - 1], p[i]
        return float(x0g + (p0 - eol) / max(p0 - p1, 1e-12) * (x1g - x0g))


def fit_trend(x: np.ndarray, y: np.ndarray, top_k: int = 3, holdout_frac: float = 0.25,
              kinds: Optional[list[str]] = None) -> TrendFit:
    """Fit all candidate fade curves, rank them by how well a fit on the first (1-holdout_frac) of the
    history extrapolates onto the last `holdout_frac`, refit the best `top_k` on the full history."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    kinds = kinds or list(_SHAPES)
    x_last, y_last = float(x[-1]), float(y[-1])

    if n < 12:                                          # too short to validate: plain linear
        c = _fit_cand("linear", 0.0, x, y, 0.3)
        return TrendFit([c], x_last, y_last, ["linear"], float("nan"))

    n_tr = max(8, int(round((1.0 - holdout_frac) * n)))
    n_tr = min(n_tr, n - 3)
    x_tr, y_tr, x_ho, y_ho = x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:]
    # robust level for anchoring in the validation step
    y_tr_last = float(np.mean(y_tr[-3:]))
    x_tr_last = float(np.mean(x_tr[-3:]))

    scored: list[tuple[float, str, float]] = []
    for kind in kinds:
        for shape in _SHAPES[kind]:
            c = _fit_cand(kind, shape, x_tr, y_tr, 0.3)
            if c is None:
                continue
            # anchor on the average of the last 3 fitted points (less noisy than the last point)
            c_anchor = float(np.mean(c.predict(x_tr[-3:])))
            pred = np.minimum.accumulate(c.predict(x_ho) + (y_tr_last - c_anchor))
            mae = float(np.mean(np.abs(pred - y_ho)))
            scored.append((mae, kind, shape))
    scored.sort(key=lambda t: t[0])

    chosen: list[_Cand] = []
    names: list[str] = []
    seen_kind: set[str] = set()
    for mae, kind, shape in scored:                     # diversity: at most one per family
        if kind in seen_kind:
            continue
        c = _fit_cand(kind, shape, x, y, 0.3)
        if c is None:
            continue
        c.holdout_mae = mae
        chosen.append(c)
        names.append(kind if shape == 0.0 else f"{kind}({shape:g})")
        seen_kind.add(kind)
        if len(chosen) >= top_k:
            break
    if not chosen:
        c = _fit_cand("linear", 0.0, x, y, 0.3)
        return TrendFit([c], x_last, y_last, ["linear"], float("nan"))

    # anchor level = average of the last 3 *fitted* values of the ensemble (robust to end noise)
    y_anchor = float(np.mean([np.mean(c.predict(x[-3:])) for c in chosen]))
    return TrendFit(chosen, x_last, y_anchor, names, float(np.mean([c.holdout_mae for c in chosen])))


# --------------------------------------------------------------------------
# 3. CAUSAL EOL ESTIMATES  (what a deployed system could have known at each cycle)
# --------------------------------------------------------------------------

def causal_eol_estimates(cycles: np.ndarray, soh: np.ndarray, eol: float, min_hist: int = 40,
                         n_cutoffs: int = 70) -> np.ndarray:
    """For every cycle i return the EOL cycle predicted using ONLY data up to cycle i.

    The (expensive) ensemble fit is refreshed at ~`n_cutoffs` cutoffs and carried forward in between
    (so the value used at cycle i never depends on anything after i). Once the *observed, causally
    smoothed* SOH is at/below `eol`, the EOL cycle is the first observed crossing.
    Returns an array of EOL-cycle estimates (np.nan before `min_hist` cycles of history)."""
    cycles = np.asarray(cycles, dtype=float)
    soh = np.asarray(soh, dtype=float)
    n = len(cycles)
    out = np.full(n, np.nan)
    if n <= min_hist:
        return out
    cut_idx = np.unique(np.linspace(min_hist, n - 1, n_cutoffs).round().astype(int))
    last_est = np.nan
    crossed_at = np.nan
    j = 0
    for i in range(min_hist, n):
        if np.isnan(crossed_at):
            # causal denoise of the history so far (edge-safe, only uses <= i)
            if i in cut_idx or j == 0:
                hist = monotone_smooth(soh[: i + 1])
                if hist[-1] <= eol:
                    k = int(np.argmax(hist <= eol))
                    crossed_at = float(cycles[k])
                    last_est = crossed_at
                else:
                    fit = fit_trend(cycles[: i + 1], hist)
                    e = fit.eol_cycle(eol)
                    last_est = e if np.isfinite(e) else last_est
                j += 1
            # direct observation also catches a crossing between refreshes
            if soh[i] <= eol and np.isfinite(soh[max(0, i - 2): i + 1]).all() and np.all(soh[max(0, i - 2): i + 1] <= eol):
                crossed_at = float(cycles[i - 2]) if i >= 2 else float(cycles[i])
                last_est = min(last_est, crossed_at) if np.isfinite(last_est) else crossed_at
        out[i] = crossed_at if np.isfinite(crossed_at) else last_est
    return out


# --------------------------------------------------------------------------
# 4. ROLLING-ORIGIN EOL VALIDATION  (the metric that actually matters for RUL)
# --------------------------------------------------------------------------

@dataclass
class RollingEOL:
    table: pd.DataFrame        # cutoff_frac, cutoff_cycle, soh_at_cutoff, pred_eol, true_eol, err_cycles, rul_err_pct
    true_eol: float
    mae_cycles: float
    note: str


def rolling_origin_eol(cycles: np.ndarray, soh_smooth: np.ndarray, eol: float, true_eol: float,
                       fractions=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> RollingEOL:
    """Pretend the file ended at several earlier cutoffs, predict the EOL cycle from that history only,
    and compare with the real (observed) EOL. Requires the EOL to be inside the data."""
    cycles = np.asarray(cycles, dtype=float)
    soh = np.asarray(soh_smooth, dtype=float)
    rows = []
    if not np.isfinite(true_eol) or true_eol > cycles[-1] or true_eol <= cycles[0]:
        return RollingEOL(pd.DataFrame(), float(true_eol), float("nan"),
                          "EOL is not observed inside the file, so rolling-origin validation is not possible.")
    k_true = int(np.argmin(np.abs(cycles - true_eol)))
    for f in fractions:
        k = int(round(f * k_true))                      # cutoff as a fraction of the life up to observed EOL
        if k < 20:
            continue
        hist = monotone_smooth(soh[: k + 1])
        fit = fit_trend(cycles[: k + 1], hist)
        pred = fit.eol_cycle(eol)
        rows.append({
            "cutoff_frac": f, "cutoff_cycle": float(cycles[k]), "soh_at_cutoff_pct": float(hist[-1] * 100),
            "pred_eol": pred, "true_eol": float(true_eol),
            "err_cycles": pred - true_eol if np.isfinite(pred) else np.nan,
            "err_pct_of_life": 100.0 * (pred - true_eol) / max(true_eol - cycles[0], 1.0) if np.isfinite(pred) else np.nan,
            "model": ", ".join(fit.names),
        })
    t = pd.DataFrame(rows)
    mae = float(np.nanmean(np.abs(t["err_cycles"]))) if len(t) else float("nan")
    return RollingEOL(t, float(true_eol), mae, "")
