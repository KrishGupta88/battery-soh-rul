"""
models.py
---------
Two interchangeable predictors (used for RUL and for SOH) that consume the per-cycle feature matrix
produced by rul_core.py:

  * MLModel  -> RandomForestRegressor (scikit-learn)
  * DLModel  -> a small LSTM (PyTorch) over sliding windows of cycles, trained with mini-batches,
                weight decay, gradient clipping, early stopping and a small seed ensemble

and one wrapper that makes both of them usable on *extrapolating* problems:

  * ResidualHybrid -> final = base + w * model(X), where `base` is an extrapolation-safe trend
    (see trend.py) and the ML / DL model only learns the residual. The blend weight `w` is chosen
    on a validation block at the END of the training data, so if the residual model does not help
    out-of-sample it is automatically switched off (w = 0) instead of adding noise.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - broken/missing install, missing CUDA libs, etc.
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

@dataclass
class EvalResult:
    mae: float
    rmse: float
    r2: float
    y_true: np.ndarray
    y_pred: np.ndarray
    cycles: np.ndarray
    mae_pre_eol: float = float("nan")   # MAE on the cycles that are still before EOL (the informative part)
    skill: float = float("nan")         # 1 - MAE / MAE(linear-extrapolation baseline); >0 = better than a straight line


def make_eval(y_true, y_pred, cycles, baseline_pred=None, pre_eol_mask=None) -> EvalResult:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cycles = np.asarray(cycles, dtype=float)
    if len(y_true) == 0:
        return EvalResult(float("nan"), float("nan"), float("nan"), y_true, y_pred, cycles)
    mae = float(mean_absolute_error(y_true, y_pred))
    out = EvalResult(
        mae=mae,
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        y_true=y_true, y_pred=y_pred, cycles=cycles,
    )
    if pre_eol_mask is not None and np.asarray(pre_eol_mask).any():
        msk = np.asarray(pre_eol_mask, dtype=bool)
        out.mae_pre_eol = float(mean_absolute_error(y_true[msk], y_pred[msk]))
    if baseline_pred is not None:
        b = float(mean_absolute_error(y_true, np.asarray(baseline_pred, dtype=float)))
        out.skill = float(1.0 - mae / b) if b > 1e-12 else float("nan")
    return out


def chronological_split(n: int, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Split indices [0..n) into (train_idx, test_idx) keeping time order —
    RUL is a time series, so a random split would leak future info into training."""
    n_test = max(1, int(round(n * test_frac)))
    n_test = min(n_test, n - 1) if n > 1 else 0
    idx = np.arange(n)
    return idx[: n - n_test], idx[n - n_test:]


# --------------------------------------------------------------------------
# ML MODEL — Random Forest
# --------------------------------------------------------------------------

class MLModel:
    name = "Random Forest (ML)"

    def __init__(self, n_estimators: int = 300, max_depth: Optional[int] = None, random_state: int = 42,
                 clip_min: Optional[float] = None, min_samples_leaf: int = 3):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            max_features=0.8, random_state=random_state, n_jobs=-1,
        )
        self.clip_min = clip_min
        self.feature_names: list[str] = []
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, cycles: Optional[np.ndarray] = None) -> "MLModel":
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y)
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame, cycles: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.model.predict(X[self.feature_names].values)
        return p if self.clip_min is None else np.clip(p, self.clip_min, None)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, cycles: np.ndarray) -> EvalResult:
        return make_eval(y, self.predict(X), cycles)


# --------------------------------------------------------------------------
# DL MODEL — LSTM (PyTorch)
# --------------------------------------------------------------------------

def make_sequences(X: np.ndarray, y: np.ndarray, cycles: np.ndarray, window: int):
    """Turn a (n_samples, n_features) matrix into overlapping sequences of
    length `window`; the label for each sequence is y at its LAST time step."""
    seqs, targets, seq_cycles = [], [], []
    for i in range(window - 1, len(X)):
        seqs.append(X[i - window + 1: i + 1])
        targets.append(y[i])
        seq_cycles.append(cycles[i])
    if not seqs:
        return np.empty((0, window, X.shape[1])), np.empty((0,)), np.empty((0,))
    return np.stack(seqs), np.array(targets), np.array(seq_cycles)


def _pad_front(X: np.ndarray, window: int) -> np.ndarray:
    """Repeat the first row so that EVERY cycle (incl. the first window-1) has a full window."""
    if window <= 1:
        return X
    return np.vstack([np.repeat(X[:1], window - 1, axis=0), X])


if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)


class DLModel:
    name = "LSTM (DL)"

    def __init__(self, window: int = 10, hidden_size: int = 32, num_layers: int = 1,
                 epochs: int = 150, lr: float = 5e-3, dropout: float = 0.1, random_state: int = 42,
                 standardize_y: bool = True, n_models: int = 3, batch_size: int = 32,
                 weight_decay: float = 1e-4, patience: int = 25, val_frac: float = 0.15):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Run: pip install torch")
        self.window = window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.random_state = random_state
        self.n_models = n_models
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_frac = val_frac
        self.feature_names: list[str] = []
        self.x_scaler = StandardScaler()
        self.standardize_y = standardize_y
        self.y_scale = 1.0
        self.y_mean = 0.0
        self.nets: list = []
        self.fitted = False
        self.loss_history: list[float] = []

    # -- training ----------------------------------------------------------
    def _train_one(self, seq_X, seq_y, seed: int):
        torch.manual_seed(seed)
        n = len(seq_X)
        n_val = int(round(self.val_frac * n)) if n >= 40 else 0
        tr_X, tr_y = seq_X[: n - n_val], seq_y[: n - n_val]      # chronological: validate on the latest block
        va_X, va_y = seq_X[n - n_val:], seq_y[n - n_val:]
        net = _LSTMNet(seq_X.shape[2], self.hidden_size, self.num_layers, self.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(5, self.patience // 3))
        loss_fn = nn.SmoothL1Loss(beta=1.0)
        Xt = torch.tensor(tr_X, dtype=torch.float32)
        yt = torch.tensor(tr_y, dtype=torch.float32)
        Xv = torch.tensor(va_X, dtype=torch.float32) if n_val else None
        yv = torch.tensor(va_y, dtype=torch.float32) if n_val else None
        g = torch.Generator().manual_seed(seed)

        best, best_state, bad, hist = float("inf"), None, 0, []
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xt), generator=g)
            ep_loss = 0.0
            for b in range(0, len(Xt), self.batch_size):
                ib = perm[b: b + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt[ib]), yt[ib])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item()) * len(ib)
            ep_loss /= len(Xt)
            hist.append(ep_loss)
            if n_val:
                net.eval()
                with torch.no_grad():
                    v = float(loss_fn(net(Xv), yv).item())
                sched.step(v)
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            else:
                sched.step(ep_loss)
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net, hist

    def fit(self, X: pd.DataFrame, y: np.ndarray, cycles: np.ndarray) -> "DLModel":
        self.feature_names = list(X.columns)
        Xs = self.x_scaler.fit_transform(X.values)
        y = np.asarray(y, dtype=float)
        if self.standardize_y:
            self.y_mean = float(np.mean(y))
            self.y_scale = float(np.std(y)) or 1.0
        else:
            self.y_mean = 0.0
            self.y_scale = max(float(np.max(np.abs(y))), 1.0)
        y_norm = (y - self.y_mean) / self.y_scale

        seq_X, seq_y, _ = make_sequences(_pad_front(Xs, self.window), np.concatenate(
            [np.repeat(y_norm[:1], self.window - 1), y_norm]), np.arange(len(Xs) + self.window - 1), self.window)
        if len(seq_X) < 2:
            raise ValueError(
                f"Not enough cycles to build even 2 training sequences with window={self.window}. "
                f"Reduce the sequence window or upload a file with more cycles."
            )
        self.nets, hists = [], []
        for k in range(self.n_models):
            net, h = self._train_one(seq_X, seq_y, self.random_state + 101 * k)
            self.nets.append(net)
            hists.append(h)
        L = min(len(h) for h in hists)
        self.loss_history = list(np.mean([h[:L] for h in hists], axis=0))
        self.fitted = True
        return self

    # -- inference ---------------------------------------------------------
    def predict(self, X: pd.DataFrame, cycles: Optional[np.ndarray] = None):
        """Length-preserving: returns (predictions, cycles) with one prediction per input row (the first
        window-1 rows use an edge-padded window). Predictions at row i only use rows <= i (causal)."""
        Xs = self.x_scaler.transform(X[self.feature_names].values)
        padded = _pad_front(Xs, self.window)
        seq_X, _, _ = make_sequences(padded, np.zeros(len(padded)), np.arange(len(padded)), self.window)
        Xt = torch.tensor(seq_X, dtype=torch.float32)
        with torch.no_grad():
            pred = np.mean([net(Xt).numpy() for net in self.nets], axis=0)
        out = pred * self.y_scale + self.y_mean
        return out, (np.asarray(cycles, dtype=float) if cycles is not None else np.arange(len(out), dtype=float))

    def evaluate(self, X: pd.DataFrame, y: np.ndarray, cycles: np.ndarray) -> EvalResult:
        pred, _ = self.predict(X, cycles)
        return make_eval(y, pred, cycles)


# --------------------------------------------------------------------------
# Hybrid: extrapolation-safe base + learned residual
# --------------------------------------------------------------------------

class ResidualHybrid:
    """final = base + w * clip(inner_model(X), +-clip_sd * std(train residual)).

    * `base` must be causal / extrapolation-safe (e.g. a fitted fade curve or the causal EOL-hat).
    * `w` is picked from `weights` by validation MAE on the last `val_frac` of the TRAIN rows; the
      inner model is then refit on all train rows. w = 0 means 'residual model did not help'.
    """

    def __init__(self, make_inner: Callable[[], object], val_frac: float = 0.2,
                 weights=(0.0, 0.25, 0.5, 1.0), clip_sd: float = 3.0, lower: Optional[float] = None):
        self.make_inner = make_inner
        self.val_frac = val_frac
        self.weights = tuple(weights)
        self.clip_sd = clip_sd
        self.lower = lower
        self.inner = None
        self.w = 0.0
        self.res_sd = 0.0
        self.val_mae_by_w: dict[float, float] = {}

    def _inner_pred(self, inner, X: pd.DataFrame, cycles: np.ndarray) -> np.ndarray:
        if isinstance(inner, DLModel):
            return inner.predict(X, cycles)[0]
        return inner.predict(X)

    def fit(self, X: pd.DataFrame, base: np.ndarray, y: np.ndarray, cycles: np.ndarray,
            scale: Optional[np.ndarray] = None) -> "ResidualHybrid":
        """`scale` (optional, > 0, one value per row): the model learns the RELATIVE residual
        (y - base) / scale and the correction is multiplied back by `scale` at prediction time. For RUL
        use scale = base + c so the correction automatically shrinks as the cell approaches EOL."""
        base, y, cycles = (np.asarray(a, dtype=float) for a in (base, y, cycles))
        sc = np.ones_like(base) if scale is None else np.asarray(scale, dtype=float)
        res = (y - base) / sc
        self.res_sd = float(np.std(res)) or 1e-9
        n = len(y)
        self.w = 0.0
        self.val_mae_by_w = {}
        # Forward-chaining validation (3 expanding folds, each validated on the block right after it):
        # the residual model is only switched on if it helps in EVERY recent regime, not just one block.
        if n >= 90:
            starts = [int(f * n) for f in (0.55, 0.70, 0.85)]
            ends = [starts[1], starts[2], n]
            per_fold = {w: [] for w in self.weights}
            try:
                for a, b in zip(starts, ends):
                    m = self.make_inner()
                    m.fit(X.iloc[:a], res[:a], cycles[:a])
                    pv = self._inner_pred(m, X.iloc[:b], cycles[:b])[a:b]
                    pv = np.clip(pv, -self.clip_sd * self.res_sd, self.clip_sd * self.res_sd)
                    for w in self.weights:
                        pred = base[a:b] + w * pv * sc[a:b]
                        if self.lower is not None:
                            pred = np.clip(pred, self.lower, None)
                        per_fold[w].append(float(np.mean(np.abs(pred - y[a:b]))))
                self.val_mae_by_w = {w: float(np.mean(v)) for w, v in per_fold.items()}
                best_w = min(self.weights, key=lambda w: (round(self.val_mae_by_w[w], 12), w))
                gain_all = 1.0 - self.val_mae_by_w[best_w] / max(self.val_mae_by_w[0.0], 1e-12)
                gain_last = 1.0 - per_fold[best_w][-1] / max(per_fold[0.0][-1], 1e-12)
                # must beat the plain trend by >= 5 % on average AND by >= 5 % in the most recent fold
                self.w = best_w if (gain_all >= 0.05 and gain_last >= 0.05) else 0.0
            except Exception:
                self.w = 0.0
        self.inner = self.make_inner()
        self.inner.fit(X, res, cycles)
        return self

    def predict(self, X: pd.DataFrame, base: np.ndarray, cycles: np.ndarray,
                scale: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict for ALL rows of X (a DL inner model needs the preceding rows as context)."""
        base = np.asarray(base, dtype=float)
        sc = np.ones_like(base) if scale is None else np.asarray(scale, dtype=float)
        r = self._inner_pred(self.inner, X, np.asarray(cycles, dtype=float))
        r = np.clip(r, -self.clip_sd * self.res_sd, self.clip_sd * self.res_sd)
        out = base + self.w * r * sc
        return out if self.lower is None else np.clip(out, self.lower, None)
