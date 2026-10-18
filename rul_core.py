"""
rul_core.py
------------
Format-agnostic battery data ingestion + feature engineering + RUL (Remaining
Useful Life) label construction + ML/DL model wrappers.

Supports two real-world input layouts out of the box and is written so a
third layout can be added by writing one more `parse_*` function and adding
a branch to `detect_format`.

Layout A - "cycler" (e.g. Neway/BTS-style cell cycler export):
    One row per data point, columns include Cycle Index, Step Type,
    Current(A), Voltage(V), Capacity(Ah), Chg. Cap.(Ah), DChg. Cap.(Ah),
    Energy(Wh), Date, SOC/DOD(%) ...

Layout B - "bms_log" (decoded CAN / BMS signal log):
    One row PER SIGNAL, one column PER TIMESTAMP. A `Signal_Name` column
    identifies each row (Cumulative_total_voltage, Current, SOC,
    Remain_capacity_(mAH), BMS_life_(0~255_cycles), N_Cell_voltage,
    N_Cell_temperature, ...). Needs transposing before use.

Both layouts are normalised into one common **per-cycle** schema:

    cycle, capacity, soh, voltage_mean, voltage_max, voltage_min,
    voltage_delta, current_mean, current_max, temp_mean, temp_max,
    n_samples, source_format

which is what the feature engineering / labeling / modeling code below
consumes, regardless of which file format it came from.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import trend as tr

warnings.filterwarnings("ignore", category=RuntimeWarning)

STANDARD_COLUMNS = [
    "cycle", "capacity", "soh", "voltage_mean", "voltage_max", "voltage_min",
    "voltage_delta", "current_mean", "current_max", "temp_mean", "temp_max",
    "n_samples", "source_format",
]


# --------------------------------------------------------------------------
# 1. FORMAT DETECTION
# --------------------------------------------------------------------------

def load_workbook_sheets(file) -> dict[str, pd.DataFrame]:
    """Read every sheet of an uploaded .xlsx/.xls/.csv into a dict of DataFrames."""
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith(".csv"):
        return {"Sheet1": pd.read_csv(file)}
    xl = pd.ExcelFile(file)
    return {s: xl.parse(s) for s in xl.sheet_names}


def detect_format(sheets: dict[str, pd.DataFrame]) -> tuple[str, str]:
    """
    Inspect every sheet and decide which known layout it is.
    Returns (format_name, sheet_name_used).
    format_name is one of: 'cycler', 'bms_log', 'unknown'
    """
    for sheet_name, df in sheets.items():
        cols = {str(c).strip() for c in df.columns}

        # Layout B is unambiguous: a Signal_Name column is very distinctive.
        if "Signal_Name" in cols:
            return "bms_log", sheet_name

        # Layout A: standard cycler export column names.
        cycler_markers = {"Cycle Index", "Current(A)", "Voltage(V)"}
        if cycler_markers.issubset(cols):
            return "cycler", sheet_name

        # Looser fallback match for cycler-style exports with slightly
        # different column spellings/units.
        has_cycle = any(re.search(r"cycle", c, re.I) for c in cols)
        has_current = any(re.search(r"current", c, re.I) for c in cols)
        has_voltage = any(re.search(r"volt", c, re.I) for c in cols)
        if has_cycle and has_current and has_voltage:
            return "cycler", sheet_name

    return "unknown", next(iter(sheets))


# --------------------------------------------------------------------------
# 2. LAYOUT A — CYCLER PARSER
# --------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *patterns: str) -> Optional[str]:
    """Return the first column whose name matches any of the regex patterns."""
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, str(c), re.I):
                return c
    return None


def parse_cycler(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw cycler-export DataFrame into the standard per-cycle schema."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_cycle = _find_col(df, r"^cycle index$", r"cycle")
    col_current = _find_col(df, r"^current\(a\)$", r"current")
    col_voltage = _find_col(df, r"^voltage\(v\)$", r"^voltage")
    col_dchg_cap = _find_col(df, r"dchg\.?\s*cap", r"discharge.*cap")
    col_chg_cap = _find_col(df, r"^chg\.?\s*cap", r"charge.*cap")
    col_cap = _find_col(df, r"^capacity\(ah\)$", r"^capacity")
    col_temp = _find_col(df, r"temp")
    col_date = _find_col(df, r"^date$", r"time.*stamp")

    if col_cycle is None:
        raise ValueError("Could not find a Cycle Index column in this sheet.")

    df[col_cycle] = pd.to_numeric(df[col_cycle], errors="coerce")
    for c in [col_current, col_voltage, col_dchg_cap, col_chg_cap, col_cap, col_temp]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for cyc, g in df.groupby(col_cycle):
        if pd.isna(cyc):
            continue
        dchg = g[col_dchg_cap].max() if col_dchg_cap else np.nan
        chg = g[col_chg_cap].max() if col_chg_cap else np.nan
        cap = g[col_cap].max() if col_cap else np.nan
        # Prefer discharge capacity as the health indicator (standard practice);
        # fall back to generic capacity, then charge capacity.
        capacity = dchg if pd.notna(dchg) and dchg > 0 else (
            cap if pd.notna(cap) and cap > 0 else chg
        )
        row = {
            "cycle": int(cyc),
            "capacity": capacity,
            "voltage_mean": g[col_voltage].mean() if col_voltage else np.nan,
            "voltage_max": g[col_voltage].max() if col_voltage else np.nan,
            "voltage_min": g[col_voltage].min() if col_voltage else np.nan,
            "current_mean": g[col_current].mean() if col_current else np.nan,
            "current_max": g[col_current].abs().max() if col_current else np.nan,
            "temp_mean": g[col_temp].mean() if col_temp else np.nan,
            "temp_max": g[col_temp].max() if col_temp else np.nan,
            "n_samples": len(g),
            "timestamp": g[col_date].iloc[-1] if col_date else None,
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    out["voltage_delta"] = out["voltage_max"] - out["voltage_min"]
    out["source_format"] = "cycler"
    out["capacity_unit"] = "Ah"
    return out


# --------------------------------------------------------------------------
# 3. LAYOUT B — BMS SIGNAL LOG PARSER
# --------------------------------------------------------------------------

_TS_FORMATS = ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]


def _parse_timestamp_col(col_name: str) -> Optional[pd.Timestamp]:
    name = re.sub(r"\.\d+$", "", str(col_name))  # strip pandas de-dup suffix e.g. ".1"
    for fmt in _TS_FORMATS:
        try:
            return pd.to_datetime(name, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(name, errors="coerce")


def parse_bms_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn a raw signal-log DataFrame (Signal_Name rows x timestamp columns)
    into the standard per-cycle schema.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    meta_cols = [c for c in df.columns if c in ("#", "MsgID (Hex)", "Msg Type", "Signal_Name")]
    ts_cols = [c for c in df.columns if c not in meta_cols]
    if "Signal_Name" not in df.columns or not ts_cols:
        raise ValueError("Expected a 'Signal_Name' column plus one column per timestamp.")

    sig_names = df["Signal_Name"].astype(str).tolist()
    sub = df[ts_cols].copy()
    sub.index = sig_names

    # Transpose -> one row per timestamp, one column per signal
    t = sub.T
    t.index = [c for c in ts_cols]
    t = t.apply(pd.to_numeric, errors="coerce")
    t["_timestamp"] = [_parse_timestamp_col(c) for c in ts_cols]
    t = t.sort_values("_timestamp").reset_index(drop=True)

    def col(name):
        return t[name] if name in t.columns else pd.Series(np.nan, index=t.index)

    cellv_cols = [c for c in t.columns if re.match(r"^\d+_Cell_voltage$", str(c))]
    cellt_cols = [c for c in t.columns if re.match(r"^\d+_Cell_temperature$", str(c))]

    total_voltage = col("Cumulative_total_voltage")
    current = col("Current")
    soc = col("SOC")
    remain_capacity = col("Remain_capacity_(mAH)")
    cycle_raw = col("BMS_life_(0~255_cycles)")
    max_cell_v = col("Maximum_cell_voltage_value")
    min_cell_v = col("Minimum_cell_voltage_value")
    max_temp = col("Maximum_temperature_value")
    min_temp = col("Minimum_temperature_value")

    cell_v_mean = t[cellv_cols].mean(axis=1) if cellv_cols else pd.Series(np.nan, index=t.index)
    cell_t_mean = t[cellt_cols].mean(axis=1) if cellt_cols else pd.Series(np.nan, index=t.index)

    # "Remain_capacity" is the charge presently left in the pack (it tracks SOC,
    # not aging) so it is a poor health indicator on its own. Normalising by SOC
    # gives an *estimated full-pack capacity* that is roughly SOC-independent
    # and is what should actually drift downward as the pack degrades.
    soc_safe = soc.where(soc > 10)  # integer-quantised SOC at low values makes the ratio very noisy
    estimated_full_capacity = remain_capacity / (soc_safe / 100.0)

    long_df = pd.DataFrame({
        "timestamp": t["_timestamp"],
        "cycle": cycle_raw,
        "capacity": estimated_full_capacity,
        "remain_capacity_raw": remain_capacity,
        "soc": soc,
        "voltage_total": total_voltage,
        "voltage_max": max_cell_v.where(max_cell_v.notna(), cell_v_mean),
        "voltage_min": min_cell_v.where(min_cell_v.notna(), cell_v_mean),
        "current": current,
        "temp_max": max_temp.where(max_temp.notna(), cell_t_mean),
        "temp_mean": cell_t_mean,
    })
    long_df = long_df.dropna(subset=["cycle"])
    long_df["cycle"] = long_df["cycle"].round().astype(int)

    # Some BMS "life" counters wrap around at 255 -> unwrap so cycle count
    # keeps increasing monotonically instead of resetting to 0.
    long_df = long_df.sort_values("timestamp")
    wrapped = long_df["cycle"].diff() < -100
    long_df["cycle"] = long_df["cycle"] + 256 * wrapped.cumsum()

    rows = []
    for cyc, g in long_df.groupby("cycle"):
        cap = g["capacity"].median()      # median: robust to SOC-quantisation / CAN glitches
        if pd.isna(cap):
            cap = g["remain_capacity_raw"].mean()  # fallback if SOC was unusable
        rows.append({
            "cycle": int(cyc),
            "capacity": cap,
            "voltage_mean": g["voltage_total"].mean(),
            "voltage_max": g["voltage_max"].max(),
            "voltage_min": g["voltage_min"].min(),
            "current_mean": g["current"].mean(),
            "current_max": g["current"].abs().max(),
            "temp_mean": g["temp_mean"].mean(),
            "temp_max": g["temp_max"].max(),
            "n_samples": len(g),
            "timestamp": g["timestamp"].iloc[-1],
        })

    out = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    out["voltage_delta"] = out["voltage_max"] - out["voltage_min"]
    out["source_format"] = "bms_log"
    out["capacity_unit"] = "mAh"
    return out


# --------------------------------------------------------------------------
# 4. TOP-LEVEL INGEST
# --------------------------------------------------------------------------

@dataclass
class IngestResult:
    format_name: str
    sheet_name: str
    per_cycle: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def ingest(file) -> IngestResult:
    sheets = load_workbook_sheets(file)
    fmt, sheet_name = detect_format(sheets)
    msgs: list[str] = []

    if fmt == "cycler":
        per_cycle = parse_cycler(sheets[sheet_name])
    elif fmt == "bms_log":
        per_cycle = parse_bms_log(sheets[sheet_name])
    else:
        raise ValueError(
            "Could not recognise this file as either the cycler-export format "
            "(Cycle Index / Current(A) / Voltage(V) columns) or the BMS signal-log "
            "format (a Signal_Name column with one timestamp per remaining column). "
            "Add a new parse_* function in rul_core.py to support a new layout."
        )

    n_before = len(per_cycle)
    per_cycle = per_cycle.dropna(subset=["capacity"])
    per_cycle = per_cycle[per_cycle["capacity"] > 0]
    if len(per_cycle) < n_before:
        msgs.append(f"Dropped {n_before - len(per_cycle)} cycle(s) with missing/zero capacity.")

    return IngestResult(format_name=fmt, sheet_name=sheet_name, per_cycle=per_cycle, warnings=msgs)


# --------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------------------------------------

def format_capacity(value: float, unit: str = "") -> str:
    """Human-readable capacity (e.g. 48000 mAh -> '48.0 Ah (48,000 mAh)')."""
    if unit == "mAh":
        return f"{value / 1000:.3g} Ah ({value:,.0f} mAh)"
    return f"{value:.4g} {unit}".strip()


def estimate_rated_capacity(capacity: np.ndarray) -> float:
    """Reference (100 % SOH) capacity = robust level of the *early-life* curve, taken as the
    intercept of a line fitted to the first ~3 % of cycles (min 5) -> not thrown off by one noisy
    first cycle or by an initial settling cycle."""
    cap = np.asarray(capacity, dtype=float)
    cap = cap[np.isfinite(cap)]
    if len(cap) == 0:
        return float("nan")
    k = min(len(cap), max(5, int(round(0.03 * len(cap)))))
    head = cap[:k]
    if k < 5:
        return float(np.median(head))
    x = np.arange(k, dtype=float)
    slope, icpt = np.polyfit(x, head, 1)
    est = float(icpt)
    med = float(np.median(head))
    return est if abs(est - med) <= 0.03 * abs(med) else med      # fall back if the tiny fit is silly


def engineer_features(per_cycle: pd.DataFrame, rated_capacity: Optional[float] = None) -> pd.DataFrame:
    """Add SOH, capacity-fade, and rolling/trend features to the per-cycle table."""
    df = per_cycle.sort_values("cycle").reset_index(drop=True).copy()

    if rated_capacity is None or rated_capacity <= 0:
        rated_capacity = estimate_rated_capacity(df["capacity"].to_numpy())

    df["rated_capacity"] = rated_capacity
    df["soh"] = (df["capacity"] / rated_capacity).clip(upper=1.2)

    # Display / label curve: spike-robust, edge-safe, monotone (non-increasing) smoothing.
    df["soh_smooth"] = tr.monotone_smooth(df["soh"].to_numpy(dtype=float))
    df["soh_diff"] = df["soh_smooth"].diff().fillna(0.0)
    df["cum_cycles"] = df["cycle"] - df["cycle"].min()

    # CAUSAL (past-only) versions of the health indicator for model inputs: a centered smoother
    # would let the model peek at future cycles. EWMA + trailing windows only use <= current cycle.
    soh = df["soh"].astype(float)
    df["soh_causal"] = soh.rolling(window=5, min_periods=1).median().ewm(span=9, adjust=False).mean()
    df["soh_slope_w"] = (df["soh_causal"].shift(20) - df["soh_causal"]).div(20.0).fillna(0.0) * 100.0
    df["soh_slope_w"] = df["soh_slope_w"].where(df["cycle"] > df["cycle"].min() + 20, 0.0)

    for c in ["voltage_delta", "current_mean", "temp_mean"]:
        if c in df.columns:
            df[c + "_roll3"] = df[c].rolling(window=3, min_periods=1).mean()

    return df


# --------------------------------------------------------------------------
# 6. RUL LABEL CONSTRUCTION (health-indicator extrapolation)
# --------------------------------------------------------------------------

@dataclass
class RULLabelResult:
    labeled: pd.DataFrame
    eol_cycle: float
    fit_slope: float
    fit_intercept: float
    reached_eol_in_data: bool
    reliable: bool
    note: str
    trend_names: str = ""


def build_rul_labels(df: pd.DataFrame, eol_soh: float = 0.80, min_points: int = 5,
                      min_relative_fade: float = 0.005) -> RULLabelResult:
    """
    Historical failure/EOL events are not present in this data, so RUL labels are constructed the
    standard way used in the battery-PHM literature: take the cycle at which the (monotone-smoothed)
    SOH crosses `eol_soh`. If the file never reaches EOL, the crossing is extrapolated with an
    ensemble of fade curves (linear / quadratic / power / exponential / local fits) chosen by
    hold-out extrapolation error, instead of a single straight line (which is badly off for
    accelerating fade).  RUL(i) = N_eol - cycle(i).
    """
    df = df.copy()
    x = df["cycle"].to_numpy(dtype=float)
    y = df["soh_smooth"].to_numpy(dtype=float)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]

    reliable = True
    trend_names = ""
    if len(x) < min_points:
        note = (f"Only {len(x)} usable cycles found (need >= {min_points}) — "
                f"RUL labels below are a rough placeholder, not a reliable fit.")
        eol_cycle = x.max() + 50 if len(x) else 50.0
        slope, intercept = -1e-4, 1.0
        reliable = False
    else:
        slope, intercept = np.polyfit(x, y, 1)
        span = y.max() - y.min()
        if span < min_relative_fade:
            reliable = False
            eol_cycle = x.max() + 200
            note = (f"The health indicator only varies by {span*100:.2f}% across the whole file "
                    f"({len(x)} cycles) — there isn't enough observable degradation to fit a trend. "
                    f"The RUL numbers below are a placeholder, not a real estimate. Upload data that "
                    f"spans more of the cell's life, or a longer logging window, for a meaningful result.")
        else:
            reached_in_data = bool((y <= eol_soh).any())
            if reached_in_data:
                k = int(np.argmax(y <= eol_soh))
                if k > 0:      # interpolate inside the crossing step (sub-cycle accuracy)
                    y0, y1 = y[k - 1], y[k]
                    eol_cycle = float(x[k - 1] + (y0 - eol_soh) / max(y0 - y1, 1e-12) * (x[k] - x[k - 1]))
                else:
                    eol_cycle = float(x[0])
                note = "The observed data already reaches the EOL threshold; EOL cycle is the first observed crossing."
            else:
                fit = tr.fit_trend(x, y)
                trend_names = ", ".join(fit.names)
                e = fit.eol_cycle(eol_soh)
                if np.isfinite(e):
                    eol_cycle = max(e, x.max() + 1)
                    note = (f"EOL cycle is extrapolated from the observed degradation trend "
                            f"(ensemble: {trend_names}; the data does not yet reach the EOL threshold).")
                else:
                    reliable = False
                    eol_cycle = x.max() * 1.5
                    note = ("The fade trend never reaches the EOL threshold within a sensible horizon — "
                            "EOL cannot be projected reliably.")

    reached_in_data = bool((y <= eol_soh).any()) if len(y) else False

    out = df.copy()
    out["rul_label"] = (eol_cycle - out["cycle"]).clip(lower=0)
    return RULLabelResult(
        labeled=out, eol_cycle=float(eol_cycle), fit_slope=float(slope),
        fit_intercept=float(intercept), reached_eol_in_data=reached_in_data,
        reliable=reliable, note=note, trend_names=trend_names,
    )


def add_causal_rul_features(labeled: pd.DataFrame, eol_soh: float) -> pd.DataFrame:
    """Add features a deployed system could actually compute at each cycle (only past data):
    `eol_hat` (EOL cycle predicted from history so far), `rul_hat` = max(eol_hat - cycle, 0)."""
    out = labeled.copy()
    cyc = out["cycle"].to_numpy(dtype=float)
    eol_hat = tr.causal_eol_estimates(cyc, out["soh"].to_numpy(dtype=float), eol_soh)
    # before enough history exists, use the earliest available estimate
    first = np.where(np.isfinite(eol_hat))[0]
    if len(first):
        eol_hat[: first[0]] = eol_hat[first[0]]
    else:
        eol_hat = np.full(len(out), cyc[-1] * 1.5)
    out["eol_hat"] = eol_hat
    out["rul_hat"] = np.clip(eol_hat - cyc, 0, None)
    return out


# --------------------------------------------------------------------------
# 7. FEATURE MATRIX FOR MODELS
# --------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "cycle", "soh_causal", "soh_slope_w", "rul_hat", "voltage_delta", "voltage_mean",
    "current_mean", "current_max", "temp_mean",
]

SOH_DERIVED_FEATURES = {"soh_causal", "soh_slope_w", "soh_smooth", "soh_diff", "rul_hat", "eol_hat"}


def build_feature_matrix(labeled: pd.DataFrame, exclude_soh: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Feature matrix for the models (all CAUSAL: each row only uses data up to that cycle).

    exclude_soh=True drops every column derived from SOH (incl. the RUL-hat feature). Use it when SOH
    itself is the prediction target, otherwise the model just copies its own answer (leakage).
    """
    cols = [c for c in FEATURE_COLUMNS if c in labeled.columns and labeled[c].notna().any()]
    if exclude_soh:
        cols = [c for c in cols if c not in SOH_DERIVED_FEATURES]
    X = labeled[cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X, cols


# --------------------------------------------------------------------------
# 8. KNEE-POINT DETECTION
# --------------------------------------------------------------------------

@dataclass
class KneeResult:
    found: bool
    cycle: float
    soh: float                      # fraction (0-1) at the knee
    strength: float                 # slope ratio after/before the knee (>=1; bigger = sharper knee)
    cycles: np.ndarray              # x values for the arrays below
    diff_curve: np.ndarray          # 'distance above straight-line fade' curve (Kneedle-style, normalised) - for the chart
    rate_pct_per_100: np.ndarray    # RAW (unsmoothed) degradation rate: SOH %-points lost per 100 cycles
    note: str
    slope_before: float = float("nan")   # %-pts / 100 cycles
    slope_after: float = float("nan")
    sse_gain: float = float("nan")       # 1 - SSE(2 lines)/SSE(1 line)
    knee_score: float = float("nan")     # height of diff_curve at the knee cycle (for plotting the marker)


def _hinge_fit(x: np.ndarray, y: np.ndarray, xk: float) -> tuple[float, np.ndarray]:
    B = np.column_stack([np.ones_like(x), x - x[0], np.clip(x - xk, 0, None)])
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    r = y - B @ coef
    return float(r @ r), coef


def detect_knee(labeled: pd.DataFrame, min_points: int = 15, min_slope_ratio: float = 1.3,
                min_sse_gain: float = 0.5) -> KneeResult:
    """Find the capacity-fade knee with a continuous **two-segment (hinge) fit**.

    The previous 'distance above the start->end chord' score peaks near the middle of *any*
    gently convex curve, so it always reported a knee, and a shrinking-window smoother made the
    last ~5 % of the curve fall off. Here:
      * the curve is smoothed edge-safely (Savitzky-Golay with local polynomial edge fits),
      * a break-point is searched on 10-90 % of the life, fitting  y = a + b(x-x0) + c*max(0, x-xk),
      * a knee is only reported when the fade rate afterwards is >= `min_slope_ratio` x the rate
        before AND two lines explain >= `min_sse_gain` of the variance a single line misses.
    """
    x = labeled["cycle"].to_numpy(dtype=float)
    y = labeled["soh_smooth"].to_numpy(dtype=float)
    raw = labeled["soh"].to_numpy(dtype=float) if "soh" in labeled.columns else y.copy()
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(raw)
    x, y, raw = x[m], y[m], raw[m]
    empty = np.array([])

    def _none(msg: str) -> KneeResult:
        return KneeResult(False, float("nan"), float("nan"), 0.0, empty, empty, empty, msg)

    if len(x) < min_points:
        return _none(f"Knee not detected: need at least {min_points} cycles (found {len(x)}).")
    if x[-1] <= x[0] or (y[0] - y[-1]) <= 1e-4:
        return _none("Knee not detected: no measurable capacity fade across the file.")

    ys = tr.smooth_series(y, frac=0.05)
    # RAW degradation rate straight from the measured per-cycle SOH (deliberately NOT smoothed)
    rate = -np.gradient(raw, x) * 100.0 * 100.0                         # %-pts per 100 cycles
    n = len(x)
    # classic 'distance above the straight start->end fade line' curve (what the chart shows)
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (ys - ys[-1]) / (ys[0] - ys[-1])
    chord = yn - (1.0 - xn)

    # knee-score profile: quality of the two-line fit as a function of the break-point
    lo, hi = max(int(0.10 * n), 3), min(int(np.ceil(0.90 * n)), n - 3)
    if hi <= lo:
        return _none("Knee not detected: too few cycles for a reliable break-point search.")
    cand = np.arange(lo, hi)
    sse1, _ = _hinge_fit(x, ys, x[-1] + 1.0)                            # single straight line
    sse = np.array([_hinge_fit(x, ys, x[i])[0] for i in cand])
    gain = 1.0 - sse / max(sse1, 1e-18)
    score = np.full(n, np.nan)
    score[cand] = gain
    best = int(cand[int(np.argmin(sse))])
    sse2, coef = _hinge_fit(x, ys, x[best])
    s_before = -coef[1] * 100.0 * 100.0
    s_after = -(coef[1] + coef[2]) * 100.0 * 100.0
    ratio = s_after / s_before if s_before > 1e-9 else float("inf") if s_after > 0 else 0.0
    sse_gain = float(gain[best - lo])

    base = dict(cycles=x, diff_curve=chord, rate_pct_per_100=rate, knee_score=float(chord[best]), slope_before=float(s_before),
                slope_after=float(s_after), sse_gain=sse_gain)
    if not (ratio >= min_slope_ratio and sse_gain >= min_sse_gain and s_before > 0):
        return KneeResult(False, float("nan"), float("nan"), float(ratio) if np.isfinite(ratio) else 0.0,
                          note=(f"No clear knee: the fade rate only changes from {s_before:.2f} to "
                                f"{s_after:.2f} %-pts/100 cycles (x{ratio:.2f}) — gradual, near-steady fade."),
                          **base)
    strength_word = "sharp" if ratio >= 2.5 else ("moderate" if ratio >= 1.8 else "gentle")
    return KneeResult(True, float(x[best]), float(ys[best]), float(ratio),
                      note=(f"Knee detected at cycle {x[best]:.0f} (SOH {ys[best]*100:.1f}%): fade rate rises from "
                            f"{s_before:.2f} to {s_after:.2f} %-pts/100 cycles (x{ratio:.1f}, {strength_word})."),
                      **base)
