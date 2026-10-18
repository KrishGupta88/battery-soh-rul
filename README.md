# Battery SOH + RUL Predictor (Streamlit, ML + DL)

A Streamlit app that ingests battery cycling/telemetry data in either of two
real-world formats, auto-detects which one it is, engineers health-indicator
features, and predicts **Remaining Useful Life (RUL)** with both a classical
ML model (Random Forest) and a DL model (LSTM).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501) and
upload a file in the sidebar.

## Supported input formats (auto-detected)

**1. Cycler export** (e.g. Neware/BTS-style .xlsx) — one row per data point:
`Cycle Index`, `Step Type`, `Current(A)`, `Voltage(V)`, `Chg. Cap.(Ah)`,
`DChg. Cap.(Ah)`, `Energy(Wh)`, `Date`, `SOC/DOD(%)`, ...

**2. BMS signal log** — one row *per signal*, one column *per timestamp*,
identified by a `Signal_Name` column: `Cumulative_total_voltage`, `Current`,
`SOC`, `Remain_capacity_(mAH)`, `BMS_life_(0~255_cycles)`,
`N_Cell_voltage` (1–48), `N_Cell_temperature` (1–16), ...

Both are normalised into the same **per-cycle** table (cycle, capacity, SOH,
voltage stats, current stats, temperature stats) before anything else runs,
so the rest of the pipeline (features → labels → ML/DL) is format-agnostic.
To add a third format, write one more `parse_*` function in `rul_core.py`
that outputs that same schema and add a detection rule to `detect_format`.

## How RUL labels are built

True end-of-life/failure events usually aren't present in a data export, so
RUL training labels are constructed the way the battery-PHM literature does
it: fit a trend line to the smoothed State-of-Health (SOH) curve and
extrapolate to the cycle where it's projected to cross an EOL threshold
(default 80% of rated capacity). `RUL(i) = projected_EOL_cycle - cycle(i)`.

The app is deliberately honest about this: if a file doesn't show enough
observable capacity fade (e.g. a short logging window, or a signal that
tracks state-of-charge rather than aging — see `Remain_capacity_(mAH)` in the
BMS-log format), it flags the trend as **unreliable** instead of silently
reporting a confident-looking but meaningless number.

## Models

- **ML — `models.MLModel`**: `RandomForestRegressor` over per-cycle features
  (cycle index, smoothed SOH, SOH delta, voltage spread, mean/peak current,
  mean temperature). Shows feature importance.
- **DL — `models.DLModel`**: a small PyTorch LSTM over sliding windows of
  consecutive cycles (window length configurable in the sidebar), predicting
  RUL at the last step of each window. Shows the training loss curve.

Both use a **chronological** train/test split (train on the earlier cycles,
test on the most recent ones) rather than a random split, since shuffling
would leak future degradation information into training.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — wiring, plots, controls |
| `rul_core.py` | Format detection, parsers, feature engineering, RUL labeling (no Streamlit/model imports — unit-testable on its own) |
| `models.py` | `MLModel` (Random Forest) and `DLModel` (PyTorch LSTM), both with `.fit()` / `.predict()` / `.evaluate()` |
| `requirements.txt` | Dependencies |

## Notes / limitations

- If a file has very few cycles, the app tells you instead of training a
  model that would just be overfitting noise (minimum 6 cycles for ML, and
  the DL sequence window + 1 for the LSTM).
- The BMS-log parser derives an SOC-normalised capacity estimate
  (`remain_capacity / (SOC/100)`) as the health indicator, since raw
  `Remain_capacity_(mAH)` just tracks present charge level, not aging.
- Cell-level fields (voltage/temperature per cell) are aggregated to
  mean/min/max per timestamp; nothing about individual cell balancing is
  modeled — extend `parse_bms_log` in `rul_core.py` if you need that.
- If PyTorch isn't installed, the ML side still works; the app shows an
  install hint instead of crashing.

## SOH + RUL output (added)

The app now predicts **State of Health (SOH, %)** and **Remaining Useful Life (RUL, cycles)** together:

- Separate Random Forest and LSTM models for SOH. Their inputs exclude `soh_smooth` / `soh_diff` (`build_feature_matrix(..., exclude_soh=True)`) so SOH is not predicted from itself.
- "Latest-cycle SOH + RUL estimate" shows measured SOH, ML SOH, DL SOH, the RUL label, ML RUL and DL RUL.
- Predicted-SOH-vs-cycle plot, and the CSV export includes `soh_pct`, `ml_predicted_soh_pct`, `ml_predicted_rul`.
- Caveat: tree models cannot extrapolate beyond the SOH/RUL range seen in training, so when the test cycles are the most degraded ones the Random Forest tends to over-estimate SOH and RUL.

## Granular SOH axis + downloadable report

- Sidebar **SOH axis tick step (%)** (0.25–5) sets the gridline spacing on the SOH plots. The axis auto-zooms to the data range plus the EOL line and draws minor gridlines at 1/5 of the step. If the step would give more than ~80 ticks for the data range it is coarsened automatically; use the plot's zoom for finer detail.
- After **Run pipeline**, **Download complete report** produces one self-contained HTML file (`report.py`): summary cards, the three charts (interactive, plotly.js embedded so it works offline), latest-cycle SOH + RUL table, model accuracy, settings, and the per-cycle data. Open it in a browser; Print → Save as PDF for a PDF copy.

## Clean charts + instant report download (latest)

- `charts.py` holds all three figures (SOH trend, predicted SOH, predicted RUL) so the app and the report look identical: SOH in %, readable "nice" tick steps (max ~16), legend above the plot, EOL threshold + projected EOL as legend entries with one small callout, shaded end-of-life zone, dotted projection to the EOL cycle, latest-SOH marker.
- Sidebar **SOH axis tick step** now defaults to **Auto**; a fixed step is coarsened if it would crowd the axis.
- Right under the SOH chart: **Download SOH report (HTML)** and **Download SOH chart (interactive HTML)** — available immediately, no model training needed.
- After **Run pipeline**: **Download complete report** (primary button). Download buttons use `on_click="ignore"` so clicking them no longer reruns the app and wipes your results (needs Streamlit >= 1.43).
- Every chart has a camera icon to save a PNG; the report has a Print / Save as PDF button.

## v3 — accuracy overhaul (trend-hybrid models, honest evaluation, real knee detection)

**New files:** `trend.py` (edge-safe smoothing + extrapolating fade-curve ensemble + causal EOL estimates + rolling-origin validation) and `pipeline.py` (Streamlit-free orchestration of all models).

What was wrong in v2 and what changed:

| Problem seen in the report | Cause | Fix |
|---|---|---|
| Knee "at cycle 347" on an almost straight fade; knee score and degradation-rate curves fall off a cliff in the last ~5 % | Centered rolling mean with `min_periods=1` shrinks at the ends (biases the last points upward); the "distance above the chord" score peaks mid-curve for *any* gently convex curve, so a knee was always reported | Savitzky-Golay smoothing with local-polynomial edge fits + monotone (isotonic) projection; knee = two-segment (hinge) fit, reported only if the fade rate rises ≥ 1.3× and two lines beat one by ≥ 50 % of the residual variance; outer 3 % of the rate curve masked |
| RF flat-lined at 82.8 % SOH, R² −3 … −6.5 | Random Forest cannot extrapolate; the chronological test block is the most degraded part | Hybrid: `final = extrapolating fade-curve trend + w · RF/LSTM(residual)`; trend family chosen by hold-out *extrapolation* error; `w` chosen by forward-chained validation (w = 0 if the ML part doesn't help) |
| RUL model saw only `cycle`, centred-smoothed SOH (look-ahead) | Features used future points | All model features are causal (past-only); new causal `rul_hat` feature = EOL predicted from history so far |
| LSTM: full-batch, 150 epochs, no validation, first `window-1` test cycles dropped | – | Mini-batches, AdamW, grad-clip, early stopping, seed ensemble, length-preserving predictions with proper context |
| R² dominated by a tiny test variance | – | Report MAE / RMSE, *pre-EOL* MAE, skill vs a straight-line baseline, and a **rolling-origin EOL error** (predict EOL from 40 … 90 % of the history) |
| `Rated capacity used 4.8e+04` | no unit | `48 Ah (48,000 mAh)`; reference capacity = robust early-life intercept instead of mean of first 3 cycles |

Notes
* Once the measured SOH has crossed the EOL threshold RUL is exactly 0 by definition; the report therefore also lists MAE on cycles still *before* EOL.
* The "latest-cycle" table uses models trained on the training block only (held-out), so the comparison against the measured value is not in-sample.
* `streamlit>=1.43` still works; newer versions print a harmless `use_container_width` deprecation warning.
