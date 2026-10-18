"""
app.py — Battery RUL Predictor (Streamlit)

Upload EITHER of two supported file formats and get BOTH a State-of-Health
(SOH) estimate and a Remaining-Useful-Life (RUL) estimate from a classical ML
model (Random Forest) and a DL model (LSTM):

  1. Cycler export  — Cycle Index / Step Type / Current(A) / Voltage(V) / ...
  2. BMS signal log — a Signal_Name column + one column per timestamp
     (Cumulative_total_voltage, Current, SOC, Remain_capacity_(mAH),
     BMS_life_(0~255_cycles), N_Cell_voltage, N_Cell_temperature, ...)

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts as ch
import models as md
import pipeline as pl
import report as rp
import rul_core as rc

st.set_page_config(page_title="Battery SOH + RUL Predictor", layout="wide", page_icon="🔋")

# --------------------------------------------------------------------------
# Sidebar — configuration
# --------------------------------------------------------------------------
st.sidebar.title("🔋 Battery SOH + RUL")
st.sidebar.caption("Works with cycler exports or BMS signal logs (.xlsx / .csv)")

uploaded = st.sidebar.file_uploader("Upload battery data", type=["xlsx", "xls", "csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Health-indicator settings")
eol_soh_pct = st.sidebar.slider("End-of-life SOH threshold (%)", 50, 95, 80, step=1,
                                 help="RUL is defined as cycles remaining until the "
                                      "capacity trend crosses this % of rated capacity.")
soh_tick_choice = st.sidebar.select_slider(
    "SOH axis tick step (%)", options=["Auto", 0.25, 0.5, 1, 2, 5], value="Auto",
    help="Spacing of the SOH axis gridlines. 'Auto' picks a clean step for the data range; "
         "a fixed step is coarsened automatically if it would crowd the axis (>16 ticks).")
soh_tick_pct = None if soh_tick_choice == "Auto" else float(soh_tick_choice)
soh_floor_pct = st.sidebar.slider(
    "SOH axis minimum (%)", 0, 90, 60, step=5,
    help="Lowest SOH shown on the SOH charts. 60 shows the full fade down to 60% of rated capacity "
         "(the axis extends further only if the data goes lower).")
rated_capacity_override = st.sidebar.number_input(
    "Rated capacity override (0 = auto-detect from first cycles)",
    min_value=0.0, value=0.0, step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Train / test split")
test_frac = st.sidebar.slider("Test fraction (most recent cycles)", 0.1, 0.4, 0.2, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("ML model (Random Forest)")
n_estimators = st.sidebar.slider("Number of trees", 50, 500, 300, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("DL model (LSTM)")
lstm_window = st.sidebar.slider("Sequence window (cycles)", 2, 30, 10)
lstm_hidden = st.sidebar.slider("Hidden units", 8, 128, 32, step=8)
lstm_epochs = st.sidebar.slider("Training epochs", 20, 400, 150, step=10)
lstm_lr = st.sidebar.select_slider("Learning rate", options=[0.001, 0.003, 0.005, 0.01, 0.03, 0.1], value=0.005)
lstm_members = st.sidebar.slider("Ensemble size (seeds averaged)", 1, 5, 3)

run_btn = st.sidebar.button("🚀 Run pipeline", type="primary", use_container_width=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def cap_label():
    unit = str(feats["capacity_unit"].iloc[0]) if "capacity_unit" in feats.columns else ""
    return rc.format_capacity(float(feats["rated_capacity"].iloc[0]), unit)


def metric_card(col, label, value, help_text=None):
    col.metric(label, value, help=help_text)


def plot_soh_curve(labeled, eol_soh, eol_cycle, reliable, knee=None):
    return ch.soh_trend_figure(labeled, eol_soh, eol_cycle, reliable, tick_step_pct=soh_tick_pct,
                               knee=knee, soh_floor_pct=soh_floor_pct)


def plot_rul_predictions(labeled, ml_eval, dl_eval, dl_available):
    return ch.rul_prediction_figure(labeled, ml_eval, dl_eval, dl_available)


def plot_soh_predictions(labeled, ml_soh_eval, dl_soh_eval, dl_available, eol_soh):
    return ch.soh_prediction_figure(labeled, ml_soh_eval, dl_soh_eval, dl_available, eol_soh,
                                    tick_step_pct=soh_tick_pct, soh_floor_pct=soh_floor_pct)


def download_button(label, data, file_name, mime, key=None, **kw):
    """Download button that does NOT rerun the script (so pipeline results stay on screen).
    Falls back to a normal button on older Streamlit versions."""
    try:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key,
                                  on_click="ignore", **kw)
    except TypeError:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, **kw)


def trend_report_html(fig, fig_knee=None):
    """Report available right after upload (no model training needed)."""
    cur_rul = f"{max(lab.eol_cycle - float(lab.labeled['cycle'].iloc[-1]), 0):.0f} cycles" if lab.reliable else "n/a"
    head = [
        ("Detected format", fmt_label),
        ("Cycles in file", str(len(lab.labeled))),
        ("Rated capacity used", cap_label()),
        ("Latest SOH (smoothed)", f"{lab.labeled['soh_smooth'].iloc[-1]:.1%}"),
        (f"EOL threshold / projected EOL cycle", f"{eol_soh_pct}% / " + (f"{lab.eol_cycle:.0f}" if lab.reliable else "n/a")),
        ("Knee point", f"cycle {knee.cycle:.0f} @ {knee.soh:.1%}" if knee.found else "no clear knee"),
        ("Current RUL (trend-based)", cur_rul),
    ]
    notes = ([lab.note] if lab.note else []) + list(ingest_res.warnings)
    notes.append("Trend-only report: ML / LSTM models were not run. Click 'Run pipeline' for the complete report.")
    return rp.build_report_html(
        title="Battery SOH report", file_name=getattr(uploaded, "name", "uploaded file"),
        headline=head, settings={"EOL SOH threshold (%)": eol_soh_pct,
                                 "Rated capacity override": rated_capacity_override or "auto"},
        notes=notes, figures=[("State of Health vs cycle", fig)] + ([("Knee curve", fig_knee)] if fig_knee is not None else []),
        tables=[])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

st.title("🔋 Battery State of Health (SOH) + Remaining Useful Life (RUL)")
st.caption(
    "Auto-detects the input format, engineers health-indicator features, builds "
    "RUL training labels from the capacity-fade trend, then trains a Random Forest "
    "(ML) and an LSTM (DL) to predict both State of Health (SOH, %) and "
    "Remaining Useful Life (RUL, cycles)."
)

if uploaded is None:
    st.info("⬅️ Upload a file in the sidebar to get started. Both a cycler export "
            "(Cycle Index / Current(A) / Voltage(V) columns) and a BMS signal log "
            "(Signal_Name + timestamp columns) are supported and auto-detected.")
    with st.expander("What do the two supported formats look like?"):
        st.markdown("""
**1. Cycler export** — one row per data point:

| Cycle Index | Step Type | Current(A) | Voltage(V) | Chg. Cap.(Ah) | DChg. Cap.(Ah) | ... |
|---|---|---|---|---|---|---|

**2. BMS signal log** — one row per signal, one column per timestamp:

| Signal_Name | 13.02.2025 17:22:51 | 13.02.2025 17:22:41 | ... |
|---|---|---|---|
| Cumulative_total_voltage | 48 | 48 | ... |
| Current | 0 | 0 | ... |
| SOC | 50 | 50 | ... |
| BMS_life_(0~255_cycles) | 64 | 52 | ... |
| 1_Cell_voltage | 3193 | 3193 | ... |
        """)
    st.stop()

try:
    ingest_res = rc.ingest(uploaded)
except Exception as e:
    st.error(f"Couldn't parse this file: {e}")
    st.stop()

fmt_label = {"cycler": "Cycler export", "bms_log": "BMS signal log"}.get(ingest_res.format_name, "Unknown")
c1, c2, c3 = st.columns(3)
metric_card(c1, "Detected format", fmt_label)
metric_card(c2, "Sheet used", ingest_res.sheet_name)
metric_card(c3, "Cycles found", len(ingest_res.per_cycle))
for w in ingest_res.warnings:
    st.warning(w)

with st.expander("Raw per-cycle data (parsed)"):
    st.dataframe(ingest_res.per_cycle, use_container_width=True)

feats = rc.engineer_features(ingest_res.per_cycle, rated_capacity=rated_capacity_override or None)
lab = rc.build_rul_labels(feats, eol_soh=eol_soh_pct / 100.0)
knee = rc.detect_knee(lab.labeled)

st.markdown("---")
st.subheader("📉 Degradation trend")
col_a, col_b = st.columns([2, 1])
with col_a:
    fig_trend = plot_soh_curve(lab.labeled, eol_soh_pct / 100.0, lab.eol_cycle, lab.reliable, knee=knee)
    st.plotly_chart(fig_trend, use_container_width=True, config=ch.PLOT_CONFIG)
    fig_knee = ch.knee_figure(knee)
    st.plotly_chart(fig_knee, use_container_width=True, config=ch.PLOT_CONFIG)
    d1, d2 = st.columns(2)
    with d1:
        download_button("📄 Download SOH report (HTML)", trend_report_html(fig_trend, fig_knee),
                        "battery_soh_report.html", "text/html", key="dl_trend_report",
                        use_container_width=True)
    with d2:
        download_button("📈 Download SOH chart (interactive HTML)",
                        fig_trend.to_html(full_html=True, include_plotlyjs=True, config=ch.PLOT_CONFIG),
                        "soh_vs_cycle.html", "text/html", key="dl_trend_chart", use_container_width=True)
with col_b:
    st.metric("Rated capacity used", cap_label())
    st.metric("Latest SOH", f"{lab.labeled['soh_smooth'].iloc[-1]:.1%}")
    st.metric("Projected EOL cycle", f"{lab.eol_cycle:.0f}" if lab.reliable else "n/a")
    st.metric("Knee point", f"cycle {knee.cycle:.0f}" if knee.found else "not detected",
              help=knee.note + " The knee is where capacity fade clearly starts to accelerate; a gentle, steady fade is reported as 'not detected'.")
    _last_cycle = float(lab.labeled["cycle"].iloc[-1])
    st.metric("Current RUL (trend-based)",
              f"{max(lab.eol_cycle - _last_cycle, 0):.0f} cycles" if lab.reliable else "n/a",
              help="Cycles left until the SOH trend crosses the EOL threshold, "
                   "counted from the last cycle in the file. 0 = EOL already reached.")
    if lab.reliable:
        st.success("Degradation trend looks usable for RUL modeling.")
    else:
        st.warning(lab.note)

if not run_btn:
    st.info("Adjust settings in the sidebar if needed, then click **Run pipeline** to train the ML and DL models.")
    st.stop()

n_cycles = len(lab.labeled)
MIN_CYCLES_FOR_ML = 30
if n_cycles < MIN_CYCLES_FOR_ML:
    st.error(
        f"Only {n_cycles} cycle(s) available — at least {MIN_CYCLES_FOR_ML} are needed to build causal features, "
        f"train and validate on a held-out split. Upload a file covering more cycles."
    )
    st.stop()

with st.spinner("Training models (trend + Random Forest" + (" + LSTM" if md.TORCH_AVAILABLE else "") + ")..."):
    try:
        res = pl.run_pipeline(lab.labeled, eol_soh_pct / 100.0, test_frac=test_frac, n_estimators=n_estimators,
                              lstm_window=lstm_window, lstm_hidden=lstm_hidden, lstm_epochs=lstm_epochs,
                              lstm_lr=lstm_lr, use_dl=True, dl_models=lstm_members)
    except Exception as e:  # keep the app alive and explain
        st.error(f"Model pipeline failed: {e}")
        st.stop()

cycles = res.cycles
tr_idx, te_idx = res.tr_idx, res.te_idx
y = lab.labeled["rul_label"].to_numpy(dtype=float)
dl_on = res.soh.dl is not None and res.rul.dl is not None and len(res.rul.dl.y_pred) > 0

st.caption(f"Chronological split: training on cycles "
           f"{cycles[tr_idx[0]]:.0f}–{cycles[tr_idx[-1]]:.0f} ({len(tr_idx)} pts), "
           f"testing on cycles {cycles[te_idx[0]]:.0f}–{cycles[te_idx[-1]]:.0f} ({len(te_idx)} pts).")
st.info(
    "**How the models work.** Tree models and LSTMs cannot extrapolate beyond what they saw in training, and the "
    "test block is by definition the *most degraded* part of the file. So each model here is a **hybrid**: an "
    "extrapolation-safe fade-curve trend (selected by hold-out extrapolation error"
    + (f": {res.trend_names}" if res.trend_names else "") + ") plus a Random Forest / LSTM that learns only the "
    "*residual*. A residual model is switched on only if it beats the plain trend in forward-chained validation "
    f"(Random Forest weight: SOH {res.soh.ml_w:g}, RUL {res.rul.ml_w:g}"
    + (f"; LSTM weight: SOH {res.soh.dl_w:g}, RUL {res.rul.dl_w:g}" if dl_on else "") + ")."
)
for n_ in res.notes:
    st.warning(n_)
if not md.TORCH_AVAILABLE:
    st.error("PyTorch isn't installed in this environment. Run `pip install torch` and restart the app to enable the LSTM.")


def _metrics_row(ev, mult, unit, label=""):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"MAE ({unit})", f"{ev.mae * mult:.2f}")
    c2.metric(f"RMSE ({unit})", f"{ev.rmse * mult:.2f}")
    c3.metric("R²", f"{ev.r2:.2f}",
              help="R² is unreliable on a short, low-variance test block (a tiny bias gives a very negative R²). "
                   "Prefer MAE and the skill score.")
    c4.metric("Skill vs straight line", f"{ev.skill:+.0%}" if np.isfinite(ev.skill) else "n/a",
              help="1 − MAE / MAE of a straight-line extrapolation of the training fade. > 0 means better than a ruler.")


st.markdown("---")
st.subheader("🤖 RUL models (held-out test cycles)")
rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("#### Random Forest hybrid — RUL")
    _metrics_row(res.rul.ml, 1.0, "cycles")
    st.markdown("**Feature importance (residual model)**")
    st.bar_chart(res.ml_rul_model.inner.feature_importance())
with rc2:
    st.markdown("#### LSTM hybrid — RUL")
    if dl_on:
        _metrics_row(res.rul.dl, 1.0, "cycles")
        st.markdown("**Training loss (mean of ensemble)**")
        st.line_chart(pd.Series(res.loss_rul, name="loss"))
    else:
        st.caption("LSTM not available.")
st.caption(f"Plain causal-trend RUL (no ML): MAE {res.rul.trend.mae:.2f} cycles · straight-line baseline: "
           f"MAE {res.rul.baseline.mae:.2f} cycles"
           + (f" · MAE on pre-EOL cycles only — RF {res.rul.ml.mae_pre_eol:.2f}" if np.isfinite(res.rul.ml.mae_pre_eol) else ""))

st.markdown("---")
st.subheader("🩺 State of Health (SOH) models (held-out test cycles)")
st.caption("SOH models use only cycle / voltage / current / temperature features plus the fade-curve trend "
           "(SOH-derived columns are excluded from the learned part so the model can't copy its target).")
sc1, sc2 = st.columns(2)
with sc1:
    st.markdown("#### Random Forest hybrid — SOH")
    _metrics_row(res.soh.ml, 100.0, "SOH %-pts")
with sc2:
    st.markdown("#### LSTM hybrid — SOH")
    if dl_on:
        _metrics_row(res.soh.dl, 100.0, "SOH %-pts")
    else:
        st.caption("LSTM not available.")
st.caption(f"Plain fade-curve trend (no ML): MAE {res.soh.trend.mae * 100:.2f} %-pts · straight-line baseline: "
           f"MAE {res.soh.baseline.mae * 100:.2f} %-pts")

if not lab.reliable:
    st.warning("Heads up: the degradation trend above was flagged as unreliable, so "
               "the metrics/predictions here are illustrating the pipeline mechanics "
               "rather than giving a trustworthy RUL estimate.")

st.markdown("---")
st.subheader("🎯 How early can the EOL be predicted? (rolling-origin validation)")
fig_roll = ch.rolling_eol_figure(res.rolling)
if len(res.rolling.table):
    st.plotly_chart(fig_roll, use_container_width=True, config=ch.PLOT_CONFIG)
    st.dataframe(res.rolling.table.drop(columns=["true_eol"]).rename(columns={
        "cutoff_frac": "life fraction", "cutoff_cycle": "cutoff cycle", "soh_at_cutoff_pct": "SOH at cutoff (%)",
        "pred_eol": "predicted EOL cycle", "err_cycles": "error (cycles)", "err_pct_of_life": "error (% of life)",
        "model": "curves used"}), use_container_width=True, hide_index=True)
else:
    st.info(res.rolling.note)

st.markdown("---")
st.subheader("📈 Predicted RUL vs cycle (test region)")
fig_rul = ch.rul_prediction_figure(lab.labeled, res.rul.ml, res.rul.dl, dl_on,
                                   trend_eval=res.rul.trend, baseline_eval=res.rul.baseline)
st.plotly_chart(fig_rul, use_container_width=True, config=ch.PLOT_CONFIG)

st.markdown("---")
st.subheader("📈 Predicted SOH vs cycle (test region)")
fig_soh_pred = ch.soh_prediction_figure(lab.labeled, res.soh.ml, res.soh.dl, dl_on, eol_soh_pct / 100.0,
                                        tick_step_pct=soh_tick_pct, soh_floor_pct=soh_floor_pct,
                                        trend_eval=res.soh.trend, baseline_eval=res.soh.baseline)
st.plotly_chart(fig_soh_pred, use_container_width=True, config=ch.PLOT_CONFIG)

st.markdown("---")
st.subheader("🔮 Latest-cycle SOH + RUL estimate")
measured_soh = float(lab.labeled["soh_smooth"].iloc[-1])
st.caption(f"Cycle {int(cycles[-1])} · end-of-life threshold {eol_soh_pct}% SOH · model values come from the models "
           f"trained on the training block only (held-out), not refit on the latest cycles.")
r1 = st.columns(3)
r1[0].metric("SOH (measured, smoothed)", f"{measured_soh:.1%}")
r1[1].metric("SOH — ML predicted", f"{res.soh.latest_ml:.1%}")
r1[2].metric("SOH — DL predicted", f"{res.soh.latest_dl:.1%}" if res.soh.latest_dl is not None else "n/a")
r2 = st.columns(3)
r2[0].metric("RUL (trend-based label)", f"{float(y[-1]):.0f} cycles" if lab.reliable else "n/a")
r2[1].metric("RUL — ML predicted", f"{res.rul.latest_ml:.0f} cycles")
r2[2].metric("RUL — DL predicted", f"{res.rul.latest_dl:.0f} cycles" if res.rul.latest_dl is not None else "n/a")

summary = pd.DataFrame({
    "Quantity": ["SOH (%)", "RUL (cycles)"],
    "Measured / label": [round(measured_soh * 100, 2), round(float(y[-1]), 1) if lab.reliable else None],
    "Trend (no ML)": [round(res.soh.latest_trend * 100, 2), round(res.rul.latest_trend, 1)],
    "ML (Random Forest hybrid)": [round(res.soh.latest_ml * 100, 2), round(res.rul.latest_ml, 1)],
    "DL (LSTM hybrid)": [round(res.soh.latest_dl * 100, 2) if res.soh.latest_dl is not None else None,
                         round(res.rul.latest_dl, 1) if res.rul.latest_dl is not None else None],
})
st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("⬇️ Download results")
export_df = lab.labeled.copy()
for col_ in ("ml_predicted_rul", "ml_predicted_soh", "trend_predicted_rul", "trend_predicted_soh"):
    export_df[col_] = np.nan
test_mask = export_df["cycle"].isin(cycles[te_idx])
export_df.loc[test_mask, "ml_predicted_rul"] = np.asarray(res.rul.ml.y_pred).tolist()
export_df.loc[test_mask, "ml_predicted_soh"] = np.asarray(res.soh.ml.y_pred).tolist()
export_df.loc[test_mask, "trend_predicted_rul"] = np.asarray(res.rul.trend.y_pred).tolist()
export_df.loc[test_mask, "trend_predicted_soh"] = np.asarray(res.soh.trend.y_pred).tolist()
if dl_on:
    export_df["dl_predicted_rul"] = np.nan
    export_df["dl_predicted_soh"] = np.nan
    export_df.loc[test_mask, "dl_predicted_rul"] = np.asarray(res.rul.dl.y_pred).tolist()
    export_df.loc[test_mask, "dl_predicted_soh"] = np.asarray(res.soh.dl.y_pred).tolist()
export_df["ml_predicted_soh_pct"] = export_df["ml_predicted_soh"] * 100
export_df["soh_pct"] = export_df["soh_smooth"] * 100
download_button(
    "Download per-cycle SOH + RUL: features, labels and predictions (CSV)",
    export_df.to_csv(index=False).encode("utf-8"), "battery_rul_results.csv", "text/csv", key="dl_csv",
)

# ----- Complete report (HTML: charts + metrics + settings; no raw dataset) -----
def _eval_rows():
    rows = []
    for target, unit, mult, tres in (("RUL", "cycles", 1.0, res.rul), ("SOH", "%-pts", 100.0, res.soh)):
        for name, ev in (("Straight-line baseline", tres.baseline), ("Fade-curve trend (no ML)", tres.trend),
                         ("Random Forest hybrid", tres.ml), ("LSTM hybrid", tres.dl)):
            if ev is None or len(ev.y_pred) == 0:
                rows.append({"Target": target, "Model": name, "MAE": None, "RMSE": None, "MAE pre-EOL": None,
                             "R²": None, "Skill vs line": None, "Unit": unit})
            else:
                rows.append({"Target": target, "Model": name, "MAE": ev.mae * mult, "RMSE": ev.rmse * mult,
                             "MAE pre-EOL": ev.mae_pre_eol * mult if np.isfinite(ev.mae_pre_eol) else None,
                             "R²": ev.r2, "Skill vs line": ev.skill if np.isfinite(ev.skill) else None, "Unit": unit})
    return pd.DataFrame(rows)


_head = [
    ("Detected format", fmt_label),
    ("Cycles in file", str(n_cycles)),
    ("Rated capacity used", cap_label()),
    ("Latest SOH (measured)", f"{measured_soh:.1%}"),
    ("SOH — ML / DL predicted", f"{res.soh.latest_ml:.1%} / " + (f"{res.soh.latest_dl:.1%}" if res.soh.latest_dl is not None else "n/a")),
    (f"EOL threshold / projected EOL cycle", f"{eol_soh_pct}% / " + (f"{lab.eol_cycle:.0f}" if lab.reliable else "n/a")),
    ("Knee point", f"cycle {knee.cycle:.0f} @ {knee.soh:.1%}" if knee.found else "no clear knee"),
    ("Current RUL (trend-based)", f"{max(lab.eol_cycle - float(cycles[-1]), 0):.0f} cycles" if lab.reliable else "n/a"),
    ("RUL — ML / DL predicted", f"{res.rul.latest_ml:.0f} / " + (f"{res.rul.latest_dl:.0f}" if res.rul.latest_dl is not None else "n/a") + " cycles"),
    ("Mean EOL error, rolling-origin", f"{res.rolling.mae_cycles:.1f} cycles" if np.isfinite(res.rolling.mae_cycles) else "n/a"),
]
_notes = [lab.note] if lab.note else []
_notes += list(ingest_res.warnings)
_notes.append(knee.note)
if not lab.reliable:
    _notes.append("The degradation trend was flagged as unreliable: treat predictions as pipeline illustration, not a trustworthy estimate.")
_notes.append("Models are hybrids: an extrapolation-safe fade-curve trend (" + res.trend_names + ") plus a Random Forest / "
              "LSTM that learns only the residual; a residual model is used only if it beats the plain trend in forward-chained validation. "
              "Once the measured SOH has crossed the EOL threshold, RUL is exactly 0 by definition, so the table also lists MAE on pre-EOL cycles only.")
_notes += list(res.notes)

_settings = {
    "EOL SOH threshold (%)": eol_soh_pct, "SOH axis tick step (%)": soh_tick_choice,
    "SOH axis minimum (%)": soh_floor_pct,
    "Rated capacity override": rated_capacity_override or "auto",
    "Test fraction": test_frac, "RF trees": n_estimators,
    "LSTM window / hidden / epochs / lr / ensemble": f"{lstm_window} / {lstm_hidden} / {lstm_epochs} / {lstm_lr} / {lstm_members}",
    "Residual blend weights (RF SOH / RUL; LSTM SOH / RUL)": f"{res.soh.ml_w:g} / {res.rul.ml_w:g}; {res.soh.dl_w:g} / {res.rul.dl_w:g}",
}
_figs = [("State of Health vs cycle", fig_trend), ("Knee curve", fig_knee), ("Predicted SOH vs cycle (test region)", fig_soh_pred),
         ("Predicted RUL vs cycle (test region)", fig_rul)]
if len(res.rolling.table):
    _figs.append(("EOL prediction vs amount of history", fig_roll))
_tables = [("Latest-cycle SOH + RUL", summary), ("Model accuracy on the held-out test cycles", _eval_rows())]
if len(res.rolling.table):
    _tables.append(("Rolling-origin EOL validation", res.rolling.table.drop(columns=["true_eol"])))

report_html = rp.build_report_html(
    title="Battery SOH + RUL report", file_name=getattr(uploaded, "name", "uploaded file"),
    headline=_head, settings=_settings, notes=_notes, figures=_figs, tables=_tables,
)
download_button(
    "📄 Download complete report (HTML — open in browser, Print → Save as PDF)",
    report_html.encode("utf-8"), "battery_soh_rul_report.html", "text/html", key="dl_full_report",
    type="primary",
)
