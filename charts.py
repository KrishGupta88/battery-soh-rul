"""
charts.py — clean, consistent Plotly figures shared by the Streamlit app and the
downloadable HTML report.

Design rules
------------
* SOH is always plotted in PERCENT with a "%" suffix on the axis.
* Y-axis uses a readable "nice" tick step (max ~16 major ticks) instead of a
  tick on every 0.25-1 %, so labels never pile up. The sidebar step is honoured
  as long as it stays readable.
* Legend sits above the plot (horizontal) so it is never clipped on the right.
* EOL threshold / projected EOL are legend entries + one small callout, so text
  never overlaps the curve.
* Transparent backgrounds + neutral grid colours → looks right on both the
  Streamlit light/dark theme and on the white report page.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE = "#2E86DE"          # SOH (smoothed) line
MEASURED = "#FF8A00"      # measured SOH (per cycle) markers - warm orange, contrasts with BLUE
MEASURED_SOFT = "rgba(255,138,0,0.75)"
RED = "#E5484D"           # EOL threshold
ORANGE = "#F59F00"
TEAL = "#12B886"          # knee point
PURPLE = "#9C6ADE"        # projected EOL
GRAY = "#8B93A1"
GRID = "rgba(128,128,128,0.22)"
AXIS_LINE = "rgba(128,128,128,0.55)"

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "filename": "battery_chart", "scale": 3},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

_NICE_STEPS = (0.1, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100)


def nice_step(span: float, requested: float | None = None, max_ticks: int = 16) -> float:
    """Pick a tick step: the requested one if it gives <= max_ticks, else the next nice one."""
    span = max(float(span), 1e-9)
    need = span / max_ticks
    if requested and requested >= need:
        return float(requested)
    return float(next((s for s in _NICE_STEPS if s >= need), _NICE_STEPS[-1]))


def _base_layout(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int,
                 subtitle: str | None = None) -> go.Figure:
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:12px;opacity:0.7'>{subtitle}</span>"
    fig.update_layout(
        title=dict(text=text, x=0.0, xanchor="left", y=0.97, yanchor="top"),
        height=height,
        margin=dict(l=70, r=30, t=105, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12), itemsizing="constant"),
        font=dict(size=13),
    )
    fig.update_xaxes(title=dict(text=xtitle, standoff=10), showgrid=True, gridcolor=GRID,
                     zeroline=False, showline=True, linecolor=AXIS_LINE, ticks="outside",
                     showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor=AXIS_LINE, spikedash="dot")
    fig.update_yaxes(title=dict(text=ytitle, standoff=10), showgrid=True, gridcolor=GRID,
                     zeroline=False, showline=True, linecolor=AXIS_LINE, ticks="outside")
    return fig


def _style_soh_yaxis(fig: go.Figure, values_pct, eol_pct: float | None, tick_step_pct: float | None,
                     top_cap: float | None = None, floor_pct: float | None = None) -> None:
    vals = np.asarray([v for v in np.ravel(values_pct) if np.isfinite(v)], dtype=float)
    if eol_pct is not None:
        vals = np.append(vals, eol_pct)
    if vals.size == 0:
        return
    lo_v, hi_v = float(vals.min()), float(vals.max())
    step = nice_step(hi_v - lo_v, tick_step_pct)
    lo = np.floor((lo_v - 0.5 * step) / step) * step
    hi = np.ceil((hi_v + 0.5 * step) / step) * step
    if top_cap is not None:
        hi = max(hi, top_cap)
    if floor_pct is not None:                 # always show the axis down to (at least) the floor, e.g. 60 %
        lo = min(lo, float(floor_pct))
        step = nice_step(hi - lo, tick_step_pct)
        lo = np.floor(lo / step) * step
        hi = np.ceil(hi / step) * step
    if step >= 1:
        fmt = ".0f"
    elif abs(step * 10 - round(step * 10)) < 1e-9:
        fmt = ".1f"
    else:
        fmt = ".2f"
    fig.update_yaxes(range=[lo, hi], dtick=step, tick0=lo, ticksuffix="%", tickformat=fmt,
                     hoverformat=".2f")


# --------------------------------------------------------------------------
# 1. SOH vs cycle (the main degradation chart)
# --------------------------------------------------------------------------

def soh_trend_figure(labeled: pd.DataFrame, eol_soh: float, eol_cycle: float, reliable: bool,
                     tick_step_pct: float | None = None, height: int = 540,
                     knee=None, soh_floor_pct: float | None = 60.0) -> go.Figure:
    """labeled: columns cycle, soh, soh_smooth (fractions 0-1). eol_soh is a fraction (0.80)."""
    cyc = labeled["cycle"].to_numpy(float)
    raw = labeled["soh"].to_numpy(float) * 100
    smooth = labeled["soh_smooth"].to_numpy(float) * 100
    eol_pct = eol_soh * 100
    last_c, last_s = cyc[-1], smooth[-1]

    x_hi = max(cyc.max(), eol_cycle if reliable else cyc.max())
    x_pad = 0.04 * (x_hi - cyc.min() if x_hi > cyc.min() else 1)
    x_range = [max(0, cyc.min() - x_pad), x_hi + x_pad]

    fig = go.Figure()

    y_all = np.concatenate([raw, smooth])
    # reserve the region under the EOL line (light red "end-of-life zone")
    lo_guess = min(np.nanmin(y_all), eol_pct) - 2
    fig.add_shape(type="rect", xref="x", yref="y", x0=x_range[0], x1=x_range[1],
                  y0=min(lo_guess, eol_pct - 50, soh_floor_pct if soh_floor_pct is not None else 1e9), y1=eol_pct, fillcolor="rgba(229,72,77,0.07)",
                  line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=cyc, y=raw, mode="markers", name="Measured SOH (per cycle)",
        marker=dict(size=5, color=MEASURED_SOFT), hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Scatter(
        x=cyc, y=smooth, mode="lines", name="SOH (smoothed)",
        line=dict(width=3, color=BLUE, shape="spline", smoothing=0.6), hovertemplate="%{y:.2f}%"))

    # EOL threshold as a real trace (appears in the legend, no overlapping text)
    fig.add_trace(go.Scatter(
        x=x_range, y=[eol_pct, eol_pct], mode="lines", name=f"EOL threshold ({eol_pct:.0f}%)",
        line=dict(color=RED, width=1.8, dash="dash"), hoverinfo="skip"))

    subtitle = None
    if reliable:
        reached = bool((smooth <= eol_pct).any())
        if not reached and eol_cycle > last_c:        # dotted projection from last point to EOL
            fig.add_trace(go.Scatter(
                x=[last_c, eol_cycle], y=[last_s, eol_pct], mode="lines", name="Projected trend",
                line=dict(color=PURPLE, width=2, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[eol_cycle, eol_cycle], y=[eol_pct - 1000, eol_pct], mode="lines",
            name="Projected EOL cycle", line=dict(color=PURPLE, width=1.8, dash="dash"),
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[eol_cycle], y=[eol_pct], mode="markers", showlegend=False,
            marker=dict(size=12, color=PURPLE, symbol="diamond", line=dict(color="white", width=1.5)),
            hovertemplate=f"Projected EOL<br>cycle {eol_cycle:.0f} @ {eol_pct:.0f}%<extra></extra>"))
        fig.add_annotation(x=eol_cycle, y=eol_pct, text=f"<b>EOL ≈ cycle {eol_cycle:.0f}</b>",
                           showarrow=True, arrowhead=0, arrowcolor=PURPLE, ax=55, ay=-45,
                           font=dict(color="white", size=12), bgcolor="rgba(156,106,222,0.95)",
                           bordercolor=PURPLE, borderpad=4)
    else:
        subtitle = "Degradation trend flagged as unreliable — no projected EOL is shown."

    # knee point (where capacity fade starts to accelerate)
    if knee is not None and getattr(knee, "found", False):
        k_c, k_s = knee.cycle, knee.soh * 100
        y_bottom = (soh_floor_pct if soh_floor_pct is not None else min(np.nanmin(y_all), eol_pct) - 2)
        fig.add_trace(go.Scatter(
            x=[k_c, k_c], y=[y_bottom - 1000, k_s], mode="lines", name="Knee point",
            line=dict(color=TEAL, width=1.8, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[k_c], y=[k_s], mode="markers", showlegend=False,
            marker=dict(size=14, color=TEAL, symbol="star", line=dict(color="white", width=1.5)),
            hovertemplate=f"Knee point<br>cycle {k_c:.0f} @ {k_s:.1f}%<extra></extra>"))
        fig.add_annotation(x=k_c, y=k_s, text=f"<b>Knee ≈ cycle {k_c:.0f}</b>", showarrow=True,
                           arrowhead=0, arrowcolor=TEAL, ax=60, ay=-50,
                           font=dict(color="white", size=12), bgcolor="rgba(18,184,134,0.95)",
                           bordercolor=TEAL, borderpad=4)

    # latest measured point
    fig.add_trace(go.Scatter(
        x=[last_c], y=[last_s], mode="markers", showlegend=False,
        marker=dict(size=10, color=BLUE, line=dict(color="white", width=2)),
        hovertemplate=f"Latest: cycle {last_c:.0f}, {last_s:.2f}%<extra></extra>"))
    fig.add_annotation(x=last_c, y=last_s, text=f"Latest {last_s:.1f}%", showarrow=True,
                       arrowhead=0, arrowcolor=BLUE, ax=-20, ay=-38,
                       font=dict(size=12, color="#1f2430"), bgcolor="rgba(46,134,222,0.18)",
                       bordercolor=BLUE, borderpad=3)

    _base_layout(fig, "State of Health vs Cycle", "Cycle number", "State of Health (% of rated capacity)",
                 height, subtitle)
    fig.update_xaxes(range=x_range)
    _style_soh_yaxis(fig, y_all, eol_pct, tick_step_pct, top_cap=None, floor_pct=soh_floor_pct)
    return fig


# --------------------------------------------------------------------------
# 2. Predicted SOH (test region)
# --------------------------------------------------------------------------

def soh_prediction_figure(labeled: pd.DataFrame, ml_eval, dl_eval, dl_available: bool,
                          eol_soh: float, tick_step_pct: float | None = None,
                          height: int = 500, soh_floor_pct: float | None = 60.0,
                          trend_eval=None, baseline_eval=None) -> go.Figure:
    cyc = labeled["cycle"].to_numpy(float)
    meas = labeled["soh_smooth"].to_numpy(float) * 100
    eol_pct = eol_soh * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cyc, y=meas, mode="lines", name="Measured SOH (smoothed)",
                             line=dict(color=GRAY, width=2.5), hovertemplate="%{y:.2f}%"))
    vals = [meas]
    if baseline_eval is not None and len(baseline_eval.y_pred):
        fig.add_trace(go.Scatter(x=baseline_eval.cycles, y=np.asarray(baseline_eval.y_pred) * 100, mode="lines",
                                 name="Straight-line baseline", line=dict(color=GRAY, width=1.5, dash="dot"),
                                 hovertemplate="%{y:.2f}%"))
        vals.append(np.asarray(baseline_eval.y_pred) * 100)
    if trend_eval is not None and len(trend_eval.y_pred):
        fig.add_trace(go.Scatter(x=trend_eval.cycles, y=np.asarray(trend_eval.y_pred) * 100, mode="lines",
                                 name="Fade-curve trend (test)", line=dict(color=PURPLE, width=2, dash="dash"),
                                 hovertemplate="%{y:.2f}%"))
        vals.append(np.asarray(trend_eval.y_pred) * 100)
    fig.add_trace(go.Scatter(x=ml_eval.cycles, y=np.asarray(ml_eval.y_pred) * 100, mode="lines",
                             name="Random Forest hybrid (test)", line=dict(color=BLUE, width=2.5),
                             hovertemplate="%{y:.2f}%"))
    vals.append(np.asarray(ml_eval.y_pred) * 100)
    if dl_available and dl_eval is not None and len(dl_eval.y_pred):
        fig.add_trace(go.Scatter(x=dl_eval.cycles, y=np.asarray(dl_eval.y_pred) * 100, mode="lines",
                                 name="LSTM hybrid (test)", line=dict(color=TEAL, width=2.5),
                                 hovertemplate="%{y:.2f}%"))
        vals.append(np.asarray(dl_eval.y_pred) * 100)
    x_range = [cyc.min() - 0.02 * np.ptp(cyc), cyc.max() + 0.02 * np.ptp(cyc)]
    fig.add_trace(go.Scatter(x=x_range, y=[eol_pct, eol_pct], mode="lines", name=f"EOL threshold ({eol_pct:.0f}%)",
                             line=dict(color=RED, width=1.8, dash="dash"), hoverinfo="skip"))
    _base_layout(fig, "Predicted SOH vs Cycle (test region)", "Cycle number", "State of Health (%)", height)
    fig.update_xaxes(range=x_range)
    _style_soh_yaxis(fig, np.concatenate(vals), eol_pct, tick_step_pct, floor_pct=soh_floor_pct)
    return fig


# --------------------------------------------------------------------------
# 3. Predicted RUL (test region)
# --------------------------------------------------------------------------

def rul_prediction_figure(labeled: pd.DataFrame, ml_eval, dl_eval, dl_available: bool,
                          height: int = 460, trend_eval=None, baseline_eval=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labeled["cycle"], y=labeled["rul_label"], mode="lines",
                             name="RUL label (trend-based)", line=dict(color=GRAY, width=2.5),
                             hovertemplate="%{y:.0f} cycles"))
    if baseline_eval is not None and len(baseline_eval.y_pred):
        fig.add_trace(go.Scatter(x=baseline_eval.cycles, y=baseline_eval.y_pred, mode="lines",
                                 name="Straight-line baseline", line=dict(color=GRAY, width=1.5, dash="dot"),
                                 hovertemplate="%{y:.0f} cycles"))
    if trend_eval is not None and len(trend_eval.y_pred):
        fig.add_trace(go.Scatter(x=trend_eval.cycles, y=trend_eval.y_pred, mode="lines",
                                 name="Causal EOL-trend (test)", line=dict(color=PURPLE, width=2, dash="dash"),
                                 hovertemplate="%{y:.0f} cycles"))
    fig.add_trace(go.Scatter(x=ml_eval.cycles, y=ml_eval.y_pred, mode="lines",
                             name="Random Forest hybrid (test)", line=dict(color=BLUE, width=2.5),
                             hovertemplate="%{y:.0f} cycles"))
    if dl_available and dl_eval is not None and len(dl_eval.y_pred):
        fig.add_trace(go.Scatter(x=dl_eval.cycles, y=dl_eval.y_pred, mode="lines",
                                 name="LSTM hybrid (test)", line=dict(color=TEAL, width=2.5),
                                 hovertemplate="%{y:.0f} cycles"))
    _base_layout(fig, "Predicted RUL vs Cycle (test region)", "Cycle number",
                 "Remaining useful life (cycles)", height)
    fig.update_yaxes(rangemode="tozero", ticksuffix="")
    return fig


def rolling_eol_figure(rolling, height: int = 420) -> go.Figure:
    """Predicted EOL cycle vs how much history was available (rolling-origin validation)."""
    fig = go.Figure()
    t = rolling.table
    if t is None or len(t) == 0:
        _base_layout(fig, "EOL prediction vs amount of history", "Cutoff cycle (history available)",
                     "Predicted EOL cycle", height, subtitle=rolling.note or "Not available.")
        return fig
    true = rolling.true_eol
    fig.add_trace(go.Scatter(x=[t["cutoff_cycle"].min(), t["cutoff_cycle"].max()], y=[true, true], mode="lines",
                             name=f"Observed EOL (cycle {true:.0f})", line=dict(color=RED, width=1.8, dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t["cutoff_cycle"], y=t["pred_eol"], mode="lines+markers",
                             name="Predicted EOL from history up to cutoff",
                             line=dict(color=PURPLE, width=2.5), marker=dict(size=8, symbol="diamond"),
                             customdata=np.stack([t["err_cycles"], t["soh_at_cutoff_pct"]], axis=-1),
                             hovertemplate="cutoff %{x:.0f} (SOH %{customdata[1]:.1f}%)<br>predicted EOL %{y:.0f}"
                                           "<br>error %{customdata[0]:+.0f} cycles<extra></extra>"))
    _base_layout(fig, "EOL prediction vs amount of history", "Cutoff cycle (history available)",
                 "Predicted EOL cycle", height,
                 subtitle=f"Mean absolute EOL error over the cutoffs: {rolling.mae_cycles:.1f} cycles")
    return fig


# --------------------------------------------------------------------------
# 4. Knee curve (Kneedle difference curve + degradation rate)
# --------------------------------------------------------------------------

def knee_figure(knee, height: int = 460) -> go.Figure:
    """Knee analysis: break-point score (peaks at the knee) and degradation rate vs cycle."""
    fig = go.Figure()
    if knee is None or len(knee.cycles) == 0:
        _base_layout(fig, "Knee curve", "Cycle number", "Knee score", height,
                     subtitle=getattr(knee, "note", "Not enough data for knee analysis."))
        return fig

    fig.add_trace(go.Scatter(
        x=knee.cycles, y=knee.diff_curve, mode="lines", name="Knee curve (distance above straight-line fade)",
        line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(18,184,134,0.12)",
        hovertemplate="%{y:.3f}", yaxis="y"))
    fig.add_trace(go.Scatter(
        x=knee.cycles, y=knee.rate_pct_per_100, mode="lines", name="Degradation rate (SOH %-pts lost / 100 cycles)",
        line=dict(color=MEASURED, width=2, dash="dash"), hovertemplate="%{y:.2f} %-pts/100 cyc", yaxis="y2"))

    subtitle = knee.note
    if knee.found:
        fig.add_trace(go.Scatter(
            x=[knee.cycle, knee.cycle], y=[min(0, float(np.nanmin(knee.diff_curve))), knee.knee_score],
            mode="lines", name="Knee point", line=dict(color=TEAL, width=1.8, dash="dot"),
            hoverinfo="skip", yaxis="y"))
        fig.add_trace(go.Scatter(
            x=[knee.cycle], y=[knee.knee_score], mode="markers", showlegend=False, yaxis="y",
            marker=dict(size=14, color=TEAL, symbol="star", line=dict(color="white", width=1.5)),
            hovertemplate=f"Knee point<br>cycle {knee.cycle:.0f}<extra></extra>"))
        fig.add_annotation(x=knee.cycle, y=knee.knee_score, text=f"<b>Knee ≈ cycle {knee.cycle:.0f}</b>",
                           showarrow=True, arrowhead=0, arrowcolor=TEAL, ax=60, ay=-40,
                           font=dict(color="white", size=12), bgcolor="rgba(18,184,134,0.95)",
                           bordercolor=TEAL, borderpad=4)

    _base_layout(fig, "Knee curve", "Cycle number", "Knee score (normalised)", height, subtitle)
    fig.update_layout(yaxis2=dict(title=dict(text="Degradation rate (%-pts / 100 cycles)", standoff=10),
                                  overlaying="y", side="right", showgrid=False, zeroline=False,
                                  showline=True, linecolor=AXIS_LINE, ticks="outside", rangemode="tozero"),
                      margin=dict(l=70, r=80, t=105, b=60))
    return fig
