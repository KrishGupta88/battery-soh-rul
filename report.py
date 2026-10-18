"""
report.py — builds a single self-contained HTML report (charts + metrics; no raw dataset)
from the results the Streamlit app already computed. Open it in any browser;
use the browser's Print -> "Save as PDF" if you want a PDF copy.
"""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import plotly.io as pio

from charts import PLOT_CONFIG

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2430}
.wrap{max-width:1100px;margin:0 auto;padding:28px 24px 60px}
h1{margin:0 0 4px;font-size:26px} h2{margin:34px 0 10px;font-size:19px;border-bottom:2px solid #e3e6ec;padding-bottom:6px}
.sub{color:#667085;font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0}
.card{background:#fff;border:1px solid #e3e6ec;border-radius:10px;padding:12px 14px}
.card .l{font-size:12px;color:#667085} .card .v{font-size:22px;font-weight:600;margin-top:2px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13px}
th,td{border:1px solid #e3e6ec;padding:6px 9px;text-align:right} th{background:#eef0f4}
td:first-child,th:first-child{text-align:left}
.note{background:#fff7e6;border:1px solid #ffd591;border-radius:8px;padding:10px 12px;margin:10px 0;font-size:13px}
.fig{background:#fff;border:1px solid #e3e6ec;border-radius:10px;padding:6px;margin:10px 0}
.scroll{max-height:420px;overflow:auto;border:1px solid #e3e6ec;border-radius:8px}
.btn{float:right;background:#2E86DE;color:#fff;border:0;border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer}
@media print{.scroll{max-height:none;overflow:visible} body{background:#fff} .btn{display:none} .fig{break-inside:avoid}}
"""


def _cards(items: list[tuple[str, str]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="l">{html.escape(l)}</div><div class="v">{html.escape(str(v))}</div></div>'
        for l, v in items) + "</div>"


def _table(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    return df.to_html(index=False, border=0, na_rep="–", float_format=lambda x: float_fmt.format(x),
                      escape=True)


def build_report_html(*, title: str, file_name: str, headline: list[tuple[str, str]],
                      settings: dict, notes: list[str], figures: list[tuple[str, object]],
                      tables: list[tuple[str, pd.DataFrame]], per_cycle: pd.DataFrame | None = None) -> str:
    """per_cycle is accepted for backward compatibility but is intentionally NOT written to the report."""
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{_CSS}</style></head><body><div class='wrap'>",
             "<button class='btn' onclick='window.print()'>Print / Save as PDF</button>",
             f"<h1>{html.escape(title)}</h1>",
             f"<div class='sub'>Source file: {html.escape(file_name)} · generated "
             f"{datetime.now():%Y-%m-%d %H:%M}</div>",
             "<h2>Summary</h2>", _cards(headline)]
    for n in notes:
        parts.append(f"<div class='note'>{html.escape(n)}</div>")

    first = True
    for fig_title, fig in figures:
        parts.append(f"<h2>{html.escape(fig_title)}</h2><div class='fig'>")
        # embed plotly.js once (offline-safe); later charts reuse it
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=True if first else False,
                              config=PLOT_CONFIG))
        parts.append("</div>")
        first = False

    for t_title, df in tables:
        parts.append(f"<h2>{html.escape(t_title)}</h2>{_table(df)}")

    parts.append("<h2>Settings used</h2>")
    parts.append(_table(pd.DataFrame({"Setting": list(settings), "Value": [str(v) for v in settings.values()]})))

    parts.append("</div></body></html>")
    return "".join(parts)
