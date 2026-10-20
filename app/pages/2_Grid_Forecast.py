"""Grid forecast page (part of: streamlit run app/Home.py)

Reads the CSVs written by `python -m gridcast backtest`. Falls back to the
synthetic demo (clearly labeled) when no real results exist yet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gridcast.config import AREAS, LOCAL_TZ, REPORTS_DIR  # noqa: E402

st.set_page_config(page_title="Wattson: Grid Forecast", page_icon="⚡", layout="wide")

available = [a for a in ("dom", "pjm", "synthetic") if (REPORTS_DIR / f"{a}_summary.json").exists()]
if not available:
    st.error("No results yet. Run `python -m gridcast demo` (synthetic) or fetch + backtest real data.")
    st.stop()

labels = {"dom": AREAS["DOM"].name, "pjm": AREAS["PJM"].name, "synthetic": "Synthetic demo data"}
tag = st.sidebar.selectbox("Grid area", available, format_func=labels.get)

st.title("Forecasting the grid in Data Center Alley")
st.caption("Day-ahead hourly electricity demand, with a data-center baseload growth signal. "
           "Source: EIA Hourly Electric Grid Monitor + Open-Meteo.")
if tag == "synthetic":
    st.warning("Synthetic data: these numbers demonstrate the pipeline, not real-world accuracy.")

summary = json.loads((REPORTS_DIR / f"{tag}_summary.json").read_text())
scores = pd.DataFrame(summary["scores"]).set_index("model")
preds = pd.read_csv(REPORTS_DIR / f"{tag}_predictions.csv", index_col="time_utc")
preds.index = pd.to_datetime(preds.index, utc=True).tz_convert(LOCAL_TZ).tz_localize(None)
trend = pd.read_csv(REPORTS_DIR / f"{tag}_baseload.csv", index_col="year")

# Headline numbers.
gbm = scores.loc["gbm"]
ref = "operator" if "operator" in scores.index else "seasonal_naive"
ref_name = "grid operator's forecast" if ref == "operator" else "same-hour-last-week baseline"
improvement = (1 - gbm["mape_pct"] / scores.loc[ref, "mape_pct"]) * 100
c1, c2, c3 = st.columns(3)
c1.metric("Model error (MAPE)", f"{gbm['mape_pct']:.2f}%")
c2.metric(f"Error reduction vs {ref_name}", f"{improvement:.0f}%")
c3.metric("Overnight baseload growth", f"{summary['baseload_growth_pct_per_year']:.1f}% / yr")

st.subheader("Forecast vs actual")
days = st.slider("Days to show", 3, 30, 7)
window = preds.iloc[-24 * days:]
cols = ["actual", "gbm"] + [c for c in ("operator", "seasonal_naive") if c in window]
st.line_chart(window[cols], y_label="MW")

st.subheader("Model comparison (walk-forward backtest)")
st.dataframe(scores.style.format({"mape_pct": "{:.2f}", "peak_mape_pct": "{:.2f}", "mae_mw": "{:,.0f}"}))

st.subheader("The data center signal: overnight baseload by year")
st.caption("Median 2-5 am demand on mild-weather nights. Homes and offices are quiet then; "
           "data centers are not, so growth here points to 24/7 load.")
st.bar_chart(trend["median_night_min_mw"], y_label="MW")
st.dataframe(trend.style.format({"median_night_min_mw": "{:,.0f}", "yoy_growth_pct": "{:+.1f}%"}))
