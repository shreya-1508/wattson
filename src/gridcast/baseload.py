"""Data center signal: growth in overnight baseload.

Homes and offices use little power at 2-5 am; data centers run flat 24/7.
So the rise in the overnight minimum, after removing weather, is a
transparent proxy for data-center load growth. We only use mild-weather
nights (little heating or cooling) so the trend is not a weather artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOCAL_TZ


def nightly_baseload(df: pd.DataFrame, mild_low: float = 55, mild_high: float = 70) -> pd.DataFrame:
    local = df.index.tz_convert(LOCAL_TZ)
    night = df[(local.hour >= 2) & (local.hour <= 5)].copy()
    night["date"] = night.index.tz_convert(LOCAL_TZ).date
    daily = night.groupby("date").agg(min_mw=("demand_mw", "min"), temp_f=("temp_f", "mean"))
    daily.index = pd.to_datetime(daily.index)
    return daily[(daily["temp_f"] >= mild_low) & (daily["temp_f"] <= mild_high)].dropna()


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Median mild-night baseload per year and year-over-year growth."""
    nb = nightly_baseload(df)
    yearly = nb.groupby(nb.index.year)["min_mw"].median().rename("median_night_min_mw").to_frame()
    yearly["yoy_growth_pct"] = yearly["median_night_min_mw"].pct_change() * 100
    yearly["mild_nights"] = nb.groupby(nb.index.year).size()
    yearly.index.name = "year"
    return yearly


def annual_growth_rate(df: pd.DataFrame) -> float:
    """Compound annual growth of baseload, from a log-linear fit over all mild nights."""
    nb = nightly_baseload(df)
    years = (nb.index - nb.index[0]).days / 365.25
    slope = np.polyfit(years, np.log(nb["min_mw"]), 1)[0]
    return float(np.expm1(slope) * 100)
