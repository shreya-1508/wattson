"""Leak-free features for a day-ahead (24 h+) demand forecast.

Rule: to predict hour t we may only use demand observed at or before t - 24h.
Temperature at hour t is used as a stand-in for the weather *forecast* that
would be available the day before (see README "Limitations").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOCAL_TZ

HORIZON = 24  # hours ahead

# Fixed-date and rule-based US federal holidays that move grid load.
def _holidays(years) -> set:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    cal = USFederalHolidayCalendar()
    days = cal.holidays(start=f"{min(years)}-01-01", end=f"{max(years)}-12-31")
    return set(days.date)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix aligned to df.index (target hour)."""
    d = df["demand_mw"]
    local = df.index.tz_convert(LOCAL_TZ)
    X = pd.DataFrame(index=df.index)

    # Calendar (local time drives human behaviour).
    X["hour"] = local.hour
    X["dow"] = local.dayofweek
    X["month"] = local.month
    X["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    hol = _holidays(sorted(set(local.year)))
    X["is_holiday"] = [int(day in hol) for day in local.date]
    X["is_weekend"] = (local.dayofweek >= 5).astype(int)

    # Weather.
    t = df["temp_f"]
    X["temp_f"] = t
    X["hdd"] = (60 - t).clip(lower=0)
    X["cdd"] = (t - 68).clip(lower=0)
    X["temp_mean_24h"] = t.rolling(24, min_periods=12).mean()  # building thermal lag

    # Lagged demand, never closer than the forecast horizon.
    for lag in (HORIZON, 48, 72, 168, 336):
        X[f"lag_{lag}h"] = d.shift(lag)
    shifted = d.shift(HORIZON)
    X["mean_24h_before"] = shifted.rolling(24, min_periods=20).mean()
    X["min_24h_before"] = shifted.rolling(24, min_periods=20).min()  # baseload signal
    X["mean_7d_before"] = shifted.rolling(168, min_periods=150).mean()
    return X


def feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(df)
    y = df["demand_mw"]
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]
