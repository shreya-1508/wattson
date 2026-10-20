"""Realistic synthetic grid data for tests, CI and offline demos.

NOT real data. Never report metrics from synthetic runs as project results.

The generator mimics a zone like Dominion: weather-driven residential load
(heating + cooling), daily and weekly shape, holidays, plus a flat 24/7
"data center" baseload that grows steadily year over year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOCAL_TZ


def make_dataset(start: str = "2022-01-01", end: str = "2026-06-30", seed: int = 7,
                 dc_base_mw: float = 2500.0, dc_growth_per_year: float = 0.18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, f"{end} 23:00", freq="h", tz="UTC")
    local = idx.tz_convert(LOCAL_TZ)
    n = len(idx)
    doy = local.dayofyear.to_numpy()
    hour = local.hour.to_numpy()

    # Temperature: seasonal + daily cycle + persistent weather noise.
    seasonal = 57 - 22 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    diurnal = 8 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather = np.convolve(rng.normal(0, 4, n + 72), np.ones(72) / 72 * 6, mode="valid")[:n]
    temp = seasonal + diurnal + weather

    # Human load: daily profile, weekend dip, heating/cooling.
    profile = 1 + 0.18 * np.sin(2 * np.pi * (hour - 11) / 24) + 0.06 * np.sin(4 * np.pi * (hour - 6) / 24)
    weekend = np.where(local.dayofweek.to_numpy() >= 5, 0.92, 1.0)
    hdd = np.clip(60 - temp, 0, None)
    cdd = np.clip(temp - 68, 0, None)
    human = (9000 * profile * weekend) + 110 * hdd + 160 * cdd ** 1.15

    # Data centers: flat around the clock, compounding growth.
    years = (idx - idx[0]).total_seconds().to_numpy() / (365.25 * 86400)
    dc = dc_base_mw * (1 + dc_growth_per_year) ** years

    demand = human + dc + rng.normal(0, 180, n)

    # Operator-style forecast: decent but imperfect.
    operator = demand * (1 + rng.normal(0, 0.035, n)) + rng.normal(0, 150, n)

    return pd.DataFrame(
        {"demand_mw": demand, "operator_forecast_mw": operator, "temp_f": temp},
        index=pd.DatetimeIndex(idx, name="time_utc"),
    )
