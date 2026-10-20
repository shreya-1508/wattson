"""Local heating/cooling degree days (base 65 F) from the free Open-Meteo archive.

Degree days measure how much heating or cooling a climate needs. They're the
same units the EIA survey uses (HDD65/CDD65), so the usage model can be
trained nationally and applied at any Virginia address.
"""
from __future__ import annotations

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def degree_days_from_daily(temps_f: pd.Series) -> pd.DataFrame:
    """Average degree days per calendar month over the years in `temps_f`."""
    df = pd.DataFrame({"t": temps_f})
    df["hdd"] = (65 - df["t"]).clip(lower=0)
    df["cdd"] = (df["t"] - 65).clip(lower=0)
    df["year"] = df.index.year
    df["month"] = df.index.month
    monthly = df.groupby(["year", "month"])[["hdd", "cdd"]].sum()
    return monthly.groupby("month").mean()  # 12 rows: typical month


def fetch_degree_days(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": "temperature_2m_mean",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        },
        timeout=60,
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]
    temps = pd.Series(daily["temperature_2m_mean"], index=pd.to_datetime(daily["time"]), dtype=float)
    return degree_days_from_daily(temps.dropna())
