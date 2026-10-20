"""Hourly historical temperature from the free Open-Meteo archive API (no key needed)."""
from __future__ import annotations

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_temperature(lat: float, lon: float, start: str, end: str) -> pd.Series:
    """Hourly 2 m air temperature in degrees F, indexed by UTC hour."""
    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
        },
        timeout=60,
    )
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    idx = pd.to_datetime(hourly["time"], utc=True)
    return pd.Series(hourly["temperature_2m"], index=idx, name="temp_f", dtype=float)
