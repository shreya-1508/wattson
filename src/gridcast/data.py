"""Download, clean and cache one hourly table per area: demand, temperature, operator forecast."""
from __future__ import annotations

import pandas as pd

from . import eia, weather
from .config import AREAS, DATA_DIR, eia_api_key

COLUMNS = ["demand_mw", "operator_forecast_mw", "temp_f"]


def clean(df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """Regular hourly index, drop impossible values, fill only short gaps."""
    df = df.copy()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC", name="time_utc")
    df = df.reindex(full)
    d = df["demand_mw"]
    # EIA-930 has occasional zeros/spikes from reporting errors: treat as missing.
    med = d.rolling(24 * 7, center=True, min_periods=24).median()
    bad = (d <= 0) | (d > 3 * med) | (d < med / 3)
    df.loc[bad, "demand_mw"] = float("nan")
    df = df.interpolate(limit=max_gap_hours, limit_area="inside")
    return df


def build(area_code: str, start: str, end: str) -> pd.DataFrame:
    area = AREAS[area_code]
    key = eia_api_key()
    if area.kind == "subba":
        grid = eia.fetch_zone_demand(area.code, area.parent, start, end, key).to_frame()
        grid["operator_forecast_mw"] = float("nan")
    else:
        grid = eia.fetch_ba_demand_and_forecast(area.code, start, end, key)

    temps = [weather.fetch_temperature(lat, lon, start, end) for lat, lon in area.weather_points]
    temp = pd.concat(temps, axis=1).mean(axis=1).rename("temp_f")

    df = grid.join(temp, how="left")[COLUMNS]
    return clean(df)


def cache_path(area_code: str):
    return DATA_DIR / f"{area_code.lower()}_hourly.csv"


def save(df: pd.DataFrame, area_code: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path(area_code), index_label="time_utc")


def load(area_code: str) -> pd.DataFrame:
    path = cache_path(area_code)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m gridcast fetch --area {area_code}")
    df = pd.read_csv(path, index_col="time_utc")
    df.index = pd.to_datetime(df.index, utc=True)
    return df
