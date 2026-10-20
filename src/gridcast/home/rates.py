"""Electricity prices: utility-by-ZIP rates (NREL) and Virginia price history (EIA)."""
from __future__ import annotations

import pandas as pd

from .. import eia

NREL_URLS = {
    "iou": "https://data.openei.org/files/8563/iou_zipcodes_2024.csv",
    "non_iou": "https://data.openei.org/files/8563/non_iou_zipcodes_2024.csv",
}
NREL_RATE_YEAR = 2024


def load_va_utilities(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine NREL files, keep Virginia, one row per (zip, utility)."""
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[df["state"].str.upper() == "VA"].copy()
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df = df[df["res_rate"] > 0]
    return (df.groupby(["zip", "utility_name"], as_index=False)
              .agg(eiaid=("eiaid", "first"), res_rate=("res_rate", "mean")))


def fetch_va_price_history(api_key: str, state: str = "VA") -> pd.DataFrame:
    """Average residential price for a state (or "US"), cents/kWh: annual and monthly."""
    out = {}
    for freq in ("annual", "monthly"):
        params = {
            "frequency": freq,
            "data[0]": "price",
            "facets[stateid][]": [state],
            "facets[sectorid][]": ["RES"],
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        raw = eia._fetch_all("electricity/retail-sales", params, api_key)
        raw = raw.rename(columns={"price": "cents_per_kwh"})
        raw["cents_per_kwh"] = pd.to_numeric(raw["cents_per_kwh"], errors="coerce")
        raw["frequency"] = freq
        out[freq] = raw[["period", "frequency", "cents_per_kwh"]].dropna()
    return pd.concat(out.values(), ignore_index=True)


def current_va_price(history: pd.DataFrame) -> float:
    """Average of the latest 12 monthly prices (cents/kWh): smooths out seasonality."""
    m = history[history["frequency"] == "monthly"].sort_values("period")
    return float(m["cents_per_kwh"].tail(12).mean())


def annual_prices(history: pd.DataFrame) -> pd.Series:
    a = history[history["frequency"] == "annual"].copy()
    a["year"] = a["period"].astype(str).str[:4].astype(int)
    return a.set_index("year")["cents_per_kwh"].sort_index()


def utility_rate_now(res_rate_2024: float, history: pd.DataFrame) -> float:
    """Bring a utility's 2024 average rate ($/kWh) up to date using the statewide trend."""
    annual = annual_prices(history)
    base = annual.get(NREL_RATE_YEAR)
    if base is None or base <= 0:
        return res_rate_2024
    return res_rate_2024 * current_va_price(history) / base
