"""Client for the EIA Open Data API v2 (Hourly Electric Grid Monitor, form EIA-930).

Two datasets are used:
  * electricity/rto/region-sub-ba-data  -> hourly demand for a utility zone (e.g. DOM)
  * electricity/rto/region-data         -> hourly demand (D) and the grid operator's
                                           day-ahead demand forecast (DF) for a whole
                                           balancing authority (e.g. PJM)
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://api.eia.gov/v2"
PAGE_SIZE = 5000  # EIA's per-request row cap


def _fetch_all(route: str, params: dict, api_key: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Page through an EIA v2 data route and return every row as a DataFrame."""
    session = session or requests.Session()
    rows: list[dict] = []
    offset = 0
    while True:
        query = {**params, "api_key": api_key, "offset": offset, "length": PAGE_SIZE}
        attempts = 5
        for attempt in range(attempts):
            try:
                resp = session.get(f"{BASE_URL}/{route}/data/", params=query, timeout=120)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == attempts - 1:
                    raise
                time.sleep(2 ** attempt)  # EIA is occasionally slow: wait and retry
                continue
            if resp.status_code < 500 and resp.status_code != 429:
                break
            time.sleep(2 ** attempt)  # back off on rate limits / server errors
        resp.raise_for_status()
        payload = resp.json()["response"]
        batch = payload.get("data", [])
        rows.extend(batch)
        total = int(payload.get("total", len(rows)))
        offset += len(batch)
        if not batch or offset >= total:
            break
    return pd.DataFrame(rows)


def _facet_params(facets: dict[str, Iterable[str]]) -> dict:
    return {f"facets[{k}][]": list(v) for k, v in facets.items()}


def _tidy(df: pd.DataFrame, value_name: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, name=value_name)
    ts = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(), index=ts, name=value_name)
    return s[~s.index.duplicated()].sort_index()


def fetch_zone_demand(zone_code: str, parent: str, start: str, end: str, api_key: str) -> pd.Series:
    """Hourly demand (MWh) for a sub-BA zone, indexed by UTC hour."""
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "start": f"{start}T00",
        "end": f"{end}T23",
        **_facet_params({"parent": [parent], "subba": [zone_code]}),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    }
    return _tidy(_fetch_all("electricity/rto/region-sub-ba-data", params, api_key), "demand_mw")


def fetch_ba_demand_and_forecast(respondent: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Hourly actual demand and the operator's day-ahead forecast for a balancing authority."""
    out = {}
    for code, name in [("D", "demand_mw"), ("DF", "operator_forecast_mw")]:
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "start": f"{start}T00",
            "end": f"{end}T23",
            **_facet_params({"respondent": [respondent], "type": [code]}),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        out[name] = _tidy(_fetch_all("electricity/rto/region-data", params, api_key), name)
    return pd.DataFrame(out)
