"""Data center locations in Virginia from OpenStreetMap (community-mapped; incomplete)."""
from __future__ import annotations

import pandas as pd
import requests

from .geo import haversine_miles

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
QUERY = """
[out:json][timeout:180];
area["ISO3166-2"="US-VA"][admin_level=4]->.va;
(
  nwr["telecom"="data_center"](area.va);
  nwr["building"="data_center"](area.va);
);
out center tags;
"""


def parse_overpass(payload: dict) -> pd.DataFrame:
    rows = []
    for el in payload.get("elements", []):
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id": f"{el['type']}/{el['id']}",
            "name": tags.get("name", "Unnamed data center"),
            "operator": tags.get("operator", ""),
            "lat": float(lat),
            "lon": float(lon),
        })
    df = pd.DataFrame(rows, columns=["osm_id", "name", "operator", "lat", "lon"])
    # A building and its site are often both tagged: merge points within ~100 m.
    df["key"] = df["lat"].round(3).astype(str) + "," + df["lon"].round(3).astype(str)
    return df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)


def fetch_va_datacenters() -> pd.DataFrame:
    resp = requests.post(OVERPASS_URL, data={"data": QUERY}, timeout=240,
                         headers={"User-Agent": "gridcast-portfolio-project"})
    resp.raise_for_status()
    return parse_overpass(resp.json())


def nearby(dcs: pd.DataFrame, lat: float, lon: float, radius_mi: float) -> pd.DataFrame:
    out = dcs.copy()
    out["miles"] = haversine_miles(lat, lon, out["lat"].to_numpy(), out["lon"].to_numpy())
    return out[out["miles"] <= radius_mi].sort_values("miles").reset_index(drop=True)
