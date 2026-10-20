"""Address lookup and distance helpers. All free, no keys.

* U.S. Census geocoder: exact match for a full street address, and map clicks
  (coordinates -> state, ZIP code, county, town).
* OpenStreetMap Nominatim: partial searches like a street or town name.
  Its usage policy asks for an identifying User-Agent and light use (searches
  happen only when someone presses Enter).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

CENSUS_ADDRESS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
CENSUS_POINT_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
VA_VIEWBOX = "-83.7,39.5,-75.2,36.5"  # left, top, right, bottom
HEADERS = {"User-Agent": "Wattson/0.2 (open-source portfolio project)"}
EARTH_RADIUS_MI = 3958.8


@dataclass(frozen=True)
class Place:
    matched_address: str
    lat: float
    lon: float
    zip: str
    state: str


# ---------- Census: full street address ----------
def parse_census(payload: dict) -> Place | None:
    matches = payload.get("result", {}).get("addressMatches", [])
    if not matches:
        return None
    m = matches[0]
    comp = m.get("addressComponents", {})
    return Place(
        matched_address=m["matchedAddress"],
        lat=float(m["coordinates"]["y"]),
        lon=float(m["coordinates"]["x"]),
        zip=str(comp.get("zip", "")).zfill(5),
        state=comp.get("state", ""),
    )


def geocode(address: str) -> Place | None:
    resp = requests.get(CENSUS_ADDRESS_URL, timeout=30,
                        params={"address": address, "benchmark": "Public_AR_Current", "format": "json"})
    resp.raise_for_status()
    return parse_census(resp.json())


# ---------- Census: map click ----------
def _first(geos: dict, name_part: str) -> dict:
    for key, items in geos.items():
        if name_part.lower() in key.lower() and items:
            return items[0]
    return {}


def parse_census_point(payload: dict, lat: float, lon: float) -> Place | None:
    geos = payload.get("result", {}).get("geographies", {})
    state = _first(geos, "States").get("STUSAB", "")
    if not state:
        return None
    zcta = _first(geos, "ZIP Code").get("ZCTA5", "")
    county = _first(geos, "Counties").get("NAME", "")
    town = (_first(geos, "Incorporated Places").get("BASENAME")
            or _first(geos, "Census Designated Places").get("BASENAME")
            or _first(geos, "County Subdivisions").get("BASENAME") or "")
    where = ", ".join(p for p in (town, county) if p)
    label = f"Selected spot in {where}, {state} {zcta}".replace(" ,", ",").strip()
    return Place(label, lat, lon, zcta, state)


def reverse(lat: float, lon: float) -> Place | None:
    resp = requests.get(CENSUS_POINT_URL, timeout=30, params={
        "x": lon, "y": lat, "benchmark": "Public_AR_Current", "vintage": "Current_Current",
        "layers": "all", "format": "json"})
    resp.raise_for_status()
    return parse_census_point(resp.json(), lat, lon)


# ---------- Nominatim: partial search ----------
def parse_nominatim(item: dict) -> Place | None:
    addr = item.get("address", {})
    if addr.get("ISO3166-2-lvl4") != "US-VA" and addr.get("state") != "Virginia":
        return None
    street = " ".join(p for p in (addr.get("house_number"), addr.get("road")) if p)
    town = (addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet")
            or addr.get("suburb") or addr.get("county") or "")
    postcode = str(addr.get("postcode", "")).split("-")[0].strip()
    label = ", ".join(p for p in (street, town, f"VA {postcode}".strip()) if p)
    return Place(label, float(item["lat"]), float(item["lon"]),
                 postcode.zfill(5) if postcode.isdigit() else "", "VA")


def suggest(query: str, limit: int = 6) -> list[Place]:
    if len(query.strip()) < 3:
        return []
    resp = requests.get(NOMINATIM_URL, headers=HEADERS, timeout=15, params={
        "q": query, "format": "jsonv2", "addressdetails": 1, "countrycodes": "us",
        "viewbox": VA_VIEWBOX, "bounded": 1, "limit": limit})
    resp.raise_for_status()
    seen, out = set(), []
    for item in resp.json():
        p = parse_nominatim(item)
        if p and p.matched_address not in seen:
            seen.add(p.matched_address)
            out.append(p)
    return out


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles; accepts scalars or arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))
