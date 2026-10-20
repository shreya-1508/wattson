"""EIA 2020 Residential Energy Consumption Survey (RECS) public microdata.

About 18,500 U.S. homes with measured annual electricity use and home details.
The codes below follow the RECS 2020 codebook. `validate` sanity-checks them
against the real data at setup time, so a code mismatch is caught loudly
instead of silently producing wrong estimates.
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

RECS_URL = "https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv"

REQUIRED = ["KWH", "KWHSPH", "KWHCOL", "TOTSQFT_EN", "TYPEHUQ", "YEARMADERANGE",
            "FUELHEAT", "FUELH2O", "NHSLDMEM", "HDD65", "CDD65", "NWEIGHT", "state_postal"]

# User-facing labels -> RECS codes.
HOME_TYPES = {
    "Single-family house (detached)": 2,
    "Townhouse / attached house": 3,
    "Apartment (building with 5+ units)": 5,
    "Apartment (building with 2-4 units)": 4,
    "Mobile home": 1,
}
YEAR_BUILT = {
    "Before 1950": 1, "1950s": 2, "1960s": 3, "1970s": 4, "1980s": 5,
    "1990s": 6, "2000s": 7, "2010-2015": 8, "2016 or later": 9,
}
FUELS = {"Electricity": 5, "Natural gas": 1, "Propane": 2, "Fuel oil": 3}
ELECTRIC, GAS = 5, 1


def validate(df: pd.DataFrame) -> list[str]:
    """Raise if columns are missing; return warnings if codes look wrong."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        hints = {m: difflib.get_close_matches(m, df.columns, n=3) for m in missing}
        raise ValueError(f"RECS file is missing columns {missing}. Closest matches: {hints}")

    warnings = []
    kwh = df["KWH"]
    if not 3000 < kwh.median() < 20000:
        warnings.append(f"Median KWH is {kwh.median():,.0f}; expected roughly 7,000-12,000.")
    by_fuel = df.groupby("FUELHEAT")["KWH"].median()
    if ELECTRIC in by_fuel and GAS in by_fuel and by_fuel[ELECTRIC] <= by_fuel[GAS]:
        warnings.append("Electric-heated homes don't use more electricity than gas-heated ones: "
                        "FUELHEAT codes may differ from the expected 5=electricity, 1=natural gas.")
    by_type = df.groupby("TYPEHUQ")["TOTSQFT_EN"].median()
    if 2 in by_type and 5 in by_type and by_type[2] <= by_type[5]:
        warnings.append("Detached houses aren't larger than big apartments: TYPEHUQ codes may differ.")
    if not df["YEARMADERANGE"].between(1, 9).all():
        warnings.append(f"YEARMADERANGE has values outside 1-9: {sorted(df['YEARMADERANGE'].unique())}")
    return warnings


def _code(s: pd.Series) -> pd.Series:
    """Map survey codes to small non-negative ints for the categorical model."""
    s = s.astype(int)
    return s.where(s >= 0, 0).replace(99, 10).clip(0, 250)


def to_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "sqft": df["TOTSQFT_EN"].astype(float),
        "people": df["NHSLDMEM"].astype(float),
        "year_built": df["YEARMADERANGE"].astype(float),
        "hdd": df["HDD65"].astype(float),
        "cdd": df["CDD65"].astype(float),
        "home_type": _code(df["TYPEHUQ"]),
        "heat_fuel": _code(df["FUELHEAT"]),
        "water_fuel": _code(df["FUELH2O"]),
    }, index=df.index)


CATEGORICAL = ["home_type", "heat_fuel", "water_fuel"]


def home_to_row(home_type: int, sqft: float, year_built: int, heat_fuel: int, water_fuel: int,
                people: int, hdd: float, cdd: float) -> pd.DataFrame:
    """One user-described home in the same shape as the training data."""
    raw = pd.DataFrame([{
        "TOTSQFT_EN": sqft, "NHSLDMEM": people, "YEARMADERANGE": year_built,
        "HDD65": hdd, "CDD65": cdd, "TYPEHUQ": home_type, "FUELHEAT": heat_fuel, "FUELH2O": water_fuel,
    }])
    return to_features(raw)


def make_synthetic(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    """Fake RECS-shaped data for tests (relationships are plausible, not real)."""
    rng = np.random.default_rng(seed)
    home_type = rng.choice([1, 2, 3, 4, 5], n, p=[0.06, 0.6, 0.08, 0.08, 0.18])
    sqft = np.where(home_type == 2, rng.normal(2200, 700, n), rng.normal(1000, 300, n)).clip(400, 6000)
    heat = rng.choice([1, 2, 3, 5, 99, -2], n, p=[0.45, 0.05, 0.04, 0.42, 0.02, 0.02])
    water = np.where(heat == 5, 5, rng.choice([1, 5], n))
    hdd = rng.uniform(500, 7000, n)
    cdd = rng.uniform(300, 3500, n)
    people = rng.integers(1, 6, n)
    year = rng.integers(1, 10, n)
    sph = np.where(heat == 5, sqft * hdd * 0.0012, 300)
    col = sqft * cdd * 0.0009
    base = 2500 + 900 * people + sqft * 1.2 + np.where(water == 5, 2200, 0) - year * 60
    kwh = (base + sph + col) * rng.lognormal(0, 0.25, n)
    return pd.DataFrame({
        "KWH": kwh, "KWHSPH": sph, "KWHCOL": col, "TOTSQFT_EN": sqft, "TYPEHUQ": home_type,
        "YEARMADERANGE": year, "FUELHEAT": heat, "FUELH2O": water, "NHSLDMEM": people,
        "HDD65": hdd, "CDD65": cdd, "NWEIGHT": rng.uniform(500, 30000, n),
        "state_postal": rng.choice(["VA", "TX", "NY", "CA"], n),
    })
