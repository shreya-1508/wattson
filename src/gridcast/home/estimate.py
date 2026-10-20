"""Glue: everything the app shows for one address, computed in plain Python (testable)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import outlook, rates
from .datacenters import nearby
from .recs import home_to_row
from .usage import UsageModel, monthly_kwh

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class HomeFacts:
    home_type: int
    sqft: float
    year_built: int
    heat_fuel: int
    water_fuel: int
    people: int


@dataclass
class Estimate:
    utility: str
    rate_now: float                 # $/kWh, effective average incl. fixed charges
    annual_kwh: dict[str, float]    # p10 / p50 / p90
    monthly: pd.DataFrame           # month, kwh, cost
    monthly_bill: dict[str, float]  # average monthly bill p10 / p50 / p90
    outlook: pd.DataFrame
    growth: dict[str, float]
    nearby_dcs: pd.DataFrame
    heat_share: float
    cool_share: float


def utilities_for_zip(utilities: pd.DataFrame, zip_code: str) -> pd.DataFrame:
    return utilities[utilities["zip"] == str(zip_code).zfill(5)].reset_index(drop=True)


def estimate(model: UsageModel, facts: HomeFacts, dd: pd.DataFrame, prices: pd.DataFrame,
             utility_name: str | None, res_rate_2024: float | None, dcs: pd.DataFrame,
             lat: float, lon: float, start_year: int, radius_mi: float = 10.0) -> Estimate:
    row = home_to_row(facts.home_type, facts.sqft, facts.year_built, facts.heat_fuel,
                      facts.water_fuel, facts.people, dd["hdd"].sum(), dd["cdd"].sum())
    p = model.predict_one(row)

    if res_rate_2024 is not None and res_rate_2024 > 0:
        rate = rates.utility_rate_now(res_rate_2024, prices)
        utility = utility_name or "Your utility"
    else:  # ZIP not in the NREL table: fall back to the statewide average
        rate = rates.current_va_price(prices) / 100
        utility = "Virginia average (utility not found for this ZIP)"

    kwh = monthly_kwh(p["p50"], p["heat_share"], p["cool_share"], dd)
    monthly = pd.DataFrame({"month": MONTHS, "kwh": kwh.to_numpy(), "cost": kwh.to_numpy() * rate})

    growth = outlook.scenario_rates(rates.annual_prices(prices))
    return Estimate(
        utility=utility,
        rate_now=rate,
        annual_kwh={k: p[k] for k in ("p10", "p50", "p90")},
        monthly=monthly,
        monthly_bill={k: p[k] * rate / 12 for k in ("p10", "p50", "p90")},
        outlook=outlook.project(rate, p["p50"], growth, start_year),
        growth=growth,
        nearby_dcs=nearby(dcs, lat, lon, radius_mi),
        heat_share=p["heat_share"],
        cool_share=p["cool_share"],
    )
