"""Future cost scenarios.

These extend Virginia's own price history forward. They are scenarios, not
predictions: real prices depend on rate cases, fuel costs and policy (for
example, Virginia's new rate class for large data centers starting in 2027).
"""
from __future__ import annotations

import pandas as pd


def cagr(series: pd.Series, years: int) -> float:
    """Compound annual growth over the last `years` complete years, as a fraction."""
    s = series.dropna().sort_index()
    if len(s) < 2:
        return 0.0
    years = min(years, len(s) - 1)
    end, start = s.iloc[-1], s.iloc[-1 - years]
    return float((end / start) ** (1 / years) - 1)


def scenario_rates(annual_prices: pd.Series) -> dict[str, float]:
    """Annual price-growth assumptions derived from history (complete years only)."""
    return {
        "Long-run trend (10-yr)": cagr(annual_prices, 10),
        "Recent trend (3-yr)": cagr(annual_prices, 3),
    }


def project(rate_now: float, annual_kwh: float, growth: dict[str, float], start_year: int,
            years: int = 5) -> pd.DataFrame:
    """Yearly rate ($/kWh) and cost per scenario, holding the home's usage constant."""
    rows = []
    for name, g in growth.items():
        for i in range(years + 1):
            rate = rate_now * (1 + g) ** i
            rows.append({"year": start_year + i, "scenario": name, "rate_per_kwh": rate,
                         "annual_cost": rate * annual_kwh, "monthly_cost": rate * annual_kwh / 12})
    return pd.DataFrame(rows)
