"""How data centers may be affecting a Virginia household's bill.

Two kinds of evidence, kept separate on purpose:
1. Measured: how Virginia's residential price has moved relative to the U.S. average.
   That gap captures *all* Virginia-specific factors, data centers among them.
2. Published estimates: figures from regulators, the grid's market monitor and the
   utility, scaled to this household's usage. Each carries its own caveat.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REFERENCE_KWH = 1000  # Virginia's standard "typical residential bill" uses 1,000 kWh/month


@dataclass(frozen=True)
class Study:
    source: str
    date: str
    url: str
    finding: str
    caveat: str
    dollars_per_month_typical: float | None = None  # at REFERENCE_KWH
    dollars_per_mwh: float | None = None


STUDIES = [
    Study(
        source="PJM Independent Market Monitor",
        date="Aug 2026",
        url="https://www.utilitydive.com/news/data-center-pjm-wholesale-market/828917/",
        finding="Existing and forecast data center load added about $10.48 per MWh (9%) to PJM wholesale "
                "power costs through mid-2026, through the capacity market alone, and raised capacity costs "
                "by $29.4 billion across the last four auctions.",
        caveat="Upper bound for a Dominion customer: Dominion supplies most of its own capacity, so less of "
               "this market cost reaches its bills than for utilities that buy all capacity from PJM.",
        dollars_per_mwh=10.48,
    ),
    Study(
        source="Dominion Energy, via Virginia Mercury",
        date="Jul 2025",
        url="https://virginiamercury.com/2025/07/24/after-capacity-auction-predicts-higher-energy-bills-youngkin-8-governors-air-frustrations-in-letter-to-pjm/",
        finding="Dominion said PJM capacity auction costs amount to about $2 on a monthly bill.",
        caveat="Covers all capacity cost changes, not only those driven by data centers.",
        dollars_per_month_typical=2.0,
    ),
    Study(
        source="JLARC (Virginia's legislative watchdog), as summarized by Clean Virginia",
        date="Dec 2024",
        url="https://www.cleanvirginia.org/2025/03/25/blog-the-rising-cost-of-power/",
        finding="With unconstrained data center growth, the average Dominion residential customer could pay "
                "about $23 more per month within five years for new infrastructure, and up to $37 more per "
                "month by 2040 (not inflation-adjusted).",
        caveat="A projection made before Virginia's 2025-2026 rate changes that assign more costs to data "
               "centers. JLARC also found that, as of 2024, other customers were not subsidizing data "
               "centers' energy costs.",
        dollars_per_month_typical=23.0,
    ),
]

POLICY = [
    ("Jan 2026", "Virginia regulators approved a Dominion base rate increase of about $16/month on a typical "
                 "bill. It reflects many costs, not only data centers.",
     "https://insideclimatenews.org/news/07012026/virginia-regulators-approve-new-dominion-rates/"),
    ("Jan 2027", "A new rate class (GS-5) for very large customers like data centers starts: 14-year contracts "
                 "and minimum charges covering most of the grid and generation costs built to serve them.",
     "https://insideclimatenews.org/news/07012026/virginia-regulators-approve-new-dominion-rates/"),
    ("2026", "Regulators ordered Dominion to design a tariff that assigns transmission costs directly to data "
             "centers and other large loads that need them.",
     "https://wtop.com/virginia/2026/08/scc-orders-dominion-to-develop-tariff-to-assign-more-transmission-costs-to-data-centers/"),
]


def personalize(monthly_kwh: float) -> pd.DataFrame:
    """Each study's figure scaled to this home's average monthly use."""
    rows = []
    for s in STUDIES:
        if s.dollars_per_mwh is not None:
            yours = s.dollars_per_mwh * monthly_kwh / 1000
        else:
            yours = s.dollars_per_month_typical * monthly_kwh / REFERENCE_KWH
        rows.append({"Source": f"{s.source} ({s.date})", "What they found": s.finding,
                     "For your home": yours, "Keep in mind": s.caveat, "url": s.url})
    return pd.DataFrame(rows)


@dataclass
class StateGap:
    base_year: int
    end_year: int
    va_change_pct: float
    us_change_pct: float
    bill_now: float
    bill_if_national: float

    @property
    def gap_dollars(self) -> float:
        return self.bill_now - self.bill_if_national


def state_vs_national(va: pd.Series, us: pd.Series, base_year: int, rate_now: float,
                      monthly_kwh: float) -> StateGap:
    """What this bill would be if Virginia prices had tracked the U.S. average since `base_year`."""
    common = va.index.intersection(us.index)
    end = int(common.max())
    va_ratio = va[end] / va[base_year]
    us_ratio = us[end] / us[base_year]
    bill_now = rate_now * monthly_kwh
    return StateGap(base_year, end, (va_ratio - 1) * 100, (us_ratio - 1) * 100,
                    bill_now, bill_now * us_ratio / va_ratio)


def indexed(series: dict[str, pd.Series], base_year: int) -> pd.DataFrame:
    """Put several yearly series on one scale: base year = 100."""
    out = {}
    for name, s in series.items():
        s = s.dropna()
        if base_year in s.index:
            out[name] = s[s.index >= base_year] / s[base_year] * 100
    return pd.DataFrame(out)
