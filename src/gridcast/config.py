"""Project-wide settings. Override paths with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("GRIDCAST_DATA_DIR", ROOT / "data"))
REPORTS_DIR = Path(os.getenv("GRIDCAST_REPORTS_DIR", ROOT / "reports"))

LOCAL_TZ = "America/New_York"


@dataclass(frozen=True)
class Area:
    """A grid area to forecast.

    kind="subba": a utility zone inside a balancing authority (EIA region-sub-ba-data).
                  EIA publishes demand only, so we benchmark against naive baselines.
    kind="ba":    a whole balancing authority (EIA region-data). EIA also publishes
                  the operator's day-ahead forecast, so we benchmark against it.
    """

    code: str
    kind: str
    parent: str | None
    name: str
    weather_points: tuple[tuple[float, float], ...]  # (lat, lon) points, averaged


AREAS = {
    # Dominion zone: Northern Virginia "Data Center Alley" (Loudoun County) + rest of VA.
    "DOM": Area("DOM", "subba", "PJM", "Dominion Energy zone (Virginia)",
                ((39.04, -77.49), (37.54, -77.44), (36.85, -76.29))),  # Ashburn, Richmond, Norfolk
    # Whole PJM grid (13 states + DC).
    "PJM": Area("PJM", "ba", None, "PJM Interconnection",
                ((39.95, -75.17), (39.96, -83.00), (41.88, -87.63),  # Philadelphia, Columbus, Chicago
                 (39.04, -77.49), (40.44, -79.99))),                  # Ashburn, Pittsburgh
}


def eia_api_key() -> str:
    key = os.getenv("EIA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "EIA_API_KEY is not set. Get a free key at https://www.eia.gov/opendata/ "
            "then add the line EIA_API_KEY=your_key to a .env file in the project folder "
            "(PowerShell alternative: $env:EIA_API_KEY=\"your_key\")."
        )
    return key
