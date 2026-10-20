"""One-time download + training for the home estimator: python -m gridcast setup-home"""
from __future__ import annotations

import io
import json

import pandas as pd
import requests

from ..config import eia_api_key
from . import HOME_DATA_DIR, datacenters, rates, recs
from .usage import UsageModel

FILES = {
    "recs": HOME_DATA_DIR / "recs2020.csv",
    "utilities": HOME_DATA_DIR / "va_utilities.csv",
    "prices": HOME_DATA_DIR / "va_prices.csv",
    "us_prices": HOME_DATA_DIR / "us_prices.csv",
    "datacenters": HOME_DATA_DIR / "va_datacenters.csv",
    "model": HOME_DATA_DIR / "usage_model.pkl",
    "eval": HOME_DATA_DIR / "usage_eval.json",
}


def _download_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=300, headers={"User-Agent": "gridcast-portfolio-project"})
    resp.raise_for_status()
    return pd.read_csv(io.BytesIO(resp.content), low_memory=False)


def step_recs() -> None:
    df = _download_csv(recs.RECS_URL)
    for w in recs.validate(df):
        print(f"  WARNING: {w}")
    df[recs.REQUIRED].to_csv(FILES["recs"], index=False)
    print(f"  {len(df):,} surveyed homes saved")


def step_model() -> None:
    raw = pd.read_csv(FILES["recs"])
    model = UsageModel().fit(raw)
    model.save(FILES["model"])
    FILES["eval"].write_text(json.dumps(model.evaluation, indent=2))
    e = model.evaluation
    print(f"  Held-out test on {e['homes_tested']:,} homes:")
    print(f"    typical error (median abs % error): {e['median_abs_pct_error']:.1f}% "
          f"vs {e['baseline_median_abs_pct_error']:.1f}% for a kWh-per-sq-ft baseline")
    print(f"    actual usage fell inside the P10-P90 range for {e['p10_p90_coverage_pct']:.0f}% of homes "
          f"(target: 80%)")


def step_utilities() -> None:
    frames = [_download_csv(u) for u in rates.NREL_URLS.values()]
    va = rates.load_va_utilities(frames)
    va.to_csv(FILES["utilities"], index=False)
    print(f"  {va['zip'].nunique():,} Virginia ZIP codes, {va['utility_name'].nunique()} utilities")


def step_prices() -> None:
    hist = rates.fetch_va_price_history(eia_api_key())
    hist.to_csv(FILES["prices"], index=False)
    print(f"  Current VA residential average: {rates.current_va_price(hist):.2f} cents/kWh")


def step_us_prices() -> None:
    hist = rates.fetch_va_price_history(eia_api_key(), state="US")
    hist.to_csv(FILES["us_prices"], index=False)
    print(f"  Current U.S. residential average: {rates.current_va_price(hist):.2f} cents/kWh")


def step_datacenters() -> None:
    dcs = datacenters.fetch_va_datacenters()
    dcs.to_csv(FILES["datacenters"], index=False)
    print(f"  {len(dcs):,} mapped data center sites")


# (label, function, file it produces). Steps whose file exists are skipped unless forced.
STEPS = [
    ("Survey of U.S. homes (EIA RECS 2020)", step_recs, "recs"),
    ("Training the usage model", step_model, "model"),
    ("Virginia utility rates by ZIP (NREL)", step_utilities, "utilities"),
    ("Virginia price history (EIA)", step_prices, "prices"),
    ("U.S. average price history, for comparison (EIA)", step_us_prices, "us_prices"),
    ("Data center locations (OpenStreetMap)", step_datacenters, "datacenters"),
]


def run(force: bool = False) -> None:
    HOME_DATA_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for label, fn, output in STEPS:
        print(f"\n{label}...")
        if FILES[output].exists() and not force:
            print("  already done (use --force to redo)")
            continue
        try:
            fn()
        except Exception as exc:  # keep going so one flaky source doesn't block the rest
            failed.append(label)
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    print("\nDone." if not failed else f"\nFinished with failures: {failed}. Re-run to retry.")
