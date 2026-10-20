"""Command line: python -m gridcast {fetch,backtest,baseload,demo} --area DOM"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import pandas as pd

from . import baseload, data, evaluate, synthetic
from .config import REPORTS_DIR, ROOT


def _read_text_any_encoding(path) -> str:
    raw = path.read_bytes()
    # Windows editors may save UTF-8 with a BOM, or UTF-16.
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _load_dotenv() -> None:
    import os

    # Notepad often saves ".env" as ".env.txt" with the extension hidden.
    for env in (ROOT / ".env", ROOT / ".env.txt"):
        if not env.exists():
            continue
        for line in _read_text_any_encoding(env).splitlines():
            line = line.strip().lstrip("﻿")
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return


def _report(df: pd.DataFrame, tag: str, test_months: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds = evaluate.backtest(df, test_months=test_months)
    scores = evaluate.score(preds)
    preds.to_csv(REPORTS_DIR / f"{tag}_predictions.csv", index_label="time_utc")
    scores.to_csv(REPORTS_DIR / f"{tag}_scores.csv", index=False)
    print(f"\nBacktest ({tag}), last {test_months} months, retrained monthly:")
    print(scores.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    trend = baseload.yearly_trend(df)
    cagr = baseload.annual_growth_rate(df)
    trend.to_csv(REPORTS_DIR / f"{tag}_baseload.csv", index_label="year")
    print(f"\nOvernight baseload (mild-weather nights) by year ({tag}):")
    print(trend.to_string(float_format=lambda v: f"{v:,.1f}"))
    print(f"Estimated baseload growth: {cagr:.1f}% per year")

    best = scores.iloc[0]
    summary = {"area": tag, "best_model": best["model"], "scores": scores.to_dict("records"),
               "baseload_growth_pct_per_year": cagr}
    (REPORTS_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))


def main(argv=None) -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="gridcast")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Download and cache hourly data from EIA + Open-Meteo")
    f.add_argument("--area", default="DOM", choices=["DOM", "PJM"])
    f.add_argument("--start", default="2021-01-01")
    f.add_argument("--end", default=str(date.today() - timedelta(days=2)))

    b = sub.add_parser("backtest", help="Walk-forward backtest + baseload analysis on cached data")
    b.add_argument("--area", default="DOM", choices=["DOM", "PJM"])
    b.add_argument("--test-months", type=int, default=12)

    d = sub.add_parser("demo", help="Run everything on SYNTHETIC data (no API key needed)")
    d.add_argument("--test-months", type=int, default=6)

    h = sub.add_parser("setup-home", help="Download data + train the model for the home cost app")
    h.add_argument("--force", action="store_true", help="Redo steps that already finished")

    args = p.parse_args(argv)
    if args.cmd == "setup-home":
        from .home import setup

        setup.run(force=args.force)
        return
    if args.cmd == "fetch":
        df = data.build(args.area, args.start, args.end)
        data.save(df, args.area)
        print(f"Saved {len(df):,} hours to {data.cache_path(args.area)}")
        print(f"Missing demand hours: {df['demand_mw'].isna().sum():,}")
    elif args.cmd == "backtest":
        _report(data.load(args.area), args.area.lower(), args.test_months)
    elif args.cmd == "demo":
        print("SYNTHETIC DATA: numbers below demonstrate the pipeline only.")
        _report(synthetic.make_dataset(), "synthetic", args.test_months)


if __name__ == "__main__":
    main()
