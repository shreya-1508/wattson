"""Walk-forward backtest: retrain each month on all prior data, forecast that month."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LOCAL_TZ
from .features import feature_target
from .models import GBMForecaster, day_naive, seasonal_naive


def mape(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def mae(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.mean(np.abs(y - yhat)))


def peak_mape(y: pd.Series, yhat: pd.Series) -> float:
    """Error on each local day's peak demand: what grid planners care most about."""
    day = y.index.tz_convert(LOCAL_TZ).date
    return mape(y.groupby(day).max(), yhat.groupby(day).max())


def backtest(df: pd.DataFrame, test_months: int = 12, min_train_days: int = 365) -> pd.DataFrame:
    """Return hourly actuals + every model's forecast over the test period."""
    X, y = feature_target(df)
    local_month = X.index.tz_convert(LOCAL_TZ).tz_localize(None).to_period("M")
    months = sorted(set(local_month))[-test_months:]
    first_ok = X.index[0] + pd.Timedelta(days=min_train_days)

    parts = []
    for m in months:
        test = local_month == m
        test_start = X.index[test][0]
        if test_start < first_ok:
            continue
        # Last training label must be known 24h before the first forecast hour.
        train = X.index <= test_start - pd.Timedelta(hours=24)
        model = GBMForecaster().fit(X[train], y[train])
        Xt = X[test]
        part = pd.DataFrame({"actual": y[test]})
        part["gbm"] = model.predict(Xt)
        part["seasonal_naive"] = seasonal_naive(Xt)
        part["day_naive"] = day_naive(Xt)
        if "operator_forecast_mw" in df and df["operator_forecast_mw"].notna().any():
            part["operator"] = df.loc[Xt.index, "operator_forecast_mw"]
        parts.append(part)
    if not parts:
        raise ValueError("Not enough history for a backtest. Fetch at least 2 years.")
    return pd.concat(parts)


def score(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in preds.columns.drop("actual"):
        p = preds[["actual", col]].dropna()
        rows.append({
            "model": col,
            "mape_pct": mape(p["actual"], p[col]),
            "peak_mape_pct": peak_mape(p["actual"], p[col]),
            "mae_mw": mae(p["actual"], p[col]),
            "hours": len(p),
        })
    return pd.DataFrame(rows).sort_values("mape_pct").reset_index(drop=True)
