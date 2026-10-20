"""Forecasting models: two naive baselines and a gradient-boosted model.

Growth handling: tree models cannot predict values above anything seen in
training, but data-center load keeps setting new records. So the GBM learns
demand *relative to the trailing 7-day mean* (a scale-free target) and we
multiply back. Level comes from recent history; shape comes from the model.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

LEVEL_COL = "mean_7d_before"
SCALED_COLS = ["lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
               "mean_24h_before", "min_24h_before"]


def seasonal_naive(X: pd.DataFrame) -> pd.Series:
    """Same hour last week."""
    return X["lag_168h"].rename("seasonal_naive")


def day_naive(X: pd.DataFrame) -> pd.Series:
    """Same hour yesterday."""
    return X["lag_24h"].rename("day_naive")


def _scale(X: pd.DataFrame) -> pd.DataFrame:
    Xs = X.copy()
    for c in SCALED_COLS:
        Xs[c] = X[c] / X[LEVEL_COL]
    return Xs.drop(columns=[LEVEL_COL])


class GBMForecaster:
    name = "gbm"

    def __init__(self, **params):
        defaults = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=48,
                        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        defaults.update(params)
        self.model = HistGradientBoostingRegressor(**defaults)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBMForecaster":
        Xs = _scale(X)
        self.feature_names_ = list(Xs.columns)
        self.model.fit(Xs, y / X[LEVEL_COL])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ratio = self.model.predict(_scale(X)[self.feature_names_])
        return pd.Series(ratio * X[LEVEL_COL].to_numpy(), index=X.index, name=self.name)
