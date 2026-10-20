"""Home electricity usage model trained on RECS.

Predicts annual kWh as a range (P10 / P50 / P90) using quantile gradient
boosting on log(kWh), plus the share of use going to heating and cooling so
the annual estimate can be spread across months by local weather.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .recs import CATEGORICAL, to_features

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


class UsageModel:
    def __init__(self):
        self.models: dict[str, HistGradientBoostingRegressor] = {}
        self.columns: list[str] = []
        self.evaluation: dict = {}
        self.widen = 0.0  # log-space range widening from conformal calibration

    def _gbm(self, **kw) -> HistGradientBoostingRegressor:
        cat = [c in CATEGORICAL for c in self.columns]
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=30, categorical_features=cat,
                                             random_state=0, **kw)

    def _fit_all(self, X: pd.DataFrame, raw: pd.DataFrame, w: np.ndarray) -> None:
        y = np.log(raw["KWH"].clip(lower=100))
        for name, q in QUANTILES.items():
            self.models[name] = self._gbm(loss="quantile", quantile=q).fit(X, y, sample_weight=w)
        kwh = raw["KWH"].clip(lower=100)
        self.models["heat_share"] = self._gbm().fit(X, (raw["KWHSPH"] / kwh).clip(0, 0.9), sample_weight=w)
        self.models["cool_share"] = self._gbm().fit(X, (raw["KWHCOL"] / kwh).clip(0, 0.9), sample_weight=w)

    def _calibrate(self, X: pd.DataFrame, actual: np.ndarray, w: np.ndarray, target: float = 0.80) -> None:
        """Split-conformal widening: stretch P10/P90 (in log space) until `target` of
        unseen homes fall inside. Quantile models fit in-sample are usually too narrow."""
        self.widen = 0.0
        pred = self._predict_frame(X)
        y = np.log(actual)
        score = np.maximum(np.log(pred["p10"]) - y, y - np.log(pred["p90"]))
        order = np.argsort(score)
        cum = np.cumsum(w[order]) / w.sum()
        self.widen = max(0.0, float(score[order][np.searchsorted(cum, target)]))

    def fit(self, raw: pd.DataFrame, evaluate: bool = True) -> "UsageModel":
        raw = raw[(raw["KWH"] > 0) & (raw["TOTSQFT_EN"] > 0)].reset_index(drop=True)
        X = to_features(raw)
        self.columns = list(X.columns)
        w = raw["NWEIGHT"].to_numpy(dtype=float)
        kwh = raw["KWH"].to_numpy(dtype=float)
        idx = np.arange(len(raw))
        rest, cal = train_test_split(idx, test_size=0.2, random_state=7)
        if evaluate:
            # 60% train / 20% calibrate / 20% test: the test homes are never seen.
            tr, te = train_test_split(rest, test_size=0.25, random_state=42)
            self._fit_all(X.iloc[tr], raw.iloc[tr], w[tr])
            self._calibrate(X.iloc[cal], kwh[cal], w[cal])
            self.evaluation = self._evaluate(X.iloc[te], raw.iloc[te], raw.iloc[tr], w[te], w[tr])
        # Final model: train on 80%, calibrate ranges on the held-back 20%.
        self._fit_all(X.iloc[rest], raw.iloc[rest], w[rest])
        self._calibrate(X.iloc[cal], kwh[cal], w[cal])
        return self

    def _evaluate(self, Xte, te, tr, w_te, w_tr) -> dict:
        actual = te["KWH"].to_numpy()
        pred = self._predict_frame(Xte)
        ape = np.abs(pred["p50"] - actual) / actual
        covered = (actual >= pred["p10"]) & (actual <= pred["p90"])
        # Baseline: national kWh-per-square-foot rate x square footage.
        per_sqft = _weighted_median((tr["KWH"] / tr["TOTSQFT_EN"]).to_numpy(), w_tr)
        base_ape = np.abs(per_sqft * te["TOTSQFT_EN"].to_numpy() - actual) / actual
        return {
            "homes_tested": int(len(te)),
            "median_abs_pct_error": float(_weighted_median(ape, w_te) * 100),
            "baseline_median_abs_pct_error": float(_weighted_median(base_ape, w_te) * 100),
            "p10_p90_coverage_pct": float(np.average(covered, weights=w_te) * 100),
        }

    def _predict_frame(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        X = X[self.columns]
        out = {n: np.exp(self.models[n].predict(X)) for n in QUANTILES}
        # Quantile models are fit separately; keep them ordered.
        out["p10"] = np.minimum(out["p10"], out["p50"]) * np.exp(-self.widen)
        out["p90"] = np.maximum(out["p90"], out["p50"]) * np.exp(self.widen)
        heat = np.clip(self.models["heat_share"].predict(X), 0, 0.9)
        cool = np.clip(self.models["cool_share"].predict(X), 0, 0.9)
        total = heat + cool
        scale = np.where(total > 0.95, 0.95 / np.maximum(total, 1e-9), 1.0)
        out["heat_share"], out["cool_share"] = heat * scale, cool * scale
        return out

    def predict_one(self, row: pd.DataFrame) -> dict[str, float]:
        return {k: float(v[0]) for k, v in self._predict_frame(row).items()}

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "UsageModel":
        return pickle.loads(path.read_bytes())


def monthly_kwh(annual: float, heat_share: float, cool_share: float, dd: pd.DataFrame) -> pd.Series:
    """Spread annual kWh over 12 months: flat base + heating by HDD + cooling by CDD."""
    def weights(col):
        total = dd[col].sum()
        return dd[col] / total if total > 0 else pd.Series(1 / 12, index=dd.index)

    base = (1 - heat_share - cool_share) * annual / 12
    months = base + heat_share * annual * weights("hdd") + cool_share * annual * weights("cdd")
    return months.rename("kwh")
