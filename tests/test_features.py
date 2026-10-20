import numpy as np
import pandas as pd

from gridcast.data import clean
from gridcast.features import HORIZON, feature_target, make_features
from gridcast.synthetic import make_dataset


def _df():
    return make_dataset(start="2024-01-01", end="2024-03-31")


def test_no_lookahead_leakage():
    """Changing demand in the last 23 hours must not change any feature for the final hour."""
    df = _df()
    base = make_features(df).iloc[-1]
    tampered = df.copy()
    tampered.iloc[-(HORIZON - 1):, tampered.columns.get_loc("demand_mw")] *= 10
    after = make_features(tampered).iloc[-1]
    pd.testing.assert_series_equal(base, after)


def test_feature_target_has_no_nans():
    X, y = feature_target(_df())
    assert len(X) == len(y) > 1000
    assert not X.isna().any().any()


def test_clean_removes_spikes_and_fills_short_gaps():
    df = _df()
    df.iloc[500, 0] = 0          # reporting error
    df.iloc[600, 0] = 1e7        # spike
    df = df.drop(df.index[700])  # one missing hour
    out = clean(df)
    assert out.index.freq is not None or (out.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    assert out["demand_mw"].iloc[500] > 1000
    assert out["demand_mw"].iloc[600] < 1e6
    assert not np.isnan(out["demand_mw"].iloc[700])
