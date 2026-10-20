from gridcast import baseload, evaluate
from gridcast.synthetic import make_dataset


def test_backtest_beats_naive_on_synthetic():
    df = make_dataset(start="2023-01-01", end="2025-03-31")
    preds = evaluate.backtest(df, test_months=2)
    scores = evaluate.score(preds).set_index("model")
    assert scores.loc["gbm", "mape_pct"] < scores.loc["seasonal_naive", "mape_pct"]
    assert {"gbm", "seasonal_naive", "day_naive", "operator"} <= set(scores.index)


def test_baseload_detects_growth():
    df = make_dataset(start="2022-01-01", end="2025-12-31", dc_growth_per_year=0.25)
    assert baseload.annual_growth_rate(df) > 2.0
    flat = make_dataset(start="2022-01-01", end="2025-12-31", dc_growth_per_year=0.0)
    assert abs(baseload.annual_growth_rate(flat)) < 1.5


def test_metrics():
    import pandas as pd

    y = pd.Series([100.0, 200.0])
    assert evaluate.mape(y, y) == 0
    assert abs(evaluate.mape(y, y * 1.1) - 10.0) < 1e-9
    assert abs(evaluate.mae(y, y + 5) - 5.0) < 1e-9
