import numpy as np
import pandas as pd
import pytest

from energy_price_predictor.backtest import (
    add_ensemble,
    inverse_mae_weights,
    dm_test,
    error_metrics,
    metrics_table,
    run_backtest,
    walk_forward,
)
from energy_price_predictor.data import TARGET_KEY
from energy_price_predictor.features import build_feature_table
from energy_price_predictor.models import NaiveForecaster
from energy_price_predictor.models.base import Forecaster
from tests.test_features import synthetic_panel


class SpyForecaster(Forecaster):
    """Records what it was trained on; predicts the training-set mean."""

    name = "spy"
    refit_every = 7
    min_train_days = 14

    def __init__(self):
        self.fits: list[tuple[pd.Timestamp, int]] = []
        self.mean = np.nan

    def fit(self, train):
        last_day = train.index.get_level_values("day").max()
        self.fits.append((last_day, len(train)))
        self.mean = train["target"].mean()

    def predict(self, rows):
        return np.full(len(rows), self.mean)


@pytest.fixture(scope="module")
def autumn():
    panel = synthetic_panel("2025-09-01", "2025-11-15", seed=2)
    return panel, build_feature_table(panel)


def test_walk_forward_trains_only_on_earlier_days(autumn):
    _, table = autumn
    days = pd.date_range("2025-10-01", "2025-10-31")
    spy = SpyForecaster()
    forecast = walk_forward(spy, table, days)
    assert len(spy.fits) == 5  # Oct 1, 8, 15, 22, 29
    for (last_train_day, _), fit_day in zip(spy.fits, days[::7]):
        assert last_train_day == fit_day - pd.Timedelta(days=1)
    assert forecast.notna().all()


def test_walk_forward_covers_dst_day(autumn):
    _, table = autumn
    forecast = walk_forward(NaiveForecaster(), table, pd.DatetimeIndex([pd.Timestamp("2025-10-26")]))
    assert len(forecast) == 100


def test_naive_rule():
    rows = pd.DataFrame({"dow": [0, 1, 5, 6, 3], "price_lag1": [1.0] * 5, "price_lag7": [7.0] * 5})
    assert NaiveForecaster().predict(rows).tolist() == [7.0, 1.0, 7.0, 7.0, 1.0]


def test_error_metrics_hand_computed():
    index = pd.date_range("2025-10-01", periods=4, freq="15min", tz="UTC")
    actual = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
    forecast = pd.Series([12.0, 18.0, 30.0, 44.0], index=index)
    metrics = error_metrics(forecast, actual)
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(6.0))
    assert metrics["mae_hourly"] == pytest.approx(1.0)  # hourly means 26 vs 25


def test_metrics_table_and_dm(autumn):
    panel, table = autumn
    days = pd.date_range("2025-10-01", "2025-11-10")
    results = run_backtest([NaiveForecaster()], table, panel[TARGET_KEY], days)
    results["perfect_ish"] = results["actual"] + np.random.default_rng(0).normal(0, 1, len(results))
    metrics = metrics_table(results)
    assert metrics.loc["naive", "rmae"] == pytest.approx(1.0)
    assert metrics.loc["perfect_ish", "rmae"] < 0.1
    _, p_value = dm_test(results, "naive", "perfect_ish")
    assert p_value < 0.01
    _, reverse_p = dm_test(results, "perfect_ish", "naive")
    assert reverse_p > 0.99


def test_add_ensemble_weights():
    index = pd.date_range("2025-10-01", periods=2, freq="15min", tz="UTC")
    results = pd.DataFrame({"actual": [0.0, 0.0], "a": [10.0, 10.0], "b": [20.0, 20.0]}, index=index)
    assert add_ensemble(results, ["a", "b"])["ensemble"].tolist() == [15.0, 15.0]
    weighted = add_ensemble(results, ["a", "b"], weights={"a": 3.0, "b": 1.0})
    assert weighted["ensemble"].tolist() == [12.5, 12.5]
    assert inverse_mae_weights(results, ["a", "b"]) == {"a": 0.1, "b": 0.05}
