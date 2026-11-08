import numpy as np
import pytest

from energy_price_predictor.features import build_feature_table
from energy_price_predictor.models.gbm import GBMForecaster
from tests.test_features import synthetic_panel


@pytest.fixture(scope="module")
def table():
    panel = synthetic_panel("2025-01-01", "2025-04-30", seed=5)
    table = build_feature_table(panel)
    rng = np.random.default_rng(6)
    table["target"] = table["price_d_mean"] + 0.002 * table["residual_load"] + rng.normal(0, 1, len(table))
    return table.dropna(subset=["target", "price_lag7"])


@pytest.mark.parametrize("relative", [True, False])
def test_gbm_beats_naive_on_learnable_target(table, relative):
    day_level = table.index.get_level_values("day")
    test_day = day_level.unique()[-1]
    model = GBMForecaster({"n_estimators": 200, "min_child_samples": 20}, relative_to_level=relative, train_start="2025-01-01")
    model.fit(table[day_level < test_day])
    rows = table.xs(test_day, level="day", drop_level=False)
    mae = np.abs(model.predict(rows) - rows["target"]).mean()
    naive_mae = np.abs(rows["price_lag1"] - rows["target"]).mean()
    assert mae < 0.5 * naive_mae


def test_recent_days_weigh_more(table):
    weights = GBMForecaster(half_life_days=30).sample_weights(table)
    days = table.index.get_level_values("day")
    assert weights[days == days.max()].max() == 1.0
    assert weights[days == days.max() - np.timedelta64(30, "D")].max() == pytest.approx(0.5)
