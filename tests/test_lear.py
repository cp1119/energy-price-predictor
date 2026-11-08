import numpy as np
import pytest

from energy_price_predictor.features import build_feature_table
from energy_price_predictor.models.lear import LEARForecaster, RobustAsinh, lear_design
from tests.test_features import synthetic_panel


@pytest.fixture(scope="module")
def linear_table():
    """Feature table whose target is a known linear function of the hourly inputs plus noise."""
    panel = synthetic_panel("2025-01-01", "2025-05-15", seed=3)
    panel = panel.groupby(panel.index.floor("h")).transform("first")  # constant within each hour
    table = build_feature_table(panel)
    rng = np.random.default_rng(4)
    wind = table["wind_on"] + table["wind_off"]
    table["target"] = 0.002 * table["load"] - 0.003 * wind + 0.4 * table["price_lag1"] + rng.normal(0, 2, len(table))
    return table.dropna(subset=["target", "price_lag7"])


def test_design_shapes(linear_table):
    inputs, targets = lear_design(linear_table)
    assert inputs.shape[1] == 8 * 24 + 7
    assert targets.shape[1] == 96
    assert inputs.notna().all().all()


def test_robust_asinh_round_trips():
    values = np.array([[-500.0], [0.0], [80.0], [3000.0]])
    scaler = RobustAsinh().fit(values)
    np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values)


def test_lear_learns_linear_structure(linear_table):
    day_level = linear_table.index.get_level_values("day")
    test_day = day_level.unique()[-1]
    model = LEARForecaster(windows=(30, 60), n_jobs=1)
    model.fit(linear_table[day_level < test_day])
    rows = linear_table.xs(test_day, level="day", drop_level=False)
    mae = np.abs(model.predict(rows) - rows["target"].to_numpy()).mean()
    naive_mae = np.abs(rows["price_lag1"] - rows["target"]).mean()
    assert mae < 0.25 * naive_mae
