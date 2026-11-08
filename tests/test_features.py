import numpy as np
import pandas as pd
import pytest

from energy_price_predictor.data import REGISTRY, TARGET_KEY
from energy_price_predictor.features import (
    build_feature_table,
    daily_matrix,
    decision_time,
    delivery_periods,
    feature_columns,
    mask_unavailable,
    to_periods,
)

SPRING_FORWARD = pd.Timestamp("2025-03-30")
FALL_BACK = pd.Timestamp("2025-10-26")


def synthetic_panel(start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Random values for every registered series on the 15-minute UTC grid."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="15min", inclusive="left")
    data = {spec.key: rng.normal(100, 30, len(index)) for spec in REGISTRY}
    for key in ("load_fc", "load_fc_smard"):
        data[key] = rng.normal(55_000, 5_000, len(index))
    for key in ("solar_fc", "wind_on_fc", "wind_off_fc"):
        data[key] = rng.uniform(0, 20_000, len(index))
    return pd.DataFrame(data, index=index)


@pytest.fixture(scope="module")
def spring_panel():
    return synthetic_panel("2025-02-15", "2025-04-15")


@pytest.fixture(scope="module")
def autumn_panel():
    return synthetic_panel("2025-09-01", "2025-11-15", seed=1)


def test_delivery_periods_handle_dst():
    assert len(delivery_periods(pd.Timestamp("2025-06-01"))) == 96
    assert len(delivery_periods(SPRING_FORWARD)) == 92
    assert len(delivery_periods(FALL_BACK)) == 100
    assert delivery_periods(pd.Timestamp("2025-06-01"))[0] == pd.Timestamp("2025-05-31 22:00", tz="UTC")


def test_decision_time_is_ten_local_on_day_d():
    assert decision_time(pd.Timestamp("2025-06-02")) == pd.Timestamp("2025-06-01 08:00", tz="UTC")
    assert decision_time(pd.Timestamp("2025-01-02")) == pd.Timestamp("2025-01-01 09:00", tz="UTC")


def test_to_periods_maps_slots_on_dst_days():
    slots = np.arange(96, dtype=float)
    spring = to_periods(SPRING_FORWARD, slots)
    assert len(spring) == 92 and 8.0 not in spring.to_numpy()
    autumn = to_periods(FALL_BACK, slots)
    assert len(autumn) == 100 and (autumn.to_numpy() == 8.0).sum() == 2


def test_daily_matrix_round_trips(autumn_panel):
    matrix = daily_matrix(autumn_panel[TARGET_KEY])
    day = pd.Timestamp("2025-10-10")
    rebuilt = to_periods(day, matrix.loc[day])
    original = autumn_panel[TARGET_KEY].loc[rebuilt.index]
    np.testing.assert_allclose(rebuilt.to_numpy(), original.to_numpy())


def test_daily_matrix_averages_fall_back_hour(autumn_panel):
    series = autumn_panel[TARGET_KEY]
    matrix = daily_matrix(series)
    periods = delivery_periods(FALL_BACK)
    local = periods.tz_convert("Europe/Berlin")
    at_two = series.loc[periods[(local.hour == 2) & (local.minute == 0)]]
    assert len(at_two) == 2
    assert matrix.loc[FALL_BACK, 8] == pytest.approx(at_two.mean())


def test_spring_forward_rows_are_dropped_but_lags_filled(spring_panel):
    table = build_feature_table(spring_panel)
    assert len(table.loc[SPRING_FORWARD]) == 92
    next_day = table.loc[SPRING_FORWARD + pd.Timedelta(days=1)]
    assert next_day.loc[8, "price_lag1"] == pytest.approx(next_day.loc[7, "price_lag1"])


@pytest.mark.parametrize("fill", [np.nan, 1e9])
@pytest.mark.parametrize(
    "panel_name, target_day",
    [
        ("spring_panel", pd.Timestamp("2025-03-20")),
        ("spring_panel", SPRING_FORWARD),
        ("spring_panel", SPRING_FORWARD + pd.Timedelta(days=1)),
        ("autumn_panel", pd.Timestamp("2025-10-01")),
        ("autumn_panel", FALL_BACK),
        ("autumn_panel", FALL_BACK + pd.Timedelta(days=1)),
    ],
)
def test_features_never_use_unavailable_data(request, panel_name, target_day, fill):
    """Overwriting everything published after the decision time must not change any feature."""
    panel = request.getfixturevalue(panel_name)
    full = build_feature_table(panel).loc[target_day]
    masked = build_feature_table(mask_unavailable(panel, target_day, fill=fill)).loc[target_day]
    columns = feature_columns(full)
    pd.testing.assert_frame_equal(full[columns], masked[columns])


def test_mask_blocks_neighbour_prices_for_target_day(autumn_panel):
    target_day = pd.Timestamp("2025-10-10")
    masked = mask_unavailable(autumn_panel, target_day)
    periods = delivery_periods(target_day)
    assert masked.loc[periods, ["price_de", "price_fr", "price_nl"]].isna().all().all()
    assert masked.loc[periods, ["load_fc", "wind_on_fc", "solar_fc"]].notna().all().all()
