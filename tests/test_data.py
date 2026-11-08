import pandas as pd

from energy_price_predictor.data import (
    REGISTRY,
    SPECS_BY_KEY,
    TARGET_KEY,
    SeriesSpec,
    load_panel,
    read_cache,
    to_quarter_hours,
    update_cache,
)


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


class FakeClient:
    """Serves values() from an in-memory series and records the requested ranges."""

    def __init__(self, series: pd.Series):
        self.series = series
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def values(self, entity, name, *, provider, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.calls.append((start, end))
        window = self.series[(self.series.index >= start) & (self.series.index <= end)]
        frame = window.to_frame("value")
        frame["knowledge_time"] = utc("2026-11-08")
        frame["annotation"] = None
        return frame


def test_hourly_series_is_repeated_over_quarter_hours():
    hourly = pd.Series([10.0, 20.0], index=pd.DatetimeIndex([utc("2025-01-01 00:00"), utc("2025-01-01 01:00")]))
    quarter_hours = to_quarter_hours(hourly, utc("2025-01-01 00:00"), utc("2025-01-01 02:00"))
    assert len(quarter_hours) == 8
    assert quarter_hours.iloc[:4].tolist() == [10.0] * 4
    assert quarter_hours.iloc[4:].tolist() == [20.0] * 4


def test_fill_stops_at_real_gaps():
    sparse = pd.Series([10.0], index=pd.DatetimeIndex([utc("2025-01-01 00:00")]))
    quarter_hours = to_quarter_hours(sparse, utc("2025-01-01 00:00"), utc("2025-01-01 02:00"))
    assert quarter_hours.notna().sum() == 4


def test_quarter_hour_series_is_not_filled_past_its_end():
    index = pd.date_range(utc("2025-10-01 00:00"), periods=4, freq="15min")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    quarter_hours = to_quarter_hours(series, index[0], utc("2025-10-01 02:00"))
    assert quarter_hours.notna().sum() == 4


def test_quarter_hour_series_passes_through():
    index = pd.date_range(utc("2025-10-01"), periods=4, freq="15min")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    quarter_hours = to_quarter_hours(series, index[0], index[-1] + pd.Timedelta("15min"))
    assert quarter_hours.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_update_cache_fetches_only_the_missing_tail(tmp_path):
    index = pd.date_range(utc("2025-01-01"), utc("2025-01-20"), freq="1h")
    truth = pd.Series(range(len(index)), index=index, dtype=float)
    spec = SeriesSpec("demo", "DE-LU", "demo.name", "entsoe", "through_d")

    client = FakeClient(truth)
    update_cache(spec, utc("2025-01-10"), client=client, start=utc("2025-01-01"), cache_dir=tmp_path)
    update_cache(spec, utc("2025-01-20"), client=client, start=utc("2025-01-01"), cache_dir=tmp_path)

    second_start = client.calls[1][0]
    assert second_start == utc("2025-01-07")  # last cached point minus the 3-day overlap
    cached = read_cache("demo", tmp_path)
    pd.testing.assert_series_equal(cached, truth.rename("demo"), check_freq=False, check_names=False)


def test_load_panel_applies_scale(tmp_path, monkeypatch):
    index = pd.date_range(utc("2025-01-01"), periods=4, freq="15min", name="valid_time")
    pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=index).to_parquet(tmp_path / "load_fc_smard.parquet")
    panel = load_panel(index[0], index[-1] + pd.Timedelta("15min"), keys=["load_fc_smard"], cache_dir=tmp_path)
    assert panel["load_fc_smard"].tolist() == [4.0, 8.0, 12.0, 16.0]


def test_registry_rules():
    keys = [spec.key for spec in REGISTRY]
    assert len(keys) == len(set(keys))
    # No price, ours or a neighbour's, may be read for D+1: all SDAC zones clear together.
    for spec in REGISTRY:
        if spec.name.startswith("price."):
            assert spec.availability == "through_d", spec.key
    assert SPECS_BY_KEY[TARGET_KEY].entity == "DE-LU"
