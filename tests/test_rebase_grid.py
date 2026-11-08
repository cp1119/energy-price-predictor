import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from energy_price_predictor.rebase_grid import (
    API_KEY_ENV_VAR,
    RebaseGridClient,
    RebaseGridError,
)

PRICE = "price.day_ahead.day_ahead_auction_price"


class FakeResponse:
    def __init__(self, status: int, body: dict, headers: dict | None = None):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    """Replays queued responses and records each request."""

    def __init__(self, responses: list[FakeResponse]):
        self.headers: dict[str, str] = {}
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def metadata_body(resolution="PT1H", history=None):
    return {
        "providers": [
            {"provider": "entsoe", "resolution": resolution, "resolution_history": history or []}
        ]
    }


def points_body(times, values):
    return {
        "unit": "EUR/MWh",
        "points": [
            {"valid_time": t, "value": v, "knowledge_time": "2026-01-01T00:00:00Z", "annotation": None}
            for t, v in zip(times, values)
        ],
    }


def make_client(responses):
    session = FakeSession(responses)
    return RebaseGridClient(api_key="test", session=session), session


def test_values_parses_points_into_utc_indexed_frame():
    client, session = make_client(
        [
            FakeResponse(200, metadata_body()),
            FakeResponse(200, points_body(["2026-11-03T00:00:00Z", "2026-11-03T01:00:00Z"], [10.0, 12.5])),
        ]
    )
    frame = client.values("DK-DK1", PRICE, provider="entsoe", start="2026-11-03", end="2026-11-03T01:00")

    assert list(frame["value"]) == [10.0, 12.5]
    assert str(frame.index.tz) == "UTC"
    assert frame.attrs["unit"] == "EUR/MWh"
    params = session.calls[1][2]["params"]
    assert params == {
        "provider": "entsoe",
        "start": "2026-11-03T00:00:00Z",
        "end": "2026-11-03T01:00:00Z",
    }
    assert session.headers["X-API-Key"] == "test"


def test_values_splits_long_ranges_using_finest_resolution():
    # PT15M in the history -> 99,999 quarter hours per window (~1,041.66 days).
    client, session = make_client(
        [FakeResponse(200, metadata_body("PT1H", [{"resolution": "PT15M"}]))]
        + [FakeResponse(200, points_body([], [])) for _ in range(3)]
    )
    client.values("DK-DK1", PRICE, provider="entsoe", start="2018-01-01", end="2026-01-01")

    windows = [(c[2]["params"]["start"], c[2]["params"]["end"]) for c in session.calls[1:]]
    assert len(windows) == 3
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert pd.Timestamp(next_start) - pd.Timestamp(prev_end) == pd.Timedelta(minutes=15)
    assert windows[-1][1] == "2026-01-01T00:00:00Z"
    for start, end in windows:
        assert (pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(minutes=15) + 1 <= 100_000


def test_multi_column_values_are_flattened():
    body = {
        "unit": "MW",
        "points": [
            {
                "valid_time": "2026-11-01T00:00:00Z",
                "values": {"p50": 1.0, "p90": 2.0},
                "knowledge_time": "2026-10-31T00:00:00Z",
                "annotation": None,
            }
        ],
    }
    client, _ = make_client([FakeResponse(200, metadata_body("PT15M")), FakeResponse(200, body)])
    frame = client.values("BE", "x", provider="entsoe", start="2026-11-01", end="2026-11-01")
    assert list(frame.columns) == ["p50", "p90", "knowledge_time", "annotation"]


def test_retries_after_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr("energy_price_predictor.rebase_grid.time.sleep", sleeps.append)
    client, _ = make_client(
        [
            FakeResponse(429, {"error": "rate_limited"}, {"Retry-After": "3"}),
            FakeResponse(200, {"name": "me"}),
        ]
    )
    assert client.me() == {"name": "me"}
    assert sleeps == [3.0]


def test_error_carries_closest_names():
    client, _ = make_client(
        [FakeResponse(404, {"error": "not_found", "message": "no such", "closest": ["DK-DK1"]})]
    )
    with pytest.raises(RebaseGridError) as info:
        client.entity("DK-DK")
    assert info.value.status == 404
    assert info.value.closest == ["DK-DK1"]


def test_snapshots_follow_cursor_into_long_frame():
    snap = lambda as_of, value: {  # noqa: E731
        "as_of": as_of,
        **points_body(["2026-11-03T00:00:00Z"], [value]),
    }
    client, session = make_client(
        [
            FakeResponse(200, {"snapshots": [snap("2026-11-02T10:00:00Z", 1.0)], "next": "abc"}),
            FakeResponse(200, {"snapshots": [snap("2026-11-02T11:00:00Z", 2.0)], "next": None}),
        ]
    )
    frame = client.snapshots("FI", "x", provider="entsoe", start="2026-11-03", end="2026-11-03", as_of="PT1H")
    assert list(frame["value"]) == [1.0, 2.0]
    assert list(frame.columns[:2]) == ["as_of", "valid_time"]
    assert session.calls[1][2]["params"]["cursor"] == "abc"


def test_many_values_collects_series_and_errors():
    page = {
        "series": [
            {"entity": "SE-SE3", "name": PRICE, "provider": "entsoe", **points_body(["2026-11-03T00:00:00Z"], [5.0])},
            {"entity": "XX", "name": PRICE, "provider": "entsoe", "error": {"code": "not_found"}},
        ],
        "next": None,
    }
    client, _ = make_client([FakeResponse(200, page)])
    frame = client.many_values(
        [("SE-SE3", PRICE, "entsoe"), ("XX", PRICE, "entsoe")], start="2026-11-03", end="2026-11-03"
    )
    assert list(frame["entity"]) == ["SE-SE3"]
    assert len(frame.attrs["errors"]) == 1


# ------------------------------------------------------------------ live tests

load_dotenv()
live = pytest.mark.skipif(not os.environ.get(API_KEY_ENV_VAR), reason=f"{API_KEY_ENV_VAR} not set")


@pytest.mark.live
@live
def test_live_key_and_day_ahead_price():
    grid = RebaseGridClient()
    assert grid.me()["email"]
    frame = grid.values("DK-DK1", PRICE, provider="entsoe", start="2026-11-03T12:00Z", end="2026-11-03T13:00Z")
    assert len(frame) == 5  # quarter hours, inclusive end
    assert frame["value"].notna().all()
