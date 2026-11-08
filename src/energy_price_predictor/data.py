"""Series registry, Parquet cache and the 15-minute panel used by the DE-LU price model.

Every input series is declared once in :data:`REGISTRY` with an availability rule saying how
far into the future its values are usable at the forecast decision time (day D, 10:00
Europe/Berlin, for delivery day D+1):

* ``"through_d"``: only delivery periods up to the end of day D. Day-ahead prices are like
  this: all SDAC zones clear at the same time, so no zone's D+1 price is known before ours.
* ``"through_d_plus_1"``: forecasts for D+1 are usable.

Historical ``knowledge_time`` values in Rebase are backfill scrape times, not publication
times, so ``as_of`` cannot enforce these rules on history. :mod:`features` applies them.

Assumption: the TSO day-ahead wind/solar forecasts are published around 18:00-21:00 CET on
D, after gate closure. They are marked ``"through_d_plus_1"`` anyway (a project decision), which
makes backtest errors somewhat optimistic relative to a live run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from energy_price_predictor.rebase_grid import RebaseGridClient

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
HISTORY_START = pd.Timestamp("2018-10-01", tz="UTC")
PANEL_FREQ = "15min"
REFRESH_OVERLAP = pd.Timedelta(days=3)

PRICE = "price.day_ahead.day_ahead_auction_price"
TARGET_KEY = "price_de"

Availability = Literal["through_d", "through_d_plus_1"]


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    entity: str
    name: str
    provider: str
    availability: Availability
    scale: float = 1.0  # multiplier applied in load_panel, e.g. SMARD MWh per quarter-hour -> MW


NEIGHBOUR_ZONES = {
    "fr": "FR",
    "nl": "NL",
    "be": "BE",
    "at": "AT",
    "ch": "CH",
    "pl": "PL",
    "cz": "CZ",
    "dk1": "DK-DK1",
    "dk2": "DK-DK2",
    "no2": "NO-NO2",
    "se4": "SE-SE4",
}

REGISTRY: tuple[SeriesSpec, ...] = (
    SeriesSpec(TARGET_KEY, "DE-LU", PRICE, "entsoe", "through_d"),
    SeriesSpec("load_fc", "DE-LU", "consumption.load_forecast.day_ahead_total_load_forecast", "entsoe", "through_d_plus_1"),
    SeriesSpec("load_fc_smard", "DE-LU", "consumption.load_forecast.total_load_forecast", "smard", "through_d_plus_1", scale=4.0),
    SeriesSpec("solar_fc", "DE-LU", "production.forecast.solar_generation_forecast.day_ahead", "entsoe", "through_d_plus_1"),
    SeriesSpec("wind_on_fc", "DE-LU", "production.forecast.wind_generation_forecast.onshore_day_ahead", "entsoe", "through_d_plus_1"),
    SeriesSpec("wind_off_fc", "DE-LU", "production.forecast.wind_generation_forecast.offshore_day_ahead", "entsoe", "through_d_plus_1"),
    *(
        SeriesSpec(f"price_{short}", zone, PRICE, "entsoe", "through_d")
        for short, zone in NEIGHBOUR_ZONES.items()
    ),
)

SPECS_BY_KEY = {spec.key: spec for spec in REGISTRY}


def cache_path(key: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / f"{key}.parquet"


def read_cache(key: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.Series:
    """The cached values of one series (indexed by UTC ``valid_time``), or an empty series."""
    path = cache_path(key, cache_dir)
    if not path.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC", name="valid_time"), name=key)
    frame = pd.read_parquet(path)
    return frame["value"].rename(key)


def update_cache(
    spec: SeriesSpec,
    end: pd.Timestamp,
    *,
    client: RebaseGridClient,
    start: pd.Timestamp = HISTORY_START,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.Series:
    """Fetch what the cache is missing up to ``end`` and write it back.

    Re-fetches a few days before the last cached point so late revisions are picked up.
    """
    cached = read_cache(spec.key, cache_dir)
    fetch_start = start if cached.empty else max(start, cached.index.max() - REFRESH_OVERLAP)
    if fetch_start > end:
        return cached
    fetched = client.values(spec.entity, spec.name, provider=spec.provider, start=fetch_start, end=end)
    new_values = pd.to_numeric(fetched["value"], errors="coerce").rename(spec.key)
    combined = pd.concat([cached, new_values])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined.to_frame("value").to_parquet(cache_path(spec.key, cache_dir))
    return combined


def to_quarter_hours(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Put a series on the 15-minute UTC grid ``[start, end)``.

    Hourly values (prices before 2025-10-01) are repeated over their four quarter-hours. Only
    hours holding a single on-the-hour value are filled, so gaps in 15-minute data, and the
    periods after the last value, stay missing.
    """
    grid = pd.date_range(start, end, freq=PANEL_FREQ, inclusive="left", tz="UTC", name="valid_time")
    if series.empty:
        return pd.Series(float("nan"), index=grid, name=series.name)
    series = series.dropna()
    values = series.reindex(grid)
    points_per_hour = series.groupby(series.index.floor("h")).size()
    on_the_hour = series.index[series.index == series.index.floor("h")]
    hourly_hours = points_per_hour.index[points_per_hour == 1].intersection(on_the_hour)
    hour_of_slot = grid.floor("h")
    to_fill = values.isna().to_numpy() & hour_of_slot.isin(hourly_hours)
    values[to_fill] = series.reindex(hour_of_slot[to_fill]).to_numpy()
    return values


def load_panel(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    keys: list[str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Wide DataFrame on the 15-minute UTC grid ``[start, end)``, one column per series key.

    Values are multiplied by each spec's ``scale`` so power series are all in MW. Reads the
    cache only; run ``scripts/build_dataset.py`` first to fill it.
    """
    keys = keys if keys is not None else [spec.key for spec in REGISTRY]
    columns = {
        key: to_quarter_hours(read_cache(key, cache_dir), start, end) * SPECS_BY_KEY[key].scale
        for key in keys
    }
    return pd.DataFrame(columns)
