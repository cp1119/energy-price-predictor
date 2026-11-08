"""Feature table for forecasting the DE-LU day-ahead price of delivery day T = D + 1.

Everything is laid out on a grid of local (Europe/Berlin) delivery days by quarter-hour of
day (0..95):

* Spring-forward days have 92 real periods. The missing slots 8-11 (02:00-03:00) are
  forward-filled so that lag features line up, then dropped from the target rows.
* Fall-back days have 100 periods. The two 02:xx hours are averaged into slots 8-11, and
  :func:`to_periods` repeats the forecast for both when mapping back to UTC.

Availability (see :mod:`data`): for target day T, prices are known up to the end of day D =
T - 1, and load/wind/solar forecasts up to the end of T. :func:`mask_unavailable` blanks
everything later, and the leakage test checks that features never depend on blanked values.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from energy_price_predictor.data import NEIGHBOUR_ZONES, REGISTRY, TARGET_KEY

LOCAL_TZ = "Europe/Berlin"
PERIODS_PER_DAY = 96
SPRING_FORWARD_SLOTS = [8, 9, 10, 11]
QH_ERA_START = pd.Timestamp("2025-10-01")
DECISION_HOUR = 10
PRICE_LAGS = [1, 2, 3, 7]
GERMAN_STATES = ["BB", "BE", "BW", "BY", "HB", "HE", "HH", "MV", "NI", "NW", "RP", "SH", "SL", "SN", "ST", "TH"]


# --------------------------------------------------------------------------- time helpers


def delivery_periods(day: pd.Timestamp) -> pd.DatetimeIndex:
    """UTC start times of every 15-minute delivery period of local day ``day``."""
    start = pd.Timestamp(day).normalize().tz_localize(LOCAL_TZ)
    end = (pd.Timestamp(day).normalize() + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    periods = pd.date_range(start, end, freq="15min", inclusive="left")
    return periods.tz_convert("UTC").rename("valid_time")


def decision_time(target_day: pd.Timestamp) -> pd.Timestamp:
    """When the forecast for ``target_day`` is made: D 10:00 local, in UTC."""
    day_before = pd.Timestamp(target_day).normalize() - pd.Timedelta(days=1)
    return (day_before + pd.Timedelta(hours=DECISION_HOUR)).tz_localize(LOCAL_TZ).tz_convert("UTC")


def local_slot(index: pd.DatetimeIndex) -> np.ndarray:
    """Quarter-hour of the local day (0..95) for UTC timestamps."""
    local = index.tz_convert(LOCAL_TZ)
    return np.asarray(local.hour * 4 + local.minute // 15)


def to_periods(target_day: pd.Timestamp, slot_values: pd.Series | np.ndarray) -> pd.Series:
    """Map 96 slot values for ``target_day`` onto its real UTC delivery periods."""
    periods = delivery_periods(target_day)
    values = np.asarray(slot_values, dtype=float)
    return pd.Series(values[local_slot(periods)], index=periods)


def periods_in_day(days: pd.DatetimeIndex) -> pd.Series:
    """Number of real 15-minute periods (92, 96 or 100) in each local day."""
    starts = days.tz_localize(LOCAL_TZ)
    ends = (days + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    hours = (ends - starts) / pd.Timedelta(minutes=15)
    return pd.Series(np.asarray(hours, dtype=int), index=days)


# --------------------------------------------------------------------------- daily matrices


def daily_matrix(series: pd.Series) -> pd.DataFrame:
    """A UTC series as local day (rows, naive dates) by slot (columns 0..95)."""
    series = series.dropna()
    local = series.index.tz_convert(LOCAL_TZ)
    frame = pd.DataFrame(
        {
            "day": local.tz_localize(None).normalize(),
            "slot": local.hour * 4 + local.minute // 15,
            "value": series.to_numpy(),
        }
    )
    matrix = frame.groupby(["day", "slot"])["value"].mean().unstack("slot")
    matrix = matrix.reindex(columns=range(PERIODS_PER_DAY))
    if matrix.empty:
        return matrix
    all_days = pd.date_range(matrix.index.min(), matrix.index.max(), freq="D")
    matrix = matrix.reindex(all_days)
    spring_days = periods_in_day(all_days) == 92
    matrix.loc[spring_days.to_numpy()] = matrix.loc[spring_days.to_numpy()].ffill(axis=1, limit=4)
    matrix.index.name = "day"
    matrix.columns.name = "slot"
    return matrix


def daily_matrices(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily matrices for every panel column, all on one shared day index."""
    matrices = {key: daily_matrix(panel[key]) for key in panel.columns}
    starts = [m.index.min() for m in matrices.values() if not m.empty]
    ends = [m.index.max() for m in matrices.values() if not m.empty]
    days = pd.date_range(min(starts), max(ends), freq="D", name="day")
    return {key: m.reindex(days) for key, m in matrices.items()}


# --------------------------------------------------------------------------- availability


def availability_horizon(target_day: pd.Timestamp, availability: str) -> pd.Timestamp:
    """First UTC instant whose values are NOT usable when forecasting ``target_day``."""
    target_day = pd.Timestamp(target_day).normalize()
    if availability == "through_d":
        first_unknown_day = target_day
    elif availability == "through_d_plus_1":
        first_unknown_day = target_day + pd.Timedelta(days=1)
    else:
        raise ValueError(f"unknown availability {availability!r}")
    return first_unknown_day.tz_localize(LOCAL_TZ).tz_convert("UTC")


def mask_unavailable(panel: pd.DataFrame, target_day: pd.Timestamp, fill: float = np.nan) -> pd.DataFrame:
    """Copy of ``panel`` with every value unusable for ``target_day`` replaced by ``fill``."""
    rules = {spec.key: spec.availability for spec in REGISTRY}
    masked = panel.copy()
    for column in masked.columns:
        horizon = availability_horizon(target_day, rules[column])
        masked.loc[masked.index >= horizon, column] = fill
    return masked


# --------------------------------------------------------------------------- calendar


@lru_cache(maxsize=None)
def _holiday_calendars(first_year: int, last_year: int) -> tuple[holidays.HolidayBase, list[holidays.HolidayBase]]:
    years = range(first_year, last_year + 1)
    national = holidays.country_holidays("DE", years=years)
    states = [holidays.country_holidays("DE", subdiv=state, years=years) for state in GERMAN_STATES]
    return national, states


def calendar_features(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-day calendar features: weekday, holidays, bridge days, DST and market era."""
    national, states = _holiday_calendars(days.min().year - 1, days.max().year + 1)
    dates = [day.date() for day in days]
    is_holiday = np.array([date in national for date in dates])
    state_share = np.array([np.mean([date in state for state in states]) for date in dates])

    def holiday_on(offset: int) -> np.ndarray:
        return np.array([(day + pd.Timedelta(days=offset)).date() in national for day in days])

    weekday = np.asarray(days.day_of_week)
    is_bridge = ((weekday == 0) & holiday_on(1)) | ((weekday == 4) & holiday_on(-1))
    is_bridge &= ~is_holiday
    month_day = days.month * 100 + days.day
    is_christmas = (month_day >= 1224) | (month_day <= 101)
    day_of_year = np.asarray(days.day_of_year)

    return pd.DataFrame(
        {
            "dow": weekday,
            "month": np.asarray(days.month),
            "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "is_holiday": is_holiday.astype(int),
            "holiday_state_share": state_share,
            "is_bridge": is_bridge.astype(int),
            "is_christmas": np.asarray(is_christmas, dtype=int),
            "n_periods": periods_in_day(days).to_numpy(),
            "is_qh_era": np.asarray(days >= QH_ERA_START, dtype=int),
        },
        index=days,
    )


# --------------------------------------------------------------------------- feature table


def _stack(matrix: pd.DataFrame, name: str) -> pd.Series:
    return matrix.stack(future_stack=True).rename(name)


def _per_day(values: pd.Series, name: str, index: pd.MultiIndex) -> pd.Series:
    """Broadcast a per-day series onto the (day, slot) index."""
    return pd.Series(values.reindex(index.get_level_values("day")).to_numpy(), index=index, name=name)


def build_feature_table(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (target day, slot) with the target and every model feature.

    Row (T, s) only uses prices up to the end of T - 1 and forecasts up to the end of T. Rows
    for slots that do not exist (spring-forward 02:xx) are dropped.
    """
    m = daily_matrices(panel)
    days = m[TARGET_KEY].index
    price = m[TARGET_KEY]
    columns: list[pd.Series] = [_stack(price, "target")]
    index = columns[0].index

    # Own price lags at the same slot, and summary statistics of day D.
    for lag in PRICE_LAGS:
        columns.append(_stack(price.shift(lag), f"price_lag{lag}"))
    day_d = price.shift(1)
    columns += [
        _per_day(day_d.mean(axis=1), "price_d_mean", index),
        _per_day(day_d.min(axis=1), "price_d_min", index),
        _per_day(day_d.max(axis=1), "price_d_max", index),
        _per_day(day_d.std(axis=1), "price_d_std", index),
        _per_day(day_d[PERIODS_PER_DAY - 1], "price_d_last", index),
        _per_day(price.shift(7).mean(axis=1), "price_d6_mean", index),
    ]

    # Fundamentals for day T (forecasts), and how they differ from day D.
    load = m["load_fc"].fillna(m["load_fc_smard"])
    solar = m["solar_fc"]
    wind = m["wind_on_fc"] + m["wind_off_fc"]
    residual = load - wind - solar
    wind_scale = wind.max(axis=1).rolling(365, min_periods=30).max()
    solar_scale = solar.max(axis=1).rolling(365, min_periods=30).max()
    columns += [
        _stack(load, "load"),
        _stack(solar, "solar"),
        _stack(m["wind_on_fc"], "wind_on"),
        _stack(m["wind_off_fc"], "wind_off"),
        _stack(residual, "residual_load"),
        _stack((wind + solar) / load, "renewable_share"),
        _stack(wind.div(wind_scale, axis=0), "wind_norm"),
        _stack(solar.div(solar_scale, axis=0), "solar_norm"),
        _stack(residual.diff(4, axis=1), "residual_ramp_1h"),
        _stack(residual - residual.shift(1), "residual_vs_d"),
        _stack(residual.shift(1), "residual_lag1"),
        _per_day(residual.mean(axis=1), "residual_day_mean", index),
        _per_day(residual.max(axis=1), "residual_day_max", index),
        _per_day(residual.min(axis=1), "residual_day_min", index),
        _per_day(wind.mean(axis=1), "wind_day_mean", index),
        _per_day(solar.sum(axis=1) / 4, "solar_day_mwh", index),
        _per_day(load.mean(axis=1), "load_day_mean", index),
    ]

    # Neighbour prices of day D (same slot and daily mean) and their spread to DE-LU.
    for short in NEIGHBOUR_ZONES:
        neighbour_d = m[f"price_{short}"].shift(1)
        columns.append(_stack(neighbour_d, f"price_{short}_lag1"))
        columns.append(_per_day(neighbour_d.mean(axis=1) - day_d.mean(axis=1), f"spread_{short}_d_mean", index))

    table = pd.concat(columns, axis=1)
    slots = table.index.get_level_values("slot")
    table.insert(1, "slot", np.asarray(slots))
    table.insert(2, "hour", np.asarray(slots) // 4)
    calendar = calendar_features(days)
    table = table.join(calendar, on="day")

    missing_slot = (table["n_periods"] == 92) & table["slot"].isin(SPRING_FORWARD_SLOTS)
    return table[~missing_slot.to_numpy()]


FEATURE_COLUMNS_EXCLUDED = {"target"}


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column not in FEATURE_COLUMNS_EXCLUDED]
