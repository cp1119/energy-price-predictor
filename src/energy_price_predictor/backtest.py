"""Walk-forward backtest, error metrics and the Diebold-Mariano test.

Forecasts are scored on real UTC delivery periods (see :func:`features.to_periods`), so DST
days count 92 or 100 periods and each period counts once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from energy_price_predictor.features import LOCAL_TZ, PERIODS_PER_DAY, to_periods
from energy_price_predictor.models.base import Forecaster

BENCHMARK = "naive"


def walk_forward(
    model: Forecaster,
    table: pd.DataFrame,
    target_days: pd.DatetimeIndex,
    *,
    progress: bool = False,
) -> pd.Series:
    """Forecast each target day using only rows of earlier days; returns a UTC-period series.

    The model is refitted every ``model.refit_every`` days. Days with less than
    ``model.min_train_days`` of history are skipped.
    """
    day_level = table.index.get_level_values("day")
    first_day = day_level.min()
    needs_fit = type(model).fit is not Forecaster.fit
    last_fit: pd.Timestamp | None = None
    pieces = []
    for count, day in enumerate(target_days):
        if (day - first_day).days < model.min_train_days or day not in day_level:
            continue
        if needs_fit and (last_fit is None or (day - last_fit).days >= model.refit_every):
            model.fit(table[day_level < day])
            last_fit = day
        rows = table.xs(day, level="day", drop_level=False)
        slot_values = np.full(PERIODS_PER_DAY, np.nan)
        slot_values[rows["slot"].to_numpy()] = model.predict(rows)
        pieces.append(to_periods(day, slot_values))
        if progress and count % 30 == 0:
            print(f"  {model.name}: {day.date()}", flush=True)
    if not pieces:
        return pd.Series(dtype=float, name=model.name)
    return pd.concat(pieces).rename(model.name)


def run_backtest(
    models: list[Forecaster],
    table: pd.DataFrame,
    actual: pd.Series,
    target_days: pd.DatetimeIndex,
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """Forecasts of every model plus the actual price, indexed by UTC delivery period."""
    columns = {model.name: walk_forward(model, table, target_days, progress=progress) for model in models}
    results = pd.DataFrame(columns)
    results.insert(0, "actual", actual.reindex(results.index))
    return results


def common_sample(results: pd.DataFrame) -> pd.DataFrame:
    """Rows where the actual and every model's forecast exist, so models are compared fairly."""
    return results.dropna()


def error_metrics(forecast: pd.Series, actual: pd.Series) -> dict[str, float]:
    error = forecast - actual
    hourly_error = forecast.resample("1h").mean() - actual.resample("1h").mean()
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mae_hourly": float(hourly_error.abs().mean()),
    }


def metrics_table(results: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """MAE, RMSE, hourly-average MAE and rMAE (relative to ``benchmark``) for every model."""
    sample = common_sample(results)
    models = [column for column in sample.columns if column != "actual"]
    table = pd.DataFrame({model: error_metrics(sample[model], sample["actual"]) for model in models}).T
    if benchmark in table.index:
        table["rmae"] = table["mae"] / table.loc[benchmark, "mae"]
    table.index.name = "model"
    return table


def breakdown(results: pd.DataFrame, by: str) -> pd.DataFrame:
    """MAE per model grouped by local ``"hour"``, ``"month"`` or ``"year"``."""
    sample = common_sample(results)
    local = sample.index.tz_convert(LOCAL_TZ)
    keys = {"hour": local.hour, "month": local.month, "year": local.year}[by]
    errors = sample.drop(columns="actual").sub(sample["actual"], axis=0).abs()
    return errors.groupby(np.asarray(keys)).mean().rename_axis(by)


def daily_mae(results: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error per local delivery day for every model."""
    sample = common_sample(results)
    errors = sample.drop(columns="actual").sub(sample["actual"], axis=0).abs()
    local_day = sample.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    return errors.groupby(local_day).mean()


def dm_test(results: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float]:
    """Diebold-Mariano test on daily MAE: H1 is that ``model_b`` is more accurate than ``model_a``.

    Uses the multivariate (daily-averaged) form common in EPF studies. Returns
    (statistic, one-sided p-value); a small p-value favours ``model_b``.
    """
    daily = daily_mae(results)
    differential = daily[model_a] - daily[model_b]
    count = len(differential)
    statistic = differential.mean() / (differential.std(ddof=1) / np.sqrt(count))
    p_value = 1 - stats.norm.cdf(statistic)
    return float(statistic), float(p_value)


def dm_matrix(results: pd.DataFrame) -> pd.DataFrame:
    """One-sided DM p-values: cell (a, b) tests whether column model b beats row model a."""
    models = [column for column in results.columns if column != "actual"]
    matrix = pd.DataFrame(np.nan, index=models, columns=models)
    for model_a in models:
        for model_b in models:
            if model_a != model_b:
                matrix.loc[model_a, model_b] = dm_test(results, model_a, model_b)[1]
    return matrix


def add_ensemble(
    results: pd.DataFrame,
    members: list[str],
    weights: dict[str, float] | None = None,
    name: str = "ensemble",
) -> pd.DataFrame:
    """Add a weighted average of ``members`` (equal weights by default) as column ``name``."""
    weights = weights or {member: 1.0 for member in members}
    total = sum(weights[member] for member in members)
    combined = sum(results[member] * weights[member] for member in members) / total
    return results.assign(**{name: combined})


def inverse_mae_weights(results: pd.DataFrame, members: list[str]) -> dict[str, float]:
    """Ensemble weights proportional to 1 / MAE, from a validation-period backtest."""
    sample = common_sample(results[["actual", *members]])
    return {member: 1.0 / (sample[member] - sample["actual"]).abs().mean() for member in members}
