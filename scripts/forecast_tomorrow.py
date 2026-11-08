"""Forecast the DE-LU day-ahead price for one delivery day with the LEAR + LightGBM ensemble.

    uv run python scripts/forecast_tomorrow.py                 # tomorrow (local time)
    uv run python scripts/forecast_tomorrow.py --day 2026-11-05 --no-refresh   # a past day, vs actual

Refreshes the cache, fits both models on every day before the target day, and writes the
15-minute forecast to results/forecast_<day>.csv.

Caveat: the TSO wind/solar forecasts for D+1 are only published around 18:00-21:00 CET on D,
after the 12:00 gate closure. Run before then, the inputs are missing and the script says so;
the backtest assumed they were available.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from energy_price_predictor import RebaseGridClient
from energy_price_predictor.data import HISTORY_START, REGISTRY, TARGET_KEY, load_panel, update_cache
from energy_price_predictor.features import LOCAL_TZ, PERIODS_PER_DAY, build_feature_table, delivery_periods, to_periods
from energy_price_predictor.models import LEARForecaster, TunedGBMForecaster

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RENEWABLE_INPUTS = ["solar", "wind_on", "wind_off"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day", help="delivery day, local date (default: tomorrow)")
    parser.add_argument("--no-refresh", action="store_true", help="use the cache as it is")
    args = parser.parse_args()

    today = pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).normalize()
    target_day = pd.Timestamp(args.day) if args.day else today + pd.Timedelta(days=1)
    panel_end = delivery_periods(target_day)[-1] + pd.Timedelta(days=1)

    if not args.no_refresh:
        client = RebaseGridClient()
        for spec in REGISTRY:
            update_cache(spec, panel_end, client=client)

    panel = load_panel(HISTORY_START, panel_end)
    table = build_feature_table(panel)
    days = table.index.get_level_values("day")
    if target_day not in days:
        raise SystemExit(f"No feature rows for {target_day.date()}: is the cache up to date?")
    rows = table.xs(target_day, level="day", drop_level=False)

    missing = rows[RENEWABLE_INPUTS].isna().mean()
    if missing.max() > 0:
        print("WARNING: wind/solar forecasts for the target day are missing "
              f"({(missing * 100).round(0).to_dict()} % of periods). They are published after gate "
              "closure; this forecast is degraded and not comparable with the backtest.\n")

    train = table[days < target_day]
    forecasts = {}
    for model in (LEARForecaster(), TunedGBMForecaster()):
        print(f"Fitting {model.name} on {train.index.get_level_values('day').nunique()} days ...", flush=True)
        model.fit(train)
        slot_values = np.full(PERIODS_PER_DAY, np.nan)
        slot_values[rows["slot"].to_numpy()] = model.predict(rows)
        forecasts[model.name] = to_periods(target_day, slot_values)

    result = pd.DataFrame(forecasts)
    result["ensemble"] = result.mean(axis=1)
    actual = panel[TARGET_KEY].reindex(result.index)
    if actual.notna().any():
        result["actual"] = actual
    result.index = result.index.tz_convert(LOCAL_TZ)

    hourly = result.groupby(result.index.hour).mean()
    print(f"\nDE-LU day-ahead forecast for {target_day.date()} (EUR/MWh, hourly means of 15-minute values):")
    print(hourly.round(2).to_string())
    print(f"\nDaily mean {result['ensemble'].mean():.2f}, min {result['ensemble'].min():.2f}, "
          f"max {result['ensemble'].max():.2f}")
    if "actual" in result:
        for column in ("lear", "gbm_tuned", "ensemble"):
            print(f"MAE {column:10s} {np.abs(result[column] - result['actual']).mean():6.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"forecast_{target_day.date()}.csv"
    result.to_csv(path)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
