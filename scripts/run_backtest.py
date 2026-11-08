"""Walk-forward backtest of DE-LU day-ahead price models on the cached data.

    uv run python scripts/run_backtest.py --models naive seasonal_naive --start 2024-11-01 --end 2026-10-31

Writes per-period forecasts and the metrics table to results/. Run scripts/build_dataset.py
first.
"""

import argparse
from pathlib import Path

import pandas as pd

from energy_price_predictor.backtest import breakdown, dm_matrix, metrics_table, run_backtest
from energy_price_predictor.data import HISTORY_START, TARGET_KEY, load_panel
from energy_price_predictor.features import build_feature_table
from energy_price_predictor.models import MODELS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ASSUMPTION_NOTE = (
    "Note: features use TSO day-ahead wind/solar forecasts, which are published after the 12:00 CET "
    "gate closure. Live errors will be somewhat higher than these."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["naive", "seasonal_naive"], choices=sorted(MODELS))
    parser.add_argument("--start", default="2024-11-01", help="first target day (local)")
    parser.add_argument("--end", default="2026-10-31", help="last target day (local)")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    target_days = pd.date_range(args.start, args.end, freq="D")
    panel_end = (target_days[-1] + pd.Timedelta(days=2)).tz_localize("UTC")
    panel = load_panel(HISTORY_START, panel_end)
    table = build_feature_table(panel)

    models = [MODELS[name]() for name in args.models]
    results = run_backtest(models, table, panel[TARGET_KEY], target_days, progress=args.progress)

    metrics = metrics_table(results)
    print(f"\nTest period {args.start} -> {args.end} ({len(target_days)} days), EUR/MWh\n")
    print(metrics.round(3).to_string())
    print("\nMAE by year:")
    print(breakdown(results, "year").round(2).to_string())
    if len(models) > 1:
        print("\nDiebold-Mariano one-sided p-values (column model beats row model):")
        print(dm_matrix(results).round(4).to_string())
    print(f"\n{ASSUMPTION_NOTE}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{args.start}_{args.end}_{'-'.join(args.models)}"
    results.to_parquet(RESULTS_DIR / f"forecasts_{stem}.parquet")
    metrics.to_csv(RESULTS_DIR / f"metrics_{stem}.csv")
    print(f"Saved results/forecasts_{stem}.parquet and results/metrics_{stem}.csv")


if __name__ == "__main__":
    main()
