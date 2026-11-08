"""Combine saved backtest forecasts, add an ensemble and print the evaluation report.

    uv run python scripts/report.py results/forecasts_A.parquet results/forecasts_B.parquet \
        --ensemble lear gbm_tuned [--weights-from results/forecasts_validation.parquet]

Columns from later files replace same-named columns from earlier ones. Without
--weights-from the ensemble is an equal-weight average.
"""

import argparse
from pathlib import Path

import pandas as pd

from energy_price_predictor.backtest import (
    add_ensemble,
    breakdown,
    dm_matrix,
    inverse_mae_weights,
    metrics_table,
)

ASSUMPTION_NOTE = (
    "Note: features use TSO day-ahead wind/solar forecasts, which are published after the 12:00 CET "
    "gate closure. Live errors will be somewhat higher than these."
)


def merge_forecasts(paths: list[Path]) -> pd.DataFrame:
    merged = pd.read_parquet(paths[0])
    for path in paths[1:]:
        frame = pd.read_parquet(path)
        models = [column for column in frame.columns if column != "actual"]
        merged = merged.drop(columns=[m for m in models if m in merged.columns]).join(frame[models], how="outer")
        merged["actual"] = merged["actual"].fillna(frame["actual"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("forecasts", nargs="+", type=Path)
    parser.add_argument("--ensemble", nargs="*", default=[], help="member models to average")
    parser.add_argument("--weights-from", type=Path, help="validation forecasts for 1/MAE weights")
    parser.add_argument("--save", type=Path, help="write the merged forecasts here")
    args = parser.parse_args()

    results = merge_forecasts(args.forecasts)
    if args.ensemble:
        weights = None
        if args.weights_from:
            weights = inverse_mae_weights(pd.read_parquet(args.weights_from), args.ensemble)
            total = sum(weights.values())
            print("Ensemble weights:", {m: round(w / total, 3) for m, w in weights.items()})
        results = add_ensemble(results, args.ensemble, weights)

    period = results.dropna().index
    print(f"\nScored periods: {period.min()} -> {period.max()} ({len(period):,d}), EUR/MWh\n")
    print(metrics_table(results).round(3).to_string())
    print("\nMAE by year:")
    print(breakdown(results, "year").round(2).to_string())
    print("\nMAE by local hour:")
    print(breakdown(results, "hour").round(1).to_string())
    print("\nDiebold-Mariano one-sided p-values (column model beats row model):")
    print(dm_matrix(results.dropna()).round(4).to_string())
    print(f"\n{ASSUMPTION_NOTE}")
    if args.save:
        results.to_parquet(args.save)
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
