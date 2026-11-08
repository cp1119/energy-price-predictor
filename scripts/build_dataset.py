"""Fetch every registered series from Rebase Grid into the local Parquet cache.

    uv run python scripts/build_dataset.py            # incremental update to now
    uv run python scripts/build_dataset.py --only price_de load_fc

Prints a coverage report on the 15-minute grid afterwards.
"""

import argparse

import pandas as pd

from energy_price_predictor import RebaseGridClient
from energy_price_predictor.data import HISTORY_START, REGISTRY, load_panel, update_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="series keys to update (default: all)")
    args = parser.parse_args()

    client = RebaseGridClient()
    # Forecasts reach into D+1 (and beyond), so fetch two days past now.
    end = pd.Timestamp.now(tz="UTC").floor("D") + pd.Timedelta(days=2)
    specs = [spec for spec in REGISTRY if not args.only or spec.key in args.only]
    for spec in specs:
        series = update_cache(spec, end, client=client)
        print(f"{spec.key:15s} {len(series):>9,d} points  {series.index.min()} -> {series.index.max()}")

    panel_end = pd.Timestamp.now(tz="UTC").floor("D")
    panel = load_panel(HISTORY_START, panel_end, keys=[spec.key for spec in specs])
    coverage = panel.notna().groupby(panel.index.year).mean().T
    print("\nShare of 15-minute periods with a value, by year:")
    print((coverage * 100).round(1).to_string())


if __name__ == "__main__":
    main()
