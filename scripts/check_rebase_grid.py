"""Check the Rebase Grid API key and read a sample of day-ahead prices."""

import pandas as pd

from energy_price_predictor import RebaseGridClient

ZONE = "DK-DK1"
NAME = "price.day_ahead.day_ahead_auction_price"


def main() -> None:
    grid = RebaseGridClient()
    owner = grid.me()
    print(f"Key OK: {owner['name']} ({owner['scope']} scope)")

    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end - pd.Timedelta(days=2)
    prices = grid.values(ZONE, NAME, provider="entsoe", start=start, end=end)
    print(f"\n{ZONE} day-ahead price ({prices.attrs['unit']}), {len(prices)} points:")
    print(prices["value"].describe().round(2).to_string())
    print("\nLast five:")
    print(prices.tail().to_string())


if __name__ == "__main__":
    main()
