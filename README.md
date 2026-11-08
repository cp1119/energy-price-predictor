# energy-price-predictor

Energy price prediction using European power system data from the
[Rebase Grid API](https://grid.rebase.energy/docs/v1/).

## Setup

```bash
uv sync
cp .env.example .env   # then set REBASE_GRID_API_KEY
uv run python scripts/check_rebase_grid.py
```

The check script confirms the key and reads the last two days of the DK1 day-ahead price.

## Using the Rebase Grid client

```python
from energy_price_predictor import RebaseGridClient

grid = RebaseGridClient()  # key from REBASE_GRID_API_KEY / .env

# Discover
grid.entities()                                  # flat table of every entity (countries, price zones, ...)
grid.timeseries_names("DK-DK1", category="price")
grid.search(entity="DE-LU", category="production", provider="entsoe")
grid.metadata("DK-DK1", "price.day_ahead.day_ahead_auction_price")

# Read: DataFrame indexed by valid_time (UTC), columns value / knowledge_time / annotation
prices = grid.values(
    "DK-DK1",
    "price.day_ahead.day_ahead_auction_price",
    provider="entsoe",
    start="2020-01-01",
    end="2026-11-01",
)

# As known at an instant (no look-ahead when building training sets)
grid.values(..., as_of="2026-10-01T10:00Z")

# Forecast vintages: one snapshot per period, long format with an as_of column
grid.snapshots(
    "FI", "consumption.load_forecast.day_ahead_total_load_forecast",
    provider="entsoe", start="2026-10-01", end="2026-10-03", as_of="PT1H",
)

# Latest values of many series in one call
grid.many_values(
    [("SE-SE3", "price.day_ahead.day_ahead_auction_price", "entsoe"),
     ("NO-NO1", "price.day_ahead.day_ahead_auction_price", "entsoe")],
    start="2026-11-01", end="2026-11-02",
)
```

Notes from the [API docs](https://grid.rebase.energy/docs/v1/guides/efficient-calls/):

- All times are UTC; `start` and `end` are inclusive. Naive timestamps are treated as UTC.
- A values call may span 100,000 points; `values()` splits longer ranges automatically using the
  series' resolution from its metadata.
- 240 requests/minute per key. 429/503 responses are retried after `Retry-After`.
- Day-ahead prices are named `price.day_ahead.day_ahead_auction_price` in every zone.
- Values carry `knowledge_time`, so a backtest can drop anything not yet published at forecast time.

Full docs for agents: [llms.txt](https://grid.rebase.energy/docs/v1/llms.txt),
[llms-full.txt](https://grid.rebase.energy/docs/v1/llms-full.txt),
[openapi.json](https://grid.rebase.energy/docs/v1/openapi.json) (also linked in `rebase-api.md`).

## Example notebook

[notebooks/rebase_grid_examples.ipynb](notebooks/rebase_grid_examples.ipynb) walks through
discovery, a multi-year DE-LU price history, a zone comparison, the hour-by-month price profile,
price against load/wind/solar forecasts, and using `as_of` to avoid look-ahead. Open it in VS Code
and pick the `.venv` kernel (`uv sync` installs `ipykernel` and `matplotlib`).

## Tests

```bash
uv run pytest            # unit tests (offline) + live tests if REBASE_GRID_API_KEY is set
uv run pytest -m "not live"
```
