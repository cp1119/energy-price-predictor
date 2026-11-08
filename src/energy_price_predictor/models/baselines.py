"""Naive benchmarks from the electricity price forecasting literature."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_price_predictor.models.base import Forecaster

WEEKLY_LAG_DAYS = {0, 5, 6}  # Monday, Saturday, Sunday copy the same day last week


class NaiveForecaster(Forecaster):
    """Standard EPF naive rule: Mon/Sat/Sun repeat last week's day, other days repeat yesterday."""

    name = "naive"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        use_weekly = rows["dow"].isin(WEEKLY_LAG_DAYS)
        return np.where(use_weekly, rows["price_lag7"], rows["price_lag1"])


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the same day last week."""

    name = "seasonal_naive"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["price_lag7"].to_numpy()
