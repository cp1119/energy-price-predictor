"""The interface every forecaster implements for the walk-forward backtest."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Predicts the target of one delivery day from rows of the feature table.

    The backtest calls :meth:`fit` with all rows whose day is before the target day (every
    ``refit_every`` days), then :meth:`predict` with the target day's rows.
    """

    name: str = "forecaster"
    refit_every: int = 1  # days between refits
    min_train_days: int = 0  # days of history needed before the first forecast

    def fit(self, train: pd.DataFrame) -> None:  # noqa: B027 - optional for rule-based models
        """Train on rows indexed by (day, slot). Rule-based models ignore this."""

    @abstractmethod
    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """Forecasts for the given rows (one target day), in row order."""
