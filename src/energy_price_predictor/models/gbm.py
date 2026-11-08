"""Global LightGBM model across all delivery slots.

Tree models cannot extrapolate beyond the price levels seen in training, and DE-LU prices
moved through very different regimes (2021-23 gas crisis). So by default the model learns
the target minus day D's mean price, with D's mean still available as a feature, and adds
the level back afterwards. Recent days get more weight (exponential decay).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from energy_price_predictor.features import feature_columns
from energy_price_predictor.models.base import Forecaster

DEFAULT_PARAMS = {
    "objective": "l1",
    "learning_rate": 0.05,
    "n_estimators": 600,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 1.0,
    "verbose": -1,
}
LEVEL_COLUMN = "price_d_mean"


class GBMForecaster(Forecaster):
    name = "gbm"

    def __init__(
        self,
        params: dict | None = None,
        *,
        relative_to_level: bool = True,
        half_life_days: float | None = 365.0,
        train_start: str = "2019-01-01",
        refit_every: int = 7,
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.relative_to_level = relative_to_level
        self.half_life_days = half_life_days
        self.train_start = pd.Timestamp(train_start)
        self.refit_every = refit_every
        self.min_train_days = 365

    def _training_rows(self, train: pd.DataFrame) -> pd.DataFrame:
        days = train.index.get_level_values("day")
        rows = train[days >= self.train_start]
        return rows.dropna(subset=["target", LEVEL_COLUMN])

    def _offset(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[LEVEL_COLUMN].to_numpy() if self.relative_to_level else np.zeros(len(rows))

    def sample_weights(self, rows: pd.DataFrame) -> np.ndarray | None:
        if self.half_life_days is None:
            return None
        days = rows.index.get_level_values("day")
        age_days = np.asarray((days.max() - days).days, dtype=float)
        return 0.5 ** (age_days / self.half_life_days)

    def fit(self, train: pd.DataFrame) -> None:
        rows = self._training_rows(train)
        self.columns = feature_columns(rows)
        target = rows["target"].to_numpy() - self._offset(rows)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(rows[self.columns], target, sample_weight=self.sample_weights(rows))

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(rows[self.columns]) + self._offset(rows)

    def feature_importance(self) -> pd.Series:
        gains = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.columns).sort_values(ascending=False)


TUNED_PARAMS_PATH = Path(__file__).resolve().parents[3] / "results" / "gbm_best_params.json"


class TunedGBMForecaster(GBMForecaster):
    """GBM with the parameters found by scripts/tune_gbm.py on the validation year."""

    name = "gbm_tuned"

    def __init__(self, refit_every: int = 7):
        tuned = json.loads(TUNED_PARAMS_PATH.read_text())
        half_life = tuned.pop("half_life_days")
        relative = tuned.pop("relative_to_level")
        super().__init__(tuned, relative_to_level=relative, half_life_days=half_life or None, refit_every=refit_every)
