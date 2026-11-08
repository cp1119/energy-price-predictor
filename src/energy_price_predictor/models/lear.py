"""LEAR: Lasso-estimated autoregressive model (Lago, De Ridder, Vrancx & De Schutter, 2021).

One linear model per delivery slot, estimated with the Lasso on variance-stabilised data and
averaged over several calibration windows. Adaptations to 15-minute DE-LU data:

* Inputs are hourly means (24 per variable) of: prices of D, D-1, D-2 and D-6 (the table's
  ``price_lag1/2/3/7``), the T load, wind and solar forecasts, and day D's residual load, plus
  day-of-week dummies. That is 199 features instead of ~1,250 at 15-minute detail.
* The Lasso penalty is selected by AIC (``LassoLarsIC``) on the longest window, where there
  are more samples than features, then reused for the shorter windows.
* Every column (and the target) is scaled by its median and MAD over the window, then passed
  through asinh, which tames spikes and negative prices.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC

from energy_price_predictor.features import PERIODS_PER_DAY
from energy_price_predictor.models.base import Forecaster

HOURLY_INPUTS = ["price_lag1", "price_lag2", "price_lag3", "price_lag7", "load", "wind", "solar", "residual_lag1"]
DEFAULT_WINDOWS = (56, 84, 364, 728)
MAD_TO_STD = 1.4826


def lear_design(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide per-day inputs (day x 199) and targets (day x 96 slots) from feature-table rows."""
    frame = table[["slot", "dow", "target", "price_lag1", "price_lag2", "price_lag3", "price_lag7",
                   "load", "solar", "residual_lag1"]].copy()
    frame["wind"] = table["wind_on"] + table["wind_off"]
    frame["hour"] = frame["slot"] // 4
    days = frame.index.get_level_values("day")

    hourly = frame.groupby([days, frame["hour"]])[HOURLY_INPUTS].mean().unstack("hour")
    all_columns = pd.MultiIndex.from_product([HOURLY_INPUTS, range(24)])
    hourly = hourly.reindex(columns=all_columns)
    hourly = hourly.T.groupby(level=0, group_keys=False).apply(lambda block: block.ffill().bfill()).T
    hourly.columns = [f"{name}_h{hour:02d}" for name, hour in hourly.columns]

    dow = frame.groupby(days)["dow"].first()
    dummies = pd.get_dummies(dow, prefix="dow").astype(float).reindex(columns=[f"dow_{d}" for d in range(7)], fill_value=0.0)
    inputs = hourly.join(dummies)

    targets = frame.groupby([days, frame["slot"]])["target"].first().unstack("slot")
    targets = targets.reindex(columns=range(PERIODS_PER_DAY))
    inputs.index.name = targets.index.name = "day"
    return inputs, targets


class RobustAsinh:
    """Median/MAD scaling followed by asinh, fitted per column."""

    def fit(self, values: np.ndarray) -> RobustAsinh:
        self.median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - self.median), axis=0) * MAD_TO_STD
        self.mad = np.where(mad > 1e-9, mad, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.arcsinh((values - self.median) / self.mad)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.sinh(values) * self.mad + self.median


def _fit_slot(inputs: np.ndarray, target: np.ndarray, windows: tuple[int, ...]) -> list[tuple]:
    """Fit one slot's models, one per window; returns (input scaler, target scaler, model) triples."""
    valid = ~np.isnan(target) & ~np.isnan(inputs).any(axis=1)
    inputs, target = inputs[valid], target[valid]
    fitted = []
    alpha = None
    for window in sorted(windows, reverse=True):
        x_window, y_window = inputs[-window:], target[-window:]
        x_scaler = RobustAsinh().fit(x_window)
        y_scaler = RobustAsinh().fit(y_window[:, None])
        x_scaled = x_scaler.transform(x_window)
        y_scaled = y_scaler.transform(y_window[:, None]).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if alpha is None:
                # With fewer samples than features AIC needs a noise estimate; the target
                # variance is an upper bound, which errs towards sparser models.
                too_short = len(y_scaled) <= x_scaled.shape[1] + 1
                noise_variance = float(np.var(y_scaled)) if too_short else None
                selector = LassoLarsIC(criterion="aic", max_iter=2000, noise_variance=noise_variance)
                selector.fit(x_scaled, y_scaled)
                alpha = max(selector.alpha_, 1e-6)
            model = Lasso(alpha=alpha, max_iter=5000).fit(x_scaled, y_scaled)
        fitted.append((x_scaler, y_scaler, model))
    return fitted


class LEARForecaster(Forecaster):
    name = "lear"

    def __init__(self, windows: tuple[int, ...] = DEFAULT_WINDOWS, refit_every: int = 7, n_jobs: int = -1):
        self.windows = windows
        self.refit_every = refit_every
        self.min_train_days = max(windows) + 7
        self.n_jobs = n_jobs
        self.slot_models: list[list[tuple]] = []

    def fit(self, train: pd.DataFrame) -> None:
        inputs, targets = lear_design(train)
        self.input_columns = list(inputs.columns)
        x_values = inputs.to_numpy(dtype=float)
        self.slot_models = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_slot)(x_values, targets[slot].to_numpy(dtype=float), self.windows)
            for slot in range(PERIODS_PER_DAY)
        )

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        inputs, _ = lear_design(rows)
        x_row = inputs.reindex(columns=self.input_columns).to_numpy(dtype=float)
        forecasts = []
        for slot in rows["slot"].to_numpy():
            # A missing input is set to the window median (0 after scaling).
            window_forecasts = [
                y_scaler.inverse(model.predict(np.nan_to_num(x_scaler.transform(x_row)))[:, None]).item()
                for x_scaler, y_scaler, model in self.slot_models[slot]
            ]
            forecasts.append(np.mean(window_forecasts))
        return np.array(forecasts)
