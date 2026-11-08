"""Forecasting models. :data:`MODELS` maps the names used by the scripts to constructors."""

from energy_price_predictor.models.base import Forecaster
from energy_price_predictor.models.baselines import NaiveForecaster, SeasonalNaiveForecaster
from energy_price_predictor.models.gbm import GBMForecaster, TunedGBMForecaster
from energy_price_predictor.models.lear import LEARForecaster


class DailyLEARForecaster(LEARForecaster):
    """LEAR recalibrated every day, as in Lago et al. (~3 h for a two-year backtest)."""

    name = "lear_daily"

    def __init__(self):
        super().__init__(refit_every=1)


MODELS = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "lear": LEARForecaster,  # weekly recalibration (~25 min for a two-year backtest)
    "lear_daily": DailyLEARForecaster,
    "gbm": GBMForecaster,  # weekly refit (~1 h for a two-year backtest)
    "gbm_tuned": TunedGBMForecaster,  # needs results/gbm_best_params.json from scripts/tune_gbm.py
}

__all__ = [
    "MODELS",
    "Forecaster",
    "GBMForecaster",
    "LEARForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "TunedGBMForecaster",
]
