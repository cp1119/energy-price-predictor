"""Tune LightGBM hyperparameters on the validation year with Optuna.

    uv run python scripts/tune_gbm.py --trials 30

Each trial fits once on target days before 2023-11-01 and scores MAE on 2023-11-01 to
2024-10-31 without refitting. The test period (from 2024-11-01) is never touched. The best
parameters go to results/gbm_best_params.json, which the "gbm_tuned" model reads.
"""

import argparse
import json

import numpy as np
import optuna
import pandas as pd

from energy_price_predictor.data import HISTORY_START, load_panel
from energy_price_predictor.features import build_feature_table
from energy_price_predictor.models.gbm import TUNED_PARAMS_PATH, GBMForecaster

VALIDATION_START = pd.Timestamp("2023-11-01")
VALIDATION_END = pd.Timestamp("2024-10-31")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    panel = load_panel(HISTORY_START, (VALIDATION_END + pd.Timedelta(days=2)).tz_localize("UTC"))
    table = build_feature_table(panel)
    days = table.index.get_level_values("day")
    train = table[days < VALIDATION_START]
    validation = table[(days >= VALIDATION_START) & (days <= VALIDATION_END)].dropna(subset=["target"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
        half_life = trial.suggest_categorical("half_life_days", [90, 180, 365, 730, 0])
        relative = trial.suggest_categorical("relative_to_level", [True, False])
        model = GBMForecaster(params, relative_to_level=relative, half_life_days=half_life or None)
        model.fit(train)
        return float(np.abs(model.predict(validation) - validation["target"]).mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    default_params = {"learning_rate": 0.05, "n_estimators": 600, "num_leaves": 63, "min_child_samples": 100,
                      "colsample_bytree": 0.7, "reg_lambda": 1.0, "half_life_days": 365, "relative_to_level": True}
    study.enqueue_trial(default_params)
    study.optimize(objective, n_trials=args.trials, callbacks=[
        lambda study, trial: print(f"trial {trial.number:3d}  MAE {trial.value:6.3f}  best {study.best_value:6.3f}", flush=True)
    ])

    print(f"\nDefault-parameter validation MAE: {study.trials[0].value:.3f}")
    print(f"Best validation MAE: {study.best_value:.3f}")
    print(json.dumps(study.best_params, indent=2))
    TUNED_PARAMS_PATH.parent.mkdir(exist_ok=True)
    TUNED_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))
    print(f"Saved {TUNED_PARAMS_PATH}")


if __name__ == "__main__":
    main()
