from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor

from modelling.common.config import RANDOM_STATE
from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "GradientBoostingRegressor"
BASE_DIR = Path(__file__).resolve().parent
PARAM_GRID = [
    {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 5, "subsample": 1.0},
    {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 5, "subsample": 1.0},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 5, "subsample": 1.0},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 5, "subsample": 1.0},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 5, "subsample": 1.0},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 5, "subsample": 1.0},
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8},
]


def build_model(params: dict):
    return GradientBoostingRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        subsample=float(params["subsample"]),
        random_state=RANDOM_STATE,
    )


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="gradient_boosting.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=PARAM_GRID,
        experiment=args.experiment,
        scale_numeric=False,
        save_feature_importance=True,
    )


if __name__ == "__main__":
    main()
