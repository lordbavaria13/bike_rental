from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from modelling.common.config import RANDOM_STATE
from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "RandomForestRegressor"
BASE_DIR = Path(__file__).resolve().parent
N_JOBS = -1
PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5, "min_samples_split": 10, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5, "min_samples_split": 10, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5, "min_samples_split": 10, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5, "min_samples_split": 10, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "min_samples_split": 10, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 5, "min_samples_split": 10, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 10, "min_samples_split": 20, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 10, "min_samples_split": 20, "max_features": "sqrt"},
]


def build_model(params: dict):
    max_depth = None if params["max_depth"] is None else int(params["max_depth"])
    return RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=max_depth,
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        max_features=str(params["max_features"]),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="random_forest.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=PARAM_GRID,
        experiment=args.experiment,
        scale_numeric=False,
        save_feature_importance=True,
        extra_model_info={"n_jobs": N_JOBS},
    )


if __name__ == "__main__":
    main()
