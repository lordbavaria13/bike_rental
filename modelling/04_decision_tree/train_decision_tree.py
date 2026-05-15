from __future__ import annotations

from pathlib import Path

from sklearn.tree import DecisionTreeRegressor

from modelling.common.config import RANDOM_STATE
from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "DecisionTreeRegressor"
BASE_DIR = Path(__file__).resolve().parent
PARAM_GRID = [
    {"max_depth": 3, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": 5, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": 8, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": 10, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": 12, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": None, "min_samples_leaf": 5, "min_samples_split": 10},
    {"max_depth": 5, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 8, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 10, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": None, "min_samples_leaf": 10, "min_samples_split": 20},
]


def build_model(params: dict):
    max_depth = None if params["max_depth"] is None else int(params["max_depth"])
    return DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        random_state=RANDOM_STATE,
    )


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="decision_tree.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=PARAM_GRID,
        experiment=args.experiment,
        scale_numeric=False,
        save_feature_importance=True,
    )


if __name__ == "__main__":
    main()
