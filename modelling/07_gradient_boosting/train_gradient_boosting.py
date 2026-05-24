from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor

from modelling.common.config import RANDOM_STATE
from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "GradientBoostingRegressor"
BASE_DIR = Path(__file__).resolve().parent

# param fine tuning
#'''
PARAM_GRID = [
    
    # Block 1: Fix values (learning_rate=0.03, depth=3), variiere n_estimators
    {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8},
    {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8},
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8}, # Bisheriger Gewinner
    {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8},
    
    # Block 2: more complexity
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "min_samples_leaf": 10, "subsample": 0.8},
    {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4, "min_samples_leaf": 10, "subsample": 0.8},
    
    # Block 3: slow learning against overfitting
    {"n_estimators": 600, "learning_rate": 0.01, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.8},

]
    #'''
#PARAM_GRID = [{"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,"min_samples_leaf": 1, "subsample": 1.0}]

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