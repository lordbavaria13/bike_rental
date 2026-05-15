from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import Lasso

from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "LassoRegression"
BASE_DIR = Path(__file__).resolve().parent
ALPHA_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
MAX_ITER = 10000
TOL = 0.0001
PARAM_GRID = [{"alpha": alpha} for alpha in ALPHA_GRID]


def build_model(params: dict):
    return Lasso(alpha=float(params["alpha"]), max_iter=MAX_ITER, tol=TOL)


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="lasso.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=PARAM_GRID,
        experiment=args.experiment,
        scale_numeric=True,
        save_coefficients=True,
        extra_model_info={"max_iter": MAX_ITER, "tol": TOL, "alpha_grid": ALPHA_GRID},
    )


if __name__ == "__main__":
    main()
