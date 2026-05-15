from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import Ridge

from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "RidgeRegression"
BASE_DIR = Path(__file__).resolve().parent
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
SOLVER = "auto"
PARAM_GRID = [{"alpha": alpha} for alpha in ALPHA_GRID]


def build_model(params: dict):
    return Ridge(alpha=float(params["alpha"]), solver=SOLVER)


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="ridge.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=PARAM_GRID,
        experiment=args.experiment,
        scale_numeric=True,
        save_coefficients=True,
        extra_model_info={"solver": SOLVER, "alpha_grid": ALPHA_GRID},
    )


if __name__ == "__main__":
    main()
