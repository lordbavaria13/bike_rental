from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LinearRegression

from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "LinearRegression"
BASE_DIR = Path(__file__).resolve().parent


def build_model(params: dict):
    return LinearRegression()


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="linear_regression.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=[{}],
        experiment=args.experiment,
        scale_numeric=True,
        save_coefficients=True,
    )


if __name__ == "__main__":
    main()
