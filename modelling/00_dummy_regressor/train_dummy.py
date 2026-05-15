from __future__ import annotations

from pathlib import Path

from sklearn.dummy import DummyRegressor

from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "DummyRegressor"
DUMMY_STRATEGY = "mean"
BASE_DIR = Path(__file__).resolve().parent


def build_model(params: dict):
    return DummyRegressor(strategy=DUMMY_STRATEGY)


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="dummy_regressor.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=[{}],
        experiment=args.experiment,
        scale_numeric=False,
        extra_model_info={"strategy": DUMMY_STRATEGY},
    )


if __name__ == "__main__":
    main()
