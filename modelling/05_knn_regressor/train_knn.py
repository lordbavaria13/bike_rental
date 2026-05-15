from __future__ import annotations

from pathlib import Path

from sklearn.neighbors import KNeighborsRegressor

from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "KNNRegressor"
BASE_DIR = Path(__file__).resolve().parent
N_JOBS = -1
PARAM_GRID = [
    {"n_neighbors": 25, "weights": "uniform", "p": 1},
    {"n_neighbors": 50, "weights": "uniform", "p": 1},
    {"n_neighbors": 75, "weights": "uniform", "p": 1},
    {"n_neighbors": 100, "weights": "uniform", "p": 1},
    {"n_neighbors": 150, "weights": "uniform", "p": 1},
    {"n_neighbors": 200, "weights": "uniform", "p": 1},
    {"n_neighbors": 25, "weights": "distance", "p": 1},
    {"n_neighbors": 50, "weights": "distance", "p": 1},
    {"n_neighbors": 75, "weights": "distance", "p": 1},
    {"n_neighbors": 100, "weights": "distance", "p": 1},
    {"n_neighbors": 150, "weights": "distance", "p": 1},
    {"n_neighbors": 200, "weights": "distance", "p": 1},
    {"n_neighbors": 25, "weights": "uniform", "p": 2},
    {"n_neighbors": 50, "weights": "uniform", "p": 2},
    {"n_neighbors": 75, "weights": "uniform", "p": 2},
    {"n_neighbors": 100, "weights": "uniform", "p": 2},
    {"n_neighbors": 150, "weights": "uniform", "p": 2},
    {"n_neighbors": 200, "weights": "uniform", "p": 2},
    {"n_neighbors": 25, "weights": "distance", "p": 2},
    {"n_neighbors": 50, "weights": "distance", "p": 2},
    {"n_neighbors": 75, "weights": "distance", "p": 2},
    {"n_neighbors": 100, "weights": "distance", "p": 2},
    {"n_neighbors": 150, "weights": "distance", "p": 2},
    {"n_neighbors": 200, "weights": "distance", "p": 2},
]


def build_model(params: dict):
    return KNeighborsRegressor(
        n_neighbors=int(params["n_neighbors"]),
        weights=str(params["weights"]),
        p=int(params["p"]),
        n_jobs=N_JOBS,
    )


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="knn.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=PARAM_GRID,
        experiment=args.experiment,
        scale_numeric=True,
        extra_model_info={"n_jobs": N_JOBS},
    )


if __name__ == "__main__":
    main()
