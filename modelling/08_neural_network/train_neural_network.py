from __future__ import annotations

from pathlib import Path

from sklearn.neural_network import MLPRegressor

from modelling.common.config import RANDOM_STATE
from modelling.common.training import parse_experiment_argument, run_regression_experiment

MODEL_NAME = "NeuralNetworkRegressor"
BASE_DIR = Path(__file__).resolve().parent
SOLVER = "adam"
MAX_ITER = 400
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
N_ITER_NO_CHANGE = 20
PARAM_GRID = [
    {"hidden_layer_sizes": (64,), "activation": "relu", "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128,), "activation": "relu", "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128, 64), "activation": "relu", "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (64,), "activation": "tanh", "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128,), "activation": "tanh", "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128, 64), "activation": "tanh", "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128, 64), "activation": "relu", "alpha": 0.001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128, 64), "activation": "tanh", "alpha": 0.001, "learning_rate_init": 0.001},
]


def _hidden_layers(value):
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, tuple):
        return value
    return (int(value),)


def build_model(params: dict):
    return MLPRegressor(
        hidden_layer_sizes=_hidden_layers(params["hidden_layer_sizes"]),
        activation=str(params["activation"]),
        alpha=float(params["alpha"]),
        learning_rate_init=float(params["learning_rate_init"]),
        solver=SOLVER,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
    )


def main() -> None:
    args = parse_experiment_argument()
    run_regression_experiment(
        model_name=MODEL_NAME,
        model_filename="neural_network.joblib",
        model_base_dir=BASE_DIR,
        build_model=build_model,
        param_grid=PARAM_GRID,
        experiment=args.experiment,
        scale_numeric=True,
        plot_loss_curve=True,
        extra_model_info={
            "solver": SOLVER,
            "max_iter": MAX_ITER,
            "early_stopping": EARLY_STOPPING,
            "validation_fraction": VALIDATION_FRACTION,
            "n_iter_no_change": N_ITER_NO_CHANGE,
        },
    )


if __name__ == "__main__":
    main()
