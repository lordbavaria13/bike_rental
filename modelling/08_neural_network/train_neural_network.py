from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor

from modelling.common.config import (
    DATA_PATH,
    TARGET_COL,
    TIME_COL,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    FIGSIZE,
    DPI,
    TITLE_SIZE,
    LABEL_SIZE,
    RANDOM_STATE,
)
from modelling.common.metrics import compute_regression_metrics
from modelling.common.plotting import (
    plot_actual_vs_predicted,
    plot_error_over_time,
    plot_residuals_histogram,
    plot_residuals_vs_predicted,
)
from modelling.common.preprocessing import load_dataset, prepare_feature_matrices
from modelling.common.split import chronological_split
from modelling.common.utils import ensure_dirs, save_dataframe, save_json


MODEL_NAME = "NeuralNetworkRegressor"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_DIR = BASE_DIR / "model"

# We test a small set of reasonable neural network settings.
# hidden_layer_sizes controls the network architecture.
# activation controls the non-linear transformation.
# alpha is the L2 regularization strength.
# learning_rate_init is the starting learning rate.
PARAM_GRID = [
    {
        "hidden_layer_sizes": (64,),
        "activation": "relu",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (128,),
        "activation": "relu",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (64,),
        "activation": "tanh",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (128,),
        "activation": "tanh",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "alpha": 0.001,
        "learning_rate_init": 0.001,
    },
    {
        "hidden_layer_sizes": (128, 64),
        "activation": "tanh",
        "alpha": 0.001,
        "learning_rate_init": 0.001,
    },
]

SOLVER = "adam"
MAX_ITER = 400
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
N_ITER_NO_CHANGE = 20


def save_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred,
    val_pred,
    test_pred,
) -> pd.DataFrame:
    """
    Save predictions for train, validation, and test in one file.

    We keep station and time information so we can inspect
    where the model performs well or badly later.
    """
    keep_cols = [TIME_COL, "start_station_id", TARGET_COL]

    def build_split_df(df_part: pd.DataFrame, preds, split_name: str) -> pd.DataFrame:
        out = df_part[keep_cols].copy()
        out["split"] = split_name
        out["prediction"] = preds
        out["residual"] = out[TARGET_COL] - out["prediction"]
        return out

    pred_df = pd.concat(
        [
            build_split_df(train_df, train_pred, "train"),
            build_split_df(val_df, val_pred, "validation"),
            build_split_df(test_df, test_pred, "test"),
        ],
        ignore_index=True,
    )

    save_dataframe(pred_df, RESULTS_DIR / "predictions.csv", index=False)
    return pred_df


def save_model_info(feature_names: list[str], metrics: dict, best_params: dict) -> None:
    """
    Save a summary of the final neural network setup.
    """
    model_info = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "feature_names": feature_names,
        "n_final_features": len(feature_names),
        "station_id_encoding": "one_hot",
        "scaling_used": True,
        "best_params": {
            "hidden_layer_sizes": list(best_params["hidden_layer_sizes"]),
            "activation": best_params["activation"],
            "alpha": float(best_params["alpha"]),
            "learning_rate_init": float(best_params["learning_rate_init"]),
        },
        "param_grid": [
            {
                "hidden_layer_sizes": list(params["hidden_layer_sizes"]),
                "activation": params["activation"],
                "alpha": float(params["alpha"]),
                "learning_rate_init": float(params["learning_rate_init"]),
            }
            for params in PARAM_GRID
        ],
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "early_stopping": EARLY_STOPPING,
        "validation_fraction": VALIDATION_FRACTION,
        "n_iter_no_change": N_ITER_NO_CHANGE,
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR),
        "model_file": str(MODEL_DIR / "neural_network.joblib"),
        "preprocessor_file": str(MODEL_DIR / "preprocessor.joblib"),
        "metrics": metrics,
    }
    save_json(model_info, MODEL_DIR / "model_info.json")


def plot_search_results(search_df: pd.DataFrame) -> None:
    """
    Plot validation RMSE for all tested parameter settings.
    """
    plot_df = search_df.copy()

    plot_df["label"] = plot_df.apply(
        lambda row: (
            f"{row['hidden_layer_sizes']}"
            f" | {row['activation']}"
            f" | a={row['alpha']}"
            f" | lr={row['learning_rate_init']}"
        ),
        axis=1,
    )

    x_pos = list(range(len(plot_df)))

    plt.figure(figsize=(13, 5))
    plt.plot(x_pos, plot_df["validation_rmse"].to_numpy(), marker="o")
    plt.xticks(x_pos, plot_df["label"].tolist(), rotation=45, ha="right")
    plt.title(
        f"{MODEL_NAME} - Validation RMSE by Parameter Setting",
        fontsize=TITLE_SIZE,
    )
    plt.xlabel("parameter setting", fontsize=LABEL_SIZE)
    plt.ylabel("Validation RMSE", fontsize=LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "validation_curve.png", dpi=DPI)
    plt.close()


def plot_loss_curve(loss_values: list[float], output_path: Path) -> None:
    """
    Plot the training loss over epochs for the final neural network.
    """
    if not loss_values:
        return

    x_pos = list(range(1, len(loss_values) + 1))

    plt.figure(figsize=FIGSIZE)
    plt.plot(x_pos, loss_values, marker="o", markersize=2)
    plt.title(f"{MODEL_NAME} - Training Loss Curve", fontsize=TITLE_SIZE)
    plt.xlabel("Iteration", fontsize=LABEL_SIZE)
    plt.ylabel("Loss", fontsize=LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def build_model(params: dict) -> MLPRegressor:
    """
    Build one neural network with the chosen parameter setting.
    """
    return MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        solver=SOLVER,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    # Create output folders before the script starts.
    ensure_dirs(RESULTS_DIR, PLOTS_DIR, MODEL_DIR)

    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    print("Creating chronological split...")
    train_df, val_df, test_df = chronological_split(
        df=df,
        time_col=TIME_COL,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("Preparing feature matrices...")
    # We use the shared preprocessing step for all models.
    # Important:
    # - start_station_id is treated as a categorical feature
    # - it is one-hot encoded after the chronological split
    # - numeric features are scaled because neural networks are scale-sensitive
    (
        preprocessor,
        feature_names,
        X_train_ready,
        X_val_ready,
        X_test_ready,
        y_train,
        y_val,
        y_test,
    ) = prepare_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=TARGET_COL,
        categorical_cols=["start_station_id"],
        scale_numeric=True,
    )

    print(f"Using {len(feature_names)} final features after preprocessing.")

    print("Searching best neural network parameters...")
    search_results = []

    # We train one neural network for each parameter setting
    # and compare validation performance.
    for params in PARAM_GRID:
        model = build_model(params)
        model.fit(X_train_ready, y_train)

        train_pred_tmp = model.predict(X_train_ready)
        val_pred_tmp = model.predict(X_val_ready)

        train_metrics_tmp = compute_regression_metrics(y_train, train_pred_tmp, "train")
        val_metrics_tmp = compute_regression_metrics(y_val, val_pred_tmp, "validation")

        search_results.append(
            {
                "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
                "activation": params["activation"],
                "alpha": float(params["alpha"]),
                "learning_rate_init": float(params["learning_rate_init"]),
                "train_rmse": float(train_metrics_tmp["train_rmse"]),
                "validation_rmse": float(val_metrics_tmp["validation_rmse"]),
                "train_mae": float(train_metrics_tmp["train_mae"]),
                "validation_mae": float(val_metrics_tmp["validation_mae"]),
                "iterations_used": int(model.n_iter_),
                "final_training_loss": float(model.loss_),
            }
        )

    search_df = pd.DataFrame(search_results).sort_values(
        ["validation_rmse", "train_rmse"]
    )
    save_dataframe(search_df, RESULTS_DIR / "hyperparameter_search.csv", index=False)
    plot_search_results(search_df)

    # We choose the setting with the lowest validation RMSE.
    best_row = search_df.iloc[0]
    best_params = {
        "hidden_layer_sizes": eval(best_row["hidden_layer_sizes"]),
        "activation": str(best_row["activation"]),
        "alpha": float(best_row["alpha"]),
        "learning_rate_init": float(best_row["learning_rate_init"]),
    }

    print(f"Best params: {best_params}")

    model = build_model(best_params)

    print("Training final neural network...")
    fit_start = time.perf_counter()
    model.fit(X_train_ready, y_train)
    fit_time = time.perf_counter() - fit_start

    print("Generating predictions...")
    pred_start = time.perf_counter()
    train_pred = model.predict(X_train_ready)
    val_pred = model.predict(X_val_ready)
    test_pred = model.predict(X_test_ready)
    predict_time = time.perf_counter() - pred_start

    # Convert values to normal Python types so JSON export stays safe.
    metrics = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "best_hidden_layer_sizes": str(best_params["hidden_layer_sizes"]),
        "best_activation": best_params["activation"],
        "best_alpha": float(best_params["alpha"]),
        "best_learning_rate_init": float(best_params["learning_rate_init"]),
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "early_stopping": EARLY_STOPPING,
        "iterations_used": int(model.n_iter_),
        "final_training_loss": float(model.loss_),
        "n_features": int(len(feature_names)),
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
    }

    metrics.update(compute_regression_metrics(y_train, train_pred, "train"))
    metrics.update(compute_regression_metrics(y_val, val_pred, "validation"))
    metrics.update(compute_regression_metrics(y_test, test_pred, "test"))

    metrics_df = pd.DataFrame([metrics])
    save_dataframe(metrics_df, RESULTS_DIR / "metrics.csv", index=False)
    save_json(metrics, RESULTS_DIR / "metrics.json")

    pred_df = save_predictions(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
    )

    print("Saving model artifacts...")
    # We save both the fitted model and the fitted preprocessor.
    # This makes the full training pipeline reproducible later.
    joblib.dump(model, MODEL_DIR / "neural_network.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    print("Creating plots...")
    plot_actual_vs_predicted(
        pred_df=pred_df,
        target_col=TARGET_COL,
        output_path=PLOTS_DIR / "actual_vs_predicted.png",
        model_name=MODEL_NAME,
        figsize=FIGSIZE,
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    plot_residuals_histogram(
        pred_df=pred_df,
        output_path=PLOTS_DIR / "residuals_histogram.png",
        model_name=MODEL_NAME,
        figsize=FIGSIZE,
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    plot_residuals_vs_predicted(
        pred_df=pred_df,
        output_path=PLOTS_DIR / "residuals_vs_predicted.png",
        model_name=MODEL_NAME,
        figsize=FIGSIZE,
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    plot_error_over_time(
        pred_df=pred_df,
        time_col=TIME_COL,
        output_path=PLOTS_DIR / "error_over_time.png",
        model_name=MODEL_NAME,
        figsize=FIGSIZE,
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    plot_loss_curve(
        loss_values=model.loss_curve_,
        output_path=PLOTS_DIR / "training_loss_curve.png",
    )

    save_model_info(feature_names, metrics, best_params)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nDone. Results saved to: {RESULTS_DIR}")
    print(f"Final training iterations: {model.n_iter_}")


if __name__ == "__main__":
    main()