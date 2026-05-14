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
from modelling.common.preprocessing import (
    get_numeric_feature_columns,
    load_dataset,
    scale_features,
    split_X_y,
)
from modelling.common.split import chronological_split
from modelling.common.utils import ensure_dirs, save_dataframe, save_json


# Name of the model that will appear in saved results.
MODEL_NAME = "NeuralNetworkRegressor"

# Paths for the current model folder and its outputs.
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_DIR = BASE_DIR / "model"

# We only test a small grid of neural network settings.
# This keeps the search simple and makes the setup comparable to the other models.
PARAM_GRID = [
    {"hidden_layer_sizes": (32, 16), "activation": "tanh", "alpha": 0.0001, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (32, 16), "activation": "tanh", "alpha": 0.001, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (64, 32), "activation": "tanh", "alpha": 0.0001, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (64, 32), "activation": "tanh", "alpha": 0.001, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (64, 32), "activation": "tanh", "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (64, 32), "activation": "tanh", "alpha": 0.001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 0.0001, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 0.001, "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 0.0001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 0.001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128, 64), "activation": "tanh", "alpha": 0.001, "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128, 64), "activation": "relu", "alpha": 0.001, "learning_rate_init": 0.001},
]

# Fixed training settings for all tested neural networks.
SOLVER = "adam"
MAX_ITER = 500
EARLY_STOPPING = False


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

    We also save residuals because they are needed for later analysis and plots.
    """
    keep_cols = [TIME_COL, "start_station_id", TARGET_COL]

    def build_split_df(df_part: pd.DataFrame, preds, split_name: str) -> pd.DataFrame:
        # Keep only the most important columns and add prediction info.
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


def save_model_info(feature_cols: list[str], metrics: dict, best_params: dict) -> None:
    """
    Save a summary file with the final model setup.

    This makes it easier to understand later which settings were used.
    """
    model_info = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "feature_columns": feature_cols,
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
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR),
        "model_file": str(MODEL_DIR / "neural_network.joblib"),
        "scaler_file": str(MODEL_DIR / "scaler.joblib"),
        "metrics": metrics,
    }
    save_json(model_info, MODEL_DIR / "model_info.json")


def plot_search_results(search_df: pd.DataFrame) -> None:
    """
    Plot validation RMSE for all tested parameter settings.

    This plot helps us see which neural network setup worked best on validation data.
    """
    plot_df = search_df.copy()

    plot_df["label"] = plot_df.apply(
        lambda row: (
            f"layers={row['hidden_layer_sizes']} | "
            f"act={row['activation']} | "
            f"alpha={row['alpha']} | "
            f"lr={row['learning_rate_init']}"
        ),
        axis=1,
    )

    x_pos = list(range(len(plot_df)))

    plt.figure(figsize=(14, 5))
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


def plot_loss_curves(loss_curves: list[dict], top_n: int = 3) -> None:
    """
    Plot the training loss curves of the best tested parameter settings.

    We only plot the top few models to keep the figure readable.
    """
    top_curves = sorted(loss_curves, key=lambda x: x["validation_rmse"])[:top_n]

    plt.figure(figsize=(10, 6))
    for curve_info in top_curves:
        plt.plot(curve_info["loss_curve"], label=curve_info["label"])

    plt.title(f"{MODEL_NAME} - Training Loss Curves", fontsize=TITLE_SIZE)
    plt.xlabel("iteration", fontsize=LABEL_SIZE)
    plt.ylabel("training loss", fontsize=LABEL_SIZE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_loss_curves.png", dpi=DPI)
    plt.close()


def plot_final_training_loss(loss_curve: list[float]) -> None:
    """
    Plot the final model's training loss over iterations.

    This gives a quick view of whether the optimization became more stable over time.
    """
    plt.figure(figsize=FIGSIZE)
    plt.plot(loss_curve, marker="o", markersize=2)
    plt.title(f"{MODEL_NAME} - Final Training Loss", fontsize=TITLE_SIZE)
    plt.xlabel("iteration", fontsize=LABEL_SIZE)
    plt.ylabel("training loss", fontsize=LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "final_training_loss.png", dpi=DPI)
    plt.close()


def build_model(params: dict) -> MLPRegressor:
    """
    Create one neural network model from a parameter dictionary.

    We use the same base settings for every run so that the comparison stays fair.
    """
    return MLPRegressor(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        activation=params["activation"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        solver=SOLVER,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOPPING,
        random_state=RANDOM_STATE,
    )


def main() -> None:
    # Make sure all output folders exist before we start.
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

    # Use exactly the same numeric feature selection as in the other models.
    # This keeps the experiment comparable.
    feature_cols = get_numeric_feature_columns(df, TARGET_COL)
    print(f"Using {len(feature_cols)} numeric feature columns.")

    # Split each dataset into X and y.
    X_train, y_train = split_X_y(train_df, feature_cols, TARGET_COL)
    X_val, y_val = split_X_y(val_df, feature_cols, TARGET_COL)
    X_test, y_test = split_X_y(test_df, feature_cols, TARGET_COL)

    print("Scaling features...")
    # Neural networks need scaled input values.
    # We fit the scaler only on the training set and apply it to validation and test.
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test
    )

    print("Searching best neural network parameters...")
    search_results = []
    loss_curves = []

    # Train one model for each parameter setting and evaluate it on validation data.
    for params in PARAM_GRID:
        model = build_model(params)
        model.fit(X_train_scaled, y_train)

        train_pred_tmp = model.predict(X_train_scaled)
        val_pred_tmp = model.predict(X_val_scaled)

        train_metrics_tmp = compute_regression_metrics(y_train, train_pred_tmp, "train")
        val_metrics_tmp = compute_regression_metrics(y_val, val_pred_tmp, "validation")

        label = (
            f"layers={params['hidden_layer_sizes']} | "
            f"act={params['activation']} | "
            f"alpha={params['alpha']} | "
            f"lr={params['learning_rate_init']}"
        )

        # Save the most important results from this parameter run.
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
                "final_training_loss": float(model.loss_curve_[-1]),
            }
        )

        # Save the training loss curve so we can compare learning behaviour.
        loss_curves.append(
            {
                "label": label,
                "validation_rmse": float(val_metrics_tmp["validation_rmse"]),
                "loss_curve": model.loss_curve_,
            }
        )

    # Sort models by validation RMSE.
    # The model with the lowest validation RMSE is selected as the final model.
    search_df = pd.DataFrame(search_results).sort_values(
        ["validation_rmse", "train_rmse"]
    )
    save_dataframe(search_df, RESULTS_DIR / "hyperparameter_search.csv", index=False)

    plot_search_results(search_df)
    plot_loss_curves(loss_curves, top_n=3)

    best_row = search_df.iloc[0]
    best_params = {
        "hidden_layer_sizes": eval(best_row["hidden_layer_sizes"]),
        "activation": str(best_row["activation"]),
        "alpha": float(best_row["alpha"]),
        "learning_rate_init": float(best_row["learning_rate_init"]),
    }

    print(f"Best params: {best_params}")

    # Train the final model again with the best settings.
    model = build_model(best_params)

    print("Training final neural network...")
    fit_start = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    fit_time = time.perf_counter() - fit_start

    print("Generating predictions...")
    pred_start = time.perf_counter()
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    predict_time = time.perf_counter() - pred_start

    # Save metadata and performance numbers.
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
        "final_training_loss": float(model.loss_curve_[-1]),
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "fit_time_seconds": fit_time,
        "predict_time_seconds": predict_time,
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
    # Save the trained model and the scaler so the predictions can be reproduced later.
    joblib.dump(model, MODEL_DIR / "neural_network.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    print("Creating plots...")
    # Create the same main diagnostic plots that we used for the other models.
    plot_final_training_loss(model.loss_curve_)

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

    save_model_info(feature_cols, metrics, best_params)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nDone. Results saved to: {RESULTS_DIR}")
    print(f"Final training iterations: {model.n_iter_}")


if __name__ == "__main__":
    main()