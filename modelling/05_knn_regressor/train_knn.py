from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

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


MODEL_NAME = "KNNRegressor"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_DIR = BASE_DIR / "model"

PARAM_GRID = [
    {"n_neighbors": 3, "weights": "uniform", "p": 2},
    {"n_neighbors": 5, "weights": "uniform", "p": 2},
    {"n_neighbors": 7, "weights": "uniform", "p": 2},
    {"n_neighbors": 9, "weights": "uniform", "p": 2},
    {"n_neighbors": 15, "weights": "uniform", "p": 2},
    {"n_neighbors": 25, "weights": "uniform", "p": 2},
    {"n_neighbors": 3, "weights": "distance", "p": 2},
    {"n_neighbors": 5, "weights": "distance", "p": 2},
    {"n_neighbors": 7, "weights": "distance", "p": 2},
    {"n_neighbors": 9, "weights": "distance", "p": 2},
    {"n_neighbors": 15, "weights": "distance", "p": 2},
    {"n_neighbors": 25, "weights": "distance", "p": 2},
]

N_JOBS = -1


def save_predictions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred,
    val_pred,
    test_pred,
) -> pd.DataFrame:
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


def save_model_info(feature_cols: list[str], metrics: dict, best_params: dict) -> None:
    model_info = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "feature_columns": feature_cols,
        "scaling_used": True,
        "best_params": best_params,
        "param_grid": PARAM_GRID,
        "n_jobs": N_JOBS,
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR),
        "model_file": str(MODEL_DIR / "knn.joblib"),
        "scaler_file": str(MODEL_DIR / "scaler.joblib"),
        "metrics": metrics,
    }
    save_json(model_info, MODEL_DIR / "model_info.json")


def plot_search_results(search_df: pd.DataFrame) -> None:
    plot_df = search_df.copy()

    plot_df["label"] = plot_df.apply(
        lambda row: f"k={int(row['n_neighbors'])} | {row['weights']} | p={int(row['p'])}",
        axis=1,
    )

    x_pos = list(range(len(plot_df)))

    plt.figure(figsize=(12, 5))
    plt.plot(x_pos, plot_df["validation_rmse"].to_numpy(), marker="o")
    plt.xticks(x_pos, plot_df["label"].tolist(), rotation=45, ha="right")
    plt.title(f"{MODEL_NAME} - Validation RMSE by Parameter Setting", fontsize=TITLE_SIZE)
    plt.xlabel("parameter setting", fontsize=LABEL_SIZE)
    plt.ylabel("Validation RMSE", fontsize=LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "validation_curve.png", dpi=DPI)
    plt.close()


def main() -> None:
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

    feature_cols = get_numeric_feature_columns(df, TARGET_COL)
    print(f"Using {len(feature_cols)} numeric feature columns.")

    X_train, y_train = split_X_y(train_df, feature_cols, TARGET_COL)
    X_val, y_val = split_X_y(val_df, feature_cols, TARGET_COL)
    X_test, y_test = split_X_y(test_df, feature_cols, TARGET_COL)

    print("Scaling features...")
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test
    )

    print("Searching best KNN parameters...")
    search_results = []

    for params in PARAM_GRID:
        model = KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            p=params["p"],
            n_jobs=N_JOBS,
        )
        model.fit(X_train_scaled, y_train)

        train_pred_tmp = model.predict(X_train_scaled)
        val_pred_tmp = model.predict(X_val_scaled)

        train_metrics_tmp = compute_regression_metrics(y_train, train_pred_tmp, "train")
        val_metrics_tmp = compute_regression_metrics(y_val, val_pred_tmp, "validation")

        search_results.append(
            {
                "n_neighbors": int(params["n_neighbors"]),
                "weights": params["weights"],
                "p": int(params["p"]),
                "train_rmse": train_metrics_tmp["train_rmse"],
                "validation_rmse": val_metrics_tmp["validation_rmse"],
                "train_mae": train_metrics_tmp["train_mae"],
                "validation_mae": val_metrics_tmp["validation_mae"],
            }
        )

    search_df = pd.DataFrame(search_results).sort_values(
        ["validation_rmse", "train_rmse"]
    )
    save_dataframe(search_df, RESULTS_DIR / "hyperparameter_search.csv", index=False)
    plot_search_results(search_df)

    best_row = search_df.iloc[0]
    best_params = {
        "n_neighbors": int(best_row["n_neighbors"]),
        "weights": str(best_row["weights"]),
        "p": int(best_row["p"]),
    }

    print(f"Best params: {best_params}")

    model = KNeighborsRegressor(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        p=best_params["p"],
        n_jobs=N_JOBS,
    )

    print("Training final KNN model...")
    fit_start = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    fit_time = time.perf_counter() - fit_start

    print("Generating predictions...")
    pred_start = time.perf_counter()
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    predict_time = time.perf_counter() - pred_start

    metrics = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "best_n_neighbors": best_params["n_neighbors"],
        "best_weights": best_params["weights"],
        "best_p": best_params["p"],
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
    joblib.dump(model, MODEL_DIR / "knn.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

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

    save_model_info(feature_cols, metrics, best_params)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print(f"\nDone. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()