from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
    plot_feature_importance,
    plot_residuals_histogram,
    plot_residuals_vs_predicted,
)
from modelling.common.preprocessing import load_dataset, prepare_feature_matrices
from modelling.common.split import chronological_split
from modelling.common.utils import ensure_dirs, save_dataframe, save_json


MODEL_NAME = "RandomForestRegressor"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_DIR = BASE_DIR / "model"

# We test a small parameter grid.
# The goal is to control tree complexity and compare a few reasonable forest sizes.
PARAM_GRID = [
    {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 5,
        "min_samples_split": 10,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "min_samples_split": 20,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "min_samples_split": 20,
        "max_features": "sqrt",
    },
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
    Save a summary of the final random forest setup.

    This file documents:
    - which transformed features were used
    - which forest parameters were selected
    - where the saved artifacts are stored
    """
    model_info = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "feature_names": feature_names,
        "n_final_features": len(feature_names),
        "station_id_encoding": "one_hot",
        "scaling_used": False,
        "best_params": best_params,
        "param_grid": PARAM_GRID,
        "n_jobs": N_JOBS,
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR),
        "model_file": str(MODEL_DIR / "random_forest.joblib"),
        "preprocessor_file": str(MODEL_DIR / "preprocessor.joblib"),
        "metrics": metrics,
    }
    save_json(model_info, MODEL_DIR / "model_info.json")


def plot_search_results(search_df: pd.DataFrame) -> None:
    """
    Plot validation RMSE for all tested forest settings.
    """
    plot_df = search_df.copy()

    plot_df["label"] = plot_df.apply(
        lambda row: (
            f"n={int(row['n_estimators'])}"
            f" | d={'None' if pd.isna(row['max_depth']) else int(row['max_depth'])}"
            f" | leaf={int(row['min_samples_leaf'])}"
        ),
        axis=1,
    )

    x_pos = list(range(len(plot_df)))

    plt.figure(figsize=(12, 5))
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
    # - numeric features are not scaled for a random forest
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
        scale_numeric=False,
    )

    print(f"Using {len(feature_names)} final features after preprocessing.")

    print("Searching best random forest parameters...")
    search_results = []

    # We train one forest for each parameter setting
    # and compare validation performance.
    for params in PARAM_GRID:
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
        model.fit(X_train_ready, y_train)

        train_pred_tmp = model.predict(X_train_ready)
        val_pred_tmp = model.predict(X_val_ready)

        train_metrics_tmp = compute_regression_metrics(y_train, train_pred_tmp, "train")
        val_metrics_tmp = compute_regression_metrics(y_val, val_pred_tmp, "validation")

        search_results.append(
            {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": params["max_depth"],
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "min_samples_split": int(params["min_samples_split"]),
                "max_features": params["max_features"],
                "train_rmse": float(train_metrics_tmp["train_rmse"]),
                "validation_rmse": float(val_metrics_tmp["validation_rmse"]),
                "train_mae": float(train_metrics_tmp["train_mae"]),
                "validation_mae": float(val_metrics_tmp["validation_mae"]),
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
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "min_samples_split": int(best_row["min_samples_split"]),
        "max_features": str(best_row["max_features"]),
    }

    print(f"Best params: {best_params}")

    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        min_samples_split=best_params["min_samples_split"],
        max_features=best_params["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    print("Training final random forest...")
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
        "best_n_estimators": int(best_params["n_estimators"]),
        "best_max_depth": best_params["max_depth"],
        "best_min_samples_leaf": int(best_params["min_samples_leaf"]),
        "best_min_samples_split": int(best_params["min_samples_split"]),
        "best_max_features": best_params["max_features"],
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
    joblib.dump(model, MODEL_DIR / "random_forest.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    # Save feature importances for later inspection.
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    save_dataframe(importance_df, RESULTS_DIR / "feature_importance.csv", index=False)

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

    # With station one-hot encoding we now have many more transformed features.
    # We increase the figure height so the importance plot stays readable.
    importance_plot_height = max(6, len(feature_names) * 0.20)

    plot_feature_importance(
        feature_names=feature_names,
        importances=model.feature_importances_,
        output_path=PLOTS_DIR / "feature_importance.png",
        model_name=MODEL_NAME,
        figsize=(12, importance_plot_height),
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    save_model_info(feature_names, metrics, best_params)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print(f"\nDone. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()