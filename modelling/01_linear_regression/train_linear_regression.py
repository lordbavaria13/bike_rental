from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

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
    plot_coefficients,
    plot_error_over_time,
    plot_residuals_histogram,
    plot_residuals_vs_predicted,
)
from modelling.common.preprocessing import load_dataset, prepare_feature_matrices
from modelling.common.split import chronological_split
from modelling.common.utils import ensure_dirs, save_dataframe, save_json


MODEL_NAME = "LinearRegression"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODEL_DIR = BASE_DIR / "model"


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

    We keep the station and time information so we can later inspect
    where the model performs well or badly.
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


def save_model_info(feature_names: list[str], metrics: dict) -> None:
    """
    Save a summary of the final model setup.

    This file documents which transformed features were used
    after one-hot encoding and scaling.
    """
    model_info = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "feature_names": feature_names,
        "n_final_features": len(feature_names),
        "station_id_encoding": "one_hot",
        "scaling_used": True,
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR),
        "model_file": str(MODEL_DIR / "linear_regression.joblib"),
        "preprocessor_file": str(MODEL_DIR / "preprocessor.joblib"),
        "metrics": metrics,
    }
    save_json(model_info, MODEL_DIR / "model_info.json")


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
    # - numeric features are scaled because linear regression benefits from that
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

    model = LinearRegression()

    print("Training linear regression...")
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
        "n_features": int(len(feature_names)),
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
        "intercept": float(model.intercept_),
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
    # We save both the model and the fitted preprocessor.
    # This makes the full pipeline reproducible later.
    joblib.dump(model, MODEL_DIR / "linear_regression.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    # Save coefficients for later inspection.
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_,
            "abs_coefficient": abs(model.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)
    save_dataframe(coef_df, RESULTS_DIR / "coefficients.csv", index=False)

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

    # The coefficient plot can become large now because we also have
    # one-hot encoded station features. We increase the figure height
    # so the plot stays readable.
    coefficient_plot_height = max(6, len(feature_names) * 0.20)

    plot_coefficients(
        feature_names=feature_names,
        coefficients=model.coef_,
        output_path=PLOTS_DIR / "coefficients.png",
        model_name=MODEL_NAME,
        figsize=(12, coefficient_plot_height),
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    save_model_info(feature_names, metrics)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print(f"\nDone. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()