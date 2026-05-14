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
from modelling.common.preprocessing import (
    get_numeric_feature_columns,
    load_dataset,
    scale_features,
    split_X_y,
)
from modelling.common.split import chronological_split
from modelling.common.utils import ensure_dirs, save_dataframe, save_json


# Name of the model for saved files and plot titles
MODEL_NAME = "LinearRegression"

# Define all folders for this model
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

    I keep only the most important columns:
    - time index
    - station id
    - true target
    - prediction
    - residual

    Residual = actual value - predicted value
    """
    keep_cols = [TIME_COL, "start_station_id", TARGET_COL]

    def build_split_df(df_part: pd.DataFrame, preds, split_name: str) -> pd.DataFrame:
        # Build one prediction table for one split
        out = df_part[keep_cols].copy()
        out["split"] = split_name
        out["prediction"] = preds
        out["residual"] = out[TARGET_COL] - out["prediction"]
        return out

    # Combine train, validation, and test predictions
    pred_df = pd.concat(
        [
            build_split_df(train_df, train_pred, "train"),
            build_split_df(val_df, val_pred, "validation"),
            build_split_df(test_df, test_pred, "test"),
        ],
        ignore_index=True,
    )

    # Save combined prediction file
    save_dataframe(pred_df, RESULTS_DIR / "predictions.csv", index=False)
    return pred_df


def save_model_info(feature_cols: list[str], metrics: dict) -> None:
    """
    Save a small metadata file for the model run.

    This helps later to see:
    - which model was used
    - which target was predicted
    - which features were used
    - whether scaling was used
    - where model and scaler were saved
    - final metrics
    """
    model_info = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "feature_columns": feature_cols,
        "scaling_used": True,
        "results_dir": str(RESULTS_DIR),
        "plots_dir": str(PLOTS_DIR),
        "model_file": str(MODEL_DIR / "linear_regression.joblib"),
        "scaler_file": str(MODEL_DIR / "scaler.joblib"),
        "metrics": metrics,
    }
    save_json(model_info, MODEL_DIR / "model_info.json")


def main() -> None:
    # Make sure all output folders already exist
    ensure_dirs(RESULTS_DIR, PLOTS_DIR, MODEL_DIR)

    # Load the final reduced modelling dataset
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Split the dataset in chronological order
    # This is important because we want a realistic time-based evaluation
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

    # Keep only numeric features and remove the target from the feature list
    feature_cols = get_numeric_feature_columns(df, TARGET_COL)
    print(f"Using {len(feature_cols)} numeric feature columns.")

    # Build feature matrix X and target vector y for all splits
    X_train, y_train = split_X_y(train_df, feature_cols, TARGET_COL)
    X_val, y_val = split_X_y(val_df, feature_cols, TARGET_COL)
    X_test, y_test = split_X_y(test_df, feature_cols, TARGET_COL)

    # Scale features because linear models usually work better
    # when the input variables are on a similar scale
    print("Scaling features...")
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test
    )

    # Create the linear regression model
    model = LinearRegression()

    # Fit the model and measure training time
    print("Training linear regression...")
    fit_start = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    fit_time = time.perf_counter() - fit_start

    # Predict on all splits and measure prediction time
    print("Generating predictions...")
    pred_start = time.perf_counter()
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    predict_time = time.perf_counter() - pred_start

    # Store general run information
    metrics = {
        "model_name": MODEL_NAME,
        "target": TARGET_COL,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "fit_time_seconds": fit_time,
        "predict_time_seconds": predict_time,
        "intercept": float(model.intercept_),
    }

    # Add evaluation metrics for train, validation, and test
    metrics.update(compute_regression_metrics(y_train, train_pred, "train"))
    metrics.update(compute_regression_metrics(y_val, val_pred, "validation"))
    metrics.update(compute_regression_metrics(y_test, test_pred, "test"))

    # Save metrics in csv and json format
    metrics_df = pd.DataFrame([metrics])
    save_dataframe(metrics_df, RESULTS_DIR / "metrics.csv", index=False)
    save_json(metrics, RESULTS_DIR / "metrics.json")

    # Save prediction table
    pred_df = save_predictions(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
    )

    print("Saving model artifacts...")

    # Save the trained model and the scaler
    # I save both because the scaler is needed later for new predictions
    joblib.dump(model, MODEL_DIR / "linear_regression.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # Save coefficients in a separate file
    # This is useful because linear regression is easy to interpret
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "coefficient": model.coef_,
            "abs_coefficient": abs(model.coef_),
        }
    ).sort_values("abs_coefficient", ascending=False)
    save_dataframe(coef_df, RESULTS_DIR / "coefficients.csv", index=False)

    print("Creating plots...")

    # Plot actual values against predicted values
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

    # Plot histogram of residuals
    plot_residuals_histogram(
        pred_df=pred_df,
        output_path=PLOTS_DIR / "residuals_histogram.png",
        model_name=MODEL_NAME,
        figsize=FIGSIZE,
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    # Plot residuals against predicted values
    plot_residuals_vs_predicted(
        pred_df=pred_df,
        output_path=PLOTS_DIR / "residuals_vs_predicted.png",
        model_name=MODEL_NAME,
        figsize=FIGSIZE,
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    # Plot prediction error over time
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

    # Plot feature coefficients
    # This helps to see which variables have stronger positive or negative effects
    plot_coefficients(
        feature_names=feature_cols,
        coefficients=model.coef_,
        output_path=PLOTS_DIR / "coefficients.png",
        model_name=MODEL_NAME,
        figsize=(10, 6),
        dpi=DPI,
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
    )

    # Save metadata about the run
    save_model_info(feature_cols, metrics)

    # Print final metrics in the terminal
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print(f"\nDone. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()