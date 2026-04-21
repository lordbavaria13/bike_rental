from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_plot_style(title: str, xlabel: str, ylabel: str, title_size: int, label_size: int) -> None:
    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=label_size)
    plt.ylabel(ylabel, fontsize=label_size)
    plt.tight_layout()


def plot_actual_vs_predicted(
    pred_df: pd.DataFrame,
    target_col: str,
    output_path: Path,
    model_name: str,
    figsize=(8, 5),
    dpi: int = 150,
    title_size: int = 13,
    label_size: int = 11,
) -> None:
    test_df = pred_df[pred_df["split"] == "test"].copy()

    plt.figure(figsize=figsize)
    plt.scatter(test_df[target_col], test_df["prediction"], alpha=0.5)
    min_val = min(test_df[target_col].min(), test_df["prediction"].min())
    max_val = max(test_df[target_col].max(), test_df["prediction"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    apply_plot_style(
        f"{model_name} - Actual vs Predicted (Test)",
        f"Actual {target_col}",
        f"Predicted {target_col}",
        title_size,
        label_size,
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_residuals_histogram(
    pred_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    figsize=(8, 5),
    dpi: int = 150,
    title_size: int = 13,
    label_size: int = 11,
) -> None:
    test_df = pred_df[pred_df["split"] == "test"].copy()

    plt.figure(figsize=figsize)
    plt.hist(test_df["residual"], bins=30)
    apply_plot_style(
        f"{model_name} - Residual Histogram (Test)",
        "Residual",
        "Frequency",
        title_size,
        label_size,
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_residuals_vs_predicted(
    pred_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    figsize=(8, 5),
    dpi: int = 150,
    title_size: int = 13,
    label_size: int = 11,
) -> None:
    test_df = pred_df[pred_df["split"] == "test"].copy()

    plt.figure(figsize=figsize)
    plt.scatter(test_df["prediction"], test_df["residual"], alpha=0.5)
    plt.axhline(0, linestyle="--")
    apply_plot_style(
        f"{model_name} - Residuals vs Predicted (Test)",
        "Predicted value",
        "Residual",
        title_size,
        label_size,
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_error_over_time(
    pred_df: pd.DataFrame,
    time_col: str,
    output_path: Path,
    model_name: str,
    figsize=(8, 5),
    dpi: int = 150,
    title_size: int = 13,
    label_size: int = 11,
) -> None:
    test_df = pred_df[pred_df["split"] == "test"].copy()
    grouped = (
        test_df.groupby(time_col, as_index=False)["residual"]
        .apply(lambda s: np.mean(np.abs(s)))
        .rename(columns={"residual": "mean_absolute_error"})
    )

    plt.figure(figsize=figsize)
    plt.plot(grouped[time_col], grouped["mean_absolute_error"])
    apply_plot_style(
        f"{model_name} - Mean Absolute Error over Time (Test)",
        time_col,
        "Mean absolute error",
        title_size,
        label_size,
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_coefficients(
    feature_names: list[str],
    coefficients,
    output_path: Path,
    model_name: str,
    figsize=(10, 6),
    dpi: int = 150,
    title_size: int = 13,
    label_size: int = 11,
) -> None:
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient")

    plt.figure(figsize=figsize)
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    apply_plot_style(
        f"{model_name} - Coefficients",
        "Coefficient value",
        "Feature",
        title_size,
        label_size,
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_feature_importance(
    feature_names: list[str],
    importances,
    output_path: Path,
    model_name: str,
    figsize=(10, 6),
    dpi: int = 150,
    title_size: int = 13,
    label_size: int = 11,
) -> None:
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance")

    plt.figure(figsize=figsize)
    plt.barh(imp_df["feature"], imp_df["importance"])
    apply_plot_style(
        f"{model_name} - Feature Importance",
        "Importance",
        "Feature",
        title_size,
        label_size,
    )
    plt.savefig(output_path, dpi=dpi)
    plt.close()
