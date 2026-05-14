from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
MODELLING_DIR = BASE_DIR / "modelling"

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Plot settings
FIGSIZE = (10, 5)
DPI = 150
TITLE_SIZE = 13
LABEL_SIZE = 11

# Metrics that we want to compare across models
METRIC_SPECS = [
    {"name": "mae", "title": "MAE Comparison"},
    {"name": "rmse", "title": "RMSE Comparison"},
    {"name": "median_ae", "title": "Median AE Comparison"},
    {"name": "mape", "title": "MAPE Comparison"},
    {"name": "r2", "title": "R² Comparison"},
    {"name": "explained_variance", "title": "Explained Variance Comparison"},
]

# Fixed model order for all overview plots
# This keeps the plots chronological and consistent.
MODEL_ORDER = [
    "00_dummy_regressor",
    "01_linear_regression",
    "02_ridge_regression",
    "03_lasso_regression",
    "04_decision_tree",
    "05_knn_regressor",
    "06_random_forest",
    "07_gradient_boosting",
    "08_neural_network",
]


def ensure_dirs(*paths: Path) -> None:
    """
    Create folders if they do not exist.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _json_converter(obj):
    """
    Convert numpy types and Path objects so they can be saved as JSON.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_json(data: dict, path: Path) -> None:
    """
    Save a dictionary as a JSON file.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_converter)


def format_model_name(folder_name: str, model_name: str | None) -> str:
    """
    Create short model labels for plots and tables.
    """
    mapping = {
        "00_dummy_regressor": "Dummy",
        "01_linear_regression": "Linear",
        "02_ridge_regression": "Ridge",
        "03_lasso_regression": "Lasso",
        "04_decision_tree": "Decision Tree",
        "05_knn_regressor": "KNN",
        "06_random_forest": "Random Forest",
        "07_gradient_boosting": "Gradient Boosting",
        "08_neural_network": "Neural Network",
    }

    if folder_name in mapping:
        return mapping[folder_name]

    if model_name:
        cleaned = model_name.replace("Regressor", "").replace("Regression", "")
        return cleaned.strip()

    return folder_name


def collect_model_metrics() -> tuple[pd.DataFrame, list[str]]:
    """
    Collect the metrics.csv file from every model folder.

    Returns:
    - combined metrics dataframe
    - list of skipped folders without usable metrics
    """
    rows: list[pd.DataFrame] = []
    skipped_models: list[str] = []

    for model_dir in sorted(MODELLING_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        if model_dir.name in {"common", "99_model_comparison", "__pycache__"}:
            continue

        metrics_path = model_dir / "results" / "metrics.csv"
        if not metrics_path.exists():
            skipped_models.append(model_dir.name)
            continue

        df = pd.read_csv(metrics_path)
        if df.empty:
            skipped_models.append(model_dir.name)
            continue

        row = df.iloc[[0]].copy()
        row["folder_name"] = model_dir.name
        row["display_name"] = format_model_name(
            model_dir.name,
            row["model_name"].iloc[0] if "model_name" in row.columns else None,
        )
        rows.append(row)

    if not rows:
        raise FileNotFoundError("No model metrics.csv files found in modelling subfolders.")

    combined = pd.concat(rows, ignore_index=True)
    return combined, skipped_models


def sort_models_chronologically(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort models in the fixed project order.

    This is important because the user wanted the overview plots
    to stay chronological and not be reordered by performance.
    """
    df = metrics_df.copy()

    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    df["sort_order"] = df["folder_name"].map(order_map).fillna(999)

    df = df.sort_values(["sort_order", "display_name"]).reset_index(drop=True)
    df = df.drop(columns=["sort_order"])

    return df


def add_rank_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ranking columns for important validation and test metrics.
    """
    df = metrics_df.copy()

    df["rank_validation_rmse"] = df["validation_rmse"].rank(method="min", ascending=True).astype(int)
    df["rank_validation_mae"] = df["validation_mae"].rank(method="min", ascending=True).astype(int)
    df["rank_validation_r2"] = df["validation_r2"].rank(method="min", ascending=False).astype(int)

    df["rank_test_rmse"] = df["test_rmse"].rank(method="min", ascending=True).astype(int)
    df["rank_test_mae"] = df["test_mae"].rank(method="min", ascending=True).astype(int)
    df["rank_test_r2"] = df["test_r2"].rank(method="min", ascending=False).astype(int)

    df["rank_sum_validation"] = (
        df["rank_validation_rmse"] + df["rank_validation_mae"] + df["rank_validation_r2"]
    )
    df["rank_sum_test"] = (
        df["rank_test_rmse"] + df["rank_test_mae"] + df["rank_test_r2"]
    )

    return df


def build_rankings(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build separate ranking tables for validation and test performance.

    These ranking tables are sorted by performance.
    The overview plots are not.
    """
    validation_rank = metrics_df.sort_values(
        by=["validation_rmse", "validation_mae", "validation_mape"]
    ).reset_index(drop=True)
    validation_rank.insert(0, "rank_validation", range(1, len(validation_rank) + 1))

    test_rank = metrics_df.sort_values(
        by=["test_rmse", "test_mae", "test_mape"]
    ).reset_index(drop=True)
    test_rank.insert(0, "rank_test", range(1, len(test_rank) + 1))

    return validation_rank, test_rank


def plot_grouped_metric_highlight(
    metrics_df: pd.DataFrame,
    metric_name: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Plot one grouped comparison chart for one metric.

    We keep the model order exactly as it appears in metrics_df.
    That means the bars stay chronological across all overview plots.
    """
    plot_df = metrics_df.copy()
    x = np.arange(len(plot_df))
    width = 0.24

    train_values = plot_df[f"train_{metric_name}"].to_numpy()
    val_values = plot_df[f"validation_{metric_name}"].to_numpy()
    test_values = plot_df[f"test_{metric_name}"].to_numpy()

    if metric_name in {"r2", "explained_variance"}:
        best_test_idx = int(np.argmax(test_values))
    else:
        best_test_idx = int(np.argmin(test_values))

    train_colors = ["C0"] * len(plot_df)
    val_colors = ["C1"] * len(plot_df)
    test_colors = ["C2"] * len(plot_df)
    test_colors[best_test_idx] = "crimson"

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, train_values, width=width, label="Train", color=train_colors)
    plt.bar(x, val_values, width=width, label="Validation", color=val_colors)
    plt.bar(x + width, test_values, width=width, label="Test", color=test_colors)

    best_height = test_values[best_test_idx]
    plt.text(
        x[best_test_idx] + width,
        best_height,
        " best",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=9,
    )

    plt.title(title, fontsize=TITLE_SIZE)
    plt.xlabel("Model", fontsize=LABEL_SIZE)
    plt.ylabel(metric_name.upper(), fontsize=LABEL_SIZE)
    plt.xticks(x, plot_df["display_name"].tolist(), rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def plot_single_metric_highlight(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    higher_is_better: bool = False,
) -> None:
    """
    Plot one single metric for all models.

    This function keeps the model order from metrics_df unchanged.
    """
    plot_df = metrics_df.copy()
    values = plot_df[metric_col].to_numpy()

    best_idx = int(np.argmax(values)) if higher_is_better else int(np.argmin(values))
    colors = ["C0"] * len(plot_df)
    colors[best_idx] = "crimson"

    x = np.arange(len(plot_df))

    plt.figure(figsize=FIGSIZE)
    plt.bar(x, values, color=colors)
    plt.text(
        x[best_idx],
        values[best_idx],
        "best",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    plt.title(title, fontsize=TITLE_SIZE)
    plt.xlabel("Model", fontsize=LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=LABEL_SIZE)
    plt.xticks(x, plot_df["display_name"].tolist(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def plot_overview_dashboard(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """
    Create one dashboard with the most important validation and test metrics.

    The model order stays chronological.
    """
    plot_df = metrics_df.copy()
    x = np.arange(len(plot_df))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    metrics = [
        ("validation_rmse", "Validation RMSE", False),
        ("test_rmse", "Test RMSE", False),
        ("validation_r2", "Validation R²", True),
        ("test_r2", "Test R²", True),
    ]

    for ax, (metric_col, title, higher_is_better) in zip(axes.flat, metrics):
        values = plot_df[metric_col].to_numpy()
        best_idx = int(np.argmax(values)) if higher_is_better else int(np.argmin(values))
        colors = ["C0"] * len(plot_df)
        colors[best_idx] = "crimson"

        ax.bar(x, values, color=colors)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["display_name"].tolist(), rotation=30, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.text(
            x[best_idx],
            values[best_idx],
            "best",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Model Comparison Overview", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_rank_sum(metrics_df: pd.DataFrame, rank_col: str, title: str, output_path: Path) -> None:
    """
    Plot rank sums.

    Here we sort by rank sum because the whole purpose of this chart
    is to show the ranking result directly.
    """
    plot_df = metrics_df.sort_values(rank_col).reset_index(drop=True)
    x = np.arange(len(plot_df))
    values = plot_df[rank_col].to_numpy()

    colors = ["C0"] * len(plot_df)
    colors[0] = "crimson"

    plt.figure(figsize=FIGSIZE)
    plt.bar(x, values, color=colors)
    plt.text(x[0], values[0], "best", ha="center", va="bottom", fontsize=9)
    plt.title(title, fontsize=TITLE_SIZE)
    plt.xlabel("Model", fontsize=LABEL_SIZE)
    plt.ylabel("Rank Sum", fontsize=LABEL_SIZE)
    plt.xticks(x, plot_df["display_name"].tolist(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def build_summary(
    metrics_df: pd.DataFrame,
    validation_rank_df: pd.DataFrame,
    test_rank_df: pd.DataFrame,
    skipped_models: list[str],
) -> dict:
    """
    Build a compact summary JSON with the most important comparison results.
    """
    best_validation = validation_rank_df.iloc[0]
    best_test = test_rank_df.iloc[0]

    best_validation_ranksum = metrics_df.sort_values("rank_sum_validation").iloc[0]
    best_test_ranksum = metrics_df.sort_values("rank_sum_test").iloc[0]

    return {
        "n_models_compared": int(len(metrics_df)),
        "models_compared": metrics_df["display_name"].tolist(),
        "skipped_models_without_metrics": skipped_models,
        "best_by_validation_rmse": {
            "display_name": best_validation["display_name"],
            "folder_name": best_validation["folder_name"],
            "validation_rmse": best_validation["validation_rmse"],
            "validation_mae": best_validation["validation_mae"],
            "validation_r2": best_validation["validation_r2"],
        },
        "best_by_test_rmse": {
            "display_name": best_test["display_name"],
            "folder_name": best_test["folder_name"],
            "test_rmse": best_test["test_rmse"],
            "test_mae": best_test["test_mae"],
            "test_r2": best_test["test_r2"],
        },
        "best_by_validation_rank_sum": {
            "display_name": best_validation_ranksum["display_name"],
            "folder_name": best_validation_ranksum["folder_name"],
            "rank_sum_validation": best_validation_ranksum["rank_sum_validation"],
        },
        "best_by_test_rank_sum": {
            "display_name": best_test_ranksum["display_name"],
            "folder_name": best_test_ranksum["folder_name"],
            "rank_sum_test": best_test_ranksum["rank_sum_test"],
        },
    }


def main() -> None:
    ensure_dirs(RESULTS_DIR, PLOTS_DIR)

    print("Collecting model metrics...")
    metrics_df, skipped_models = collect_model_metrics()

    # Keep the main comparison dataframe in chronological project order.
    metrics_df = sort_models_chronologically(metrics_df)
    metrics_df = add_rank_columns(metrics_df)

    print(f"Models found: {len(metrics_df)}")
    if skipped_models:
        print("Skipped folders without metrics:")
        for name in skipped_models:
            print(f"  - {name}")

    # Save full metrics table
    metrics_df.to_csv(RESULTS_DIR / "all_model_metrics.csv", index=False)

    # Build separate ranking tables
    validation_rank_df, test_rank_df = build_rankings(metrics_df)
    validation_rank_df.to_csv(RESULTS_DIR / "model_ranking_validation_rmse.csv", index=False)
    test_rank_df.to_csv(RESULTS_DIR / "model_ranking_test_rmse.csv", index=False)

    # Save a smaller comparison table for quick reading
    compact_cols = [
        "display_name",
        "folder_name",
        "validation_mae",
        "validation_rmse",
        "validation_r2",
        "validation_mape",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_mape",
        "fit_time_seconds",
        "predict_time_seconds",
        "rank_sum_validation",
        "rank_sum_test",
    ]
    metrics_df[compact_cols].to_csv(RESULTS_DIR / "model_comparison_compact.csv", index=False)

    print("Creating comparison plots...")

    # Main overview plots in chronological order
    for spec in METRIC_SPECS:
        plot_grouped_metric_highlight(
            metrics_df=metrics_df,
            metric_name=spec["name"],
            title=spec["title"],
            output_path=PLOTS_DIR / f"{spec['name']}_comparison.png",
        )

    plot_single_metric_highlight(
        metrics_df=metrics_df,
        metric_col="fit_time_seconds",
        title="Fit Time Comparison",
        ylabel="Seconds",
        output_path=PLOTS_DIR / "fit_time_comparison.png",
        higher_is_better=False,
    )

    plot_single_metric_highlight(
        metrics_df=metrics_df,
        metric_col="predict_time_seconds",
        title="Predict Time Comparison",
        ylabel="Seconds",
        output_path=PLOTS_DIR / "predict_time_comparison.png",
        higher_is_better=False,
    )

    # Ranking plots
    plot_single_metric_highlight(
        metrics_df=validation_rank_df,
        metric_col="validation_rmse",
        title="Validation RMSE Ranking",
        ylabel="Validation RMSE",
        output_path=PLOTS_DIR / "validation_rmse_ranking.png",
        higher_is_better=False,
    )

    plot_single_metric_highlight(
        metrics_df=test_rank_df,
        metric_col="test_rmse",
        title="Test RMSE Ranking",
        ylabel="Test RMSE",
        output_path=PLOTS_DIR / "test_rmse_ranking.png",
        higher_is_better=False,
    )

    plot_rank_sum(
        metrics_df=metrics_df,
        rank_col="rank_sum_validation",
        title="Validation Rank Sum Comparison",
        output_path=PLOTS_DIR / "validation_rank_sum.png",
    )

    plot_rank_sum(
        metrics_df=metrics_df,
        rank_col="rank_sum_test",
        title="Test Rank Sum Comparison",
        output_path=PLOTS_DIR / "test_rank_sum.png",
    )

    plot_overview_dashboard(
        metrics_df=metrics_df,
        output_path=PLOTS_DIR / "model_comparison_overview.png",
    )

    summary = build_summary(metrics_df, validation_rank_df, test_rank_df, skipped_models)
    save_json(summary, RESULTS_DIR / "comparison_summary.json")

    print("\nValidation ranking:")
    print(
        validation_rank_df[
            ["rank_validation", "display_name", "validation_rmse", "validation_mae", "validation_r2"]
        ].to_string(index=False)
    )

    print("\nTest ranking:")
    print(
        test_rank_df[
            ["rank_test", "display_name", "test_rmse", "test_mae", "test_r2"]
        ].to_string(index=False)
    )

    print(f"\nDone. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()