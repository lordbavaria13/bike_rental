from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from modelling.common.config import DPI, EXPERIMENTS
from modelling.common.utils import ensure_dirs, save_json

BASE_DIR = Path(__file__).resolve().parents[2]
MODELLING_DIR = BASE_DIR / "modelling"
THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

FIGSIZE = (12, 5)
TITLE_SIZE = 13
LABEL_SIZE = 11

# Consistent colors for the two experiment variants in all comparison plots.
# This keeps the current plot structure unchanged, but makes the lag setup visible.
EXPERIMENT_COLORS = {
    "without_lag": "#4C78A8",  # blue
    "with_lag": "#F58518",     # orange
}
EXPERIMENT_LABELS = {
    "without_lag": "without lag",
    "with_lag": "with lag",
}

MODEL_FOLDERS = [
    ("00_dummy_regressor", "Dummy"),
    ("01_linear_regression", "Linear"),
    ("02_ridge_regression", "Ridge"),
    ("03_lasso_regression", "Lasso"),
    ("04_decision_tree", "Decision Tree"),
    ("05_knn_regressor", "KNN"),
    ("06_random_forest", "Random Forest"),
    ("07_gradient_boosting", "Gradient Boosting"),
    ("08_neural_network", "Neural Network"),
]

LOWER_IS_BETTER = ["mae", "rmse", "median_ae", "mape"]
HIGHER_IS_BETTER = ["r2", "explained_variance"]


def collect_model_metrics() -> tuple[pd.DataFrame, list[str]]:
    records: list[pd.DataFrame] = []
    skipped: list[str] = []

    for order, (folder_name, display_name) in enumerate(MODEL_FOLDERS):
        for experiment in EXPERIMENTS:
            metrics_path = MODELLING_DIR / folder_name / "results" / experiment / "metrics.csv"
            if not metrics_path.exists():
                skipped.append(f"{folder_name}/{experiment}")
                continue

            df = pd.read_csv(metrics_path)
            if df.empty:
                skipped.append(f"{folder_name}/{experiment} (empty)")
                continue

            row = df.iloc[[0]].copy()
            row["folder_name"] = folder_name
            row["display_name"] = display_name
            row["model_order"] = order
            row["experiment"] = experiment
            row["variant_label"] = "with lag" if experiment == "with_lag" else "without lag"
            row["display_variant"] = row["display_name"] + " (" + row["variant_label"] + ")"
            records.append(row)

    if not records:
        raise FileNotFoundError(
            "No model metrics found. Run the model scripts before the comparison."
        )

    metrics_df = pd.concat(records, ignore_index=True)
    metrics_df = metrics_df.sort_values(["model_order", "experiment"]).reset_index(drop=True)
    return metrics_df, skipped


def add_rank_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = metrics_df.copy()

    for split in ["validation", "test"]:
        rank_cols: list[str] = []
        for metric in LOWER_IS_BETTER:
            col = f"{split}_{metric}"
            if col in out.columns:
                rank_col = f"rank_{split}_{metric}"
                out[rank_col] = out[col].rank(method="min", ascending=True)
                rank_cols.append(rank_col)
        for metric in HIGHER_IS_BETTER:
            col = f"{split}_{metric}"
            if col in out.columns:
                rank_col = f"rank_{split}_{metric}"
                out[rank_col] = out[col].rank(method="min", ascending=False)
                rank_cols.append(rank_col)
        out[f"rank_sum_{split}"] = out[rank_cols].sum(axis=1)

    return out


def build_rankings(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_rank_df = metrics_df.sort_values(
        ["validation_rmse", "validation_mae"], ascending=[True, True]
    ).reset_index(drop=True)
    validation_rank_df["rank_validation"] = np.arange(1, len(validation_rank_df) + 1)

    test_rank_df = metrics_df.sort_values(
        ["test_rmse", "test_mae"], ascending=[True, True]
    ).reset_index(drop=True)
    test_rank_df["rank_test"] = np.arange(1, len(test_rank_df) + 1)

    return validation_rank_df, test_rank_df


def build_lag_delta_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for folder_name, display_name in MODEL_FOLDERS:
        model_df = metrics_df[metrics_df["folder_name"] == folder_name]
        if set(model_df["experiment"]) != set(EXPERIMENTS):
            continue

        without_row = model_df[model_df["experiment"] == "without_lag"].iloc[0]
        with_row = model_df[model_df["experiment"] == "with_lag"].iloc[0]

        row = {
            "folder_name": folder_name,
            "display_name": display_name,
            "test_rmse_without_lag": without_row["test_rmse"],
            "test_rmse_with_lag": with_row["test_rmse"],
            "test_rmse_delta_with_minus_without": with_row["test_rmse"] - without_row["test_rmse"],
            "test_rmse_improvement_percent": (
                (without_row["test_rmse"] - with_row["test_rmse"]) / without_row["test_rmse"] * 100
            ),
            "test_mae_without_lag": without_row["test_mae"],
            "test_mae_with_lag": with_row["test_mae"],
            "test_mae_delta_with_minus_without": with_row["test_mae"] - without_row["test_mae"],
            "test_r2_without_lag": without_row["test_r2"],
            "test_r2_with_lag": with_row["test_r2"],
            "test_r2_delta_with_minus_without": with_row["test_r2"] - without_row["test_r2"],
            "validation_rmse_without_lag": without_row["validation_rmse"],
            "validation_rmse_with_lag": with_row["validation_rmse"],
            "validation_rmse_delta_with_minus_without": with_row["validation_rmse"] - without_row["validation_rmse"],
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("test_rmse_delta_with_minus_without")


def plot_grouped_variant_metric(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plot_df = metrics_df.pivot(index="display_name", columns="experiment", values=metric_col)
    plot_df = plot_df.reindex([display for _, display in MODEL_FOLDERS])

    x = np.arange(len(plot_df))
    width = 0.38

    plt.figure(figsize=FIGSIZE)
    if "without_lag" in plot_df.columns:
        plt.bar(
            x - width / 2,
            plot_df["without_lag"],
            width=width,
            label=EXPERIMENT_LABELS["without_lag"],
            color=EXPERIMENT_COLORS["without_lag"],
        )
    if "with_lag" in plot_df.columns:
        plt.bar(
            x + width / 2,
            plot_df["with_lag"],
            width=width,
            label=EXPERIMENT_LABELS["with_lag"],
            color=EXPERIMENT_COLORS["with_lag"],
        )

    plt.title(title, fontsize=TITLE_SIZE)
    plt.xlabel("Model", fontsize=LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=LABEL_SIZE)
    plt.xticks(x, plot_df.index.tolist(), rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def plot_delta_bar(
    delta_df: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if delta_df.empty:
        return

    plot_df = delta_df.sort_values(metric_col).copy()
    x = np.arange(len(plot_df))

    plt.figure(figsize=FIGSIZE)
    plt.bar(x, plot_df[metric_col])
    plt.axhline(0, linestyle="--")
    plt.title(title, fontsize=TITLE_SIZE)
    plt.xlabel("Model", fontsize=LABEL_SIZE)
    plt.ylabel(ylabel, fontsize=LABEL_SIZE)
    plt.xticks(x, plot_df["display_name"].tolist(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def plot_overview(metrics_df: pd.DataFrame, output_path: Path) -> None:
    compact = metrics_df.copy()
    compact = compact.sort_values(["model_order", "experiment"])
    x = np.arange(len(compact))

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    plot_specs = [
        ("validation_rmse", "Validation RMSE"),
        ("test_rmse", "Test RMSE"),
        ("validation_r2", "Validation R²"),
        ("test_r2", "Test R²"),
    ]

    bar_colors = compact["experiment"].map(EXPERIMENT_COLORS).fillna("#999999").tolist()

    for ax, (metric_col, title) in zip(axes.flat, plot_specs):
        ax.bar(x, compact[metric_col].to_numpy(), color=bar_colors)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(compact["display_variant"].tolist(), rotation=45, ha="right", fontsize=8)

    legend_handles = [
        Patch(facecolor=EXPERIMENT_COLORS["without_lag"], label=EXPERIMENT_LABELS["without_lag"]),
        Patch(facecolor=EXPERIMENT_COLORS["with_lag"], label=EXPERIMENT_LABELS["with_lag"]),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Model Comparison: without lag vs with lag", fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def build_summary(
    metrics_df: pd.DataFrame,
    validation_rank_df: pd.DataFrame,
    test_rank_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    skipped: list[str],
) -> dict:
    best_validation = validation_rank_df.iloc[0]
    best_test = test_rank_df.iloc[0]

    summary = {
        "n_rows_compared": int(len(metrics_df)),
        "n_models": int(metrics_df["folder_name"].nunique()),
        "experiments": list(EXPERIMENTS),
        "skipped_model_experiments": skipped,
        "best_by_validation_rmse": {
            "display_variant": best_validation["display_variant"],
            "validation_rmse": float(best_validation["validation_rmse"]),
            "validation_mae": float(best_validation["validation_mae"]),
            "validation_r2": float(best_validation["validation_r2"]),
        },
        "best_by_test_rmse": {
            "display_variant": best_test["display_variant"],
            "test_rmse": float(best_test["test_rmse"]),
            "test_mae": float(best_test["test_mae"]),
            "test_r2": float(best_test["test_r2"]),
        },
    }

    if not delta_df.empty:
        best_lag_gain = delta_df.iloc[0]
        summary["largest_test_rmse_gain_from_lag"] = {
            "display_name": best_lag_gain["display_name"],
            "test_rmse_delta_with_minus_without": float(best_lag_gain["test_rmse_delta_with_minus_without"]),
            "test_rmse_improvement_percent": float(best_lag_gain["test_rmse_improvement_percent"]),
        }

    return summary


def main() -> None:
    ensure_dirs(RESULTS_DIR, PLOTS_DIR)

    print("Collecting model metrics for both experiment variants...")
    metrics_df, skipped = collect_model_metrics()
    metrics_df = add_rank_columns(metrics_df)
    validation_rank_df, test_rank_df = build_rankings(metrics_df)
    delta_df = build_lag_delta_table(metrics_df)

    metrics_df.to_csv(RESULTS_DIR / "all_model_metrics.csv", index=False)
    validation_rank_df.to_csv(RESULTS_DIR / "model_ranking_validation_rmse.csv", index=False)
    test_rank_df.to_csv(RESULTS_DIR / "model_ranking_test_rmse.csv", index=False)
    delta_df.to_csv(RESULTS_DIR / "lag_delta_comparison.csv", index=False)

    compact_cols = [
        "display_name",
        "experiment",
        "variant_label",
        "n_features",
        "n_train",
        "n_validation",
        "n_test",
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
    available_compact_cols = [col for col in compact_cols if col in metrics_df.columns]
    metrics_df[available_compact_cols].to_csv(
        RESULTS_DIR / "model_comparison_compact.csv",
        index=False,
    )

    plot_grouped_variant_metric(
        metrics_df,
        "test_rmse",
        "Test RMSE: without lag vs with lag",
        "Test RMSE",
        PLOTS_DIR / "test_rmse_without_vs_with_lag.png",
    )
    plot_grouped_variant_metric(
        metrics_df,
        "test_mae",
        "Test MAE: without lag vs with lag",
        "Test MAE",
        PLOTS_DIR / "test_mae_without_vs_with_lag.png",
    )
    plot_grouped_variant_metric(
        metrics_df,
        "test_r2",
        "Test R²: without lag vs with lag",
        "Test R²",
        PLOTS_DIR / "test_r2_without_vs_with_lag.png",
    )
    plot_delta_bar(
        delta_df,
        "test_rmse_delta_with_minus_without",
        "Lag effect on Test RMSE (negative is better)",
        "Test RMSE with lag - without lag",
        PLOTS_DIR / "test_rmse_lag_delta.png",
    )
    plot_overview(metrics_df, PLOTS_DIR / "model_comparison_overview.png")

    summary = build_summary(metrics_df, validation_rank_df, test_rank_df, delta_df, skipped)
    save_json(summary, RESULTS_DIR / "comparison_summary.json")

    print("\nTest ranking:")
    print(test_rank_df[["rank_test", "display_variant", "test_rmse", "test_mae", "test_r2"]].to_string(index=False))

    print("\nLag delta comparison:")
    if delta_df.empty:
        print("No complete with/without lag pairs found.")
    else:
        print(delta_df[["display_name", "test_rmse_without_lag", "test_rmse_with_lag", "test_rmse_delta_with_minus_without"]].to_string(index=False))

    if skipped:
        print("\nSkipped missing metric files:")
        for item in skipped:
            print(f"  - {item}")

    print(f"\nDone. Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
