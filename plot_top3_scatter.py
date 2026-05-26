from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

GREEN = "#92D401"
DARK_GREY = "#555555"
LIGHT_GREY = "#D9D9D9"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "presentation_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "total_rentals"
PRED_COL = "prediction"
SPLIT_COL = "split"

MODEL_CONFIGS = [
    {
        "label": "Gradient Boosting",
        "result_dir": ROOT / "modelling" / "07_gradient_boosting" / "results" / "with_lag",
    },
    {
        "label": "Random Forest",
        "result_dir": ROOT / "modelling" / "06_random_forest" / "results" / "with_lag",
    },
    {
        "label": "Lasso",
        "result_dir": ROOT / "modelling" / "03_lasso_regression" / "results" / "with_lag",
    },
]


# ============================================================
# Helpers
# ============================================================

def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_test_predictions(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    df = pd.read_csv(predictions_path)

    required_cols = {TARGET_COL, PRED_COL, SPLIT_COL}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing columns in {predictions_path}: {sorted(missing_cols)}\n"
            f"Available columns: {list(df.columns)}"
        )

    test_df = df[df[SPLIT_COL].astype(str).str.lower() == "test"].copy()

    if test_df.empty:
        raise ValueError(f"No test rows found in {predictions_path}")

    test_df[TARGET_COL] = pd.to_numeric(test_df[TARGET_COL], errors="coerce")
    test_df[PRED_COL] = pd.to_numeric(test_df[PRED_COL], errors="coerce")

    test_df = test_df.dropna(subset=[TARGET_COL, PRED_COL])

    if test_df.empty:
        raise ValueError(f"No valid numeric test predictions found in {predictions_path}")

    return test_df


def compute_metrics(test_df: pd.DataFrame) -> dict:
    y_true = test_df[TARGET_COL].to_numpy()
    y_pred = test_df[PRED_COL].to_numpy()

    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def check_consistency(model_label: str, computed: dict, stored: dict) -> None:
    stored_mae = stored["test_mae"]
    stored_rmse = stored["test_rmse"]
    stored_r2 = stored["test_r2"]

    mae_diff = abs(computed["mae"] - stored_mae)
    rmse_diff = abs(computed["rmse"] - stored_rmse)
    r2_diff = abs(computed["r2"] - stored_r2)

    print(f"\n{model_label}")
    print(f"Computed from predictions: MAE={computed['mae']:.4f}, RMSE={computed['rmse']:.4f}, R²={computed['r2']:.4f}")
    print(f"Stored in metrics.json:   MAE={stored_mae:.4f}, RMSE={stored_rmse:.4f}, R²={stored_r2:.4f}")

    if mae_diff > 1e-6 or rmse_diff > 1e-6 or r2_diff > 1e-6:
        print("WARNING: predictions.csv and metrics.json are not exactly consistent.")
    else:
        print("OK: predictions.csv and metrics.json are consistent.")


# ============================================================
# Main plotting
# ============================================================

def main() -> None:
    plot_data = []

    for config in MODEL_CONFIGS:
        label = config["label"]
        result_dir = config["result_dir"]

        predictions_path = result_dir / "predictions.csv"
        metrics_path = result_dir / "metrics.json"

        test_df = load_test_predictions(predictions_path)
        stored_metrics = load_metrics(metrics_path)
        computed_metrics = compute_metrics(test_df)

        check_consistency(label, computed_metrics, stored_metrics)

        plot_data.append(
            {
                "label": label,
                "test_df": test_df,
                "metrics": stored_metrics,
            }
        )

    all_actual = np.concatenate(
        [item["test_df"][TARGET_COL].to_numpy() for item in plot_data]
    )
    all_predicted = np.concatenate(
        [item["test_df"][PRED_COL].to_numpy() for item in plot_data]
    )

    axis_min = 0
    axis_max = max(all_actual.max(), all_predicted.max()) * 1.05

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.8),
        sharex=True,
        sharey=True,
    )

    for ax, item in zip(axes, plot_data):
        label = item["label"]
        test_df = item["test_df"]
        metrics = item["metrics"]

        ax.scatter(
            test_df[TARGET_COL],
            test_df[PRED_COL],
            s=14,
            alpha=0.35,
            color=GREEN,
            edgecolors="none",
            rasterized=True,
        )

        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            linewidth=1.4,
            color=DARK_GREY,
        )

        ax.set_title(
            label,
            fontsize=15,
            fontweight="bold",
            color=DARK_GREY,
            pad=10,
        )

        metric_text = (
            f"RMSE: {metrics['test_rmse']:.2f}\n"
            f"MAE: {metrics['test_mae']:.2f}\n"
            f"$R^2$: {metrics['test_r2']:.3f}"
        )

        ax.text(
            0.05,
            0.95,
            metric_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color=DARK_GREY,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=LIGHT_GREY,
                alpha=0.92,
            ),
        )

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.grid(True, color=LIGHT_GREY, linewidth=0.6, alpha=0.7)
        ax.tick_params(axis="both", labelsize=9, colors=DARK_GREY)
        ax.set_xlabel("Actual rentals", fontsize=11, color=DARK_GREY)

    axes[0].set_ylabel("Predicted rentals", fontsize=11, color=DARK_GREY)

    fig.suptitle(
        "Actual vs. Predicted Daily Rentals on Test Data",
        fontsize=17,
        fontweight="bold",
        color=DARK_GREY,
        y=1.04,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "top3_actual_vs_predicted_scatter.png"
    pdf_path = OUTPUT_DIR / "top3_actual_vs_predicted_scatter.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved plots:")
    print(f"- {png_path.relative_to(ROOT)}")
    print(f"- {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()