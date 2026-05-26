from pathlib import Path
import json
import re
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
RESULTS_ROOT = ROOT / "modelling"

OUTPUT_DIR = ROOT / "presentation_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENT = "with_lag"

TARGET_COL = "total_rentals"
PRED_COL = "prediction"
SPLIT_COL = "split"

SELECTED_MODELS = [
    {
        "label": "Gradient Boosting",
        "aliases": ["gradient_boost", "gradient_boosting", "gradientboost", "boost"],
    },
    {
        "label": "Neural Network",
        "aliases": ["neural_network", "neural", "mlp", "nn"],
    },
    {
        "label": "Lasso",
        "aliases": ["lasso"],
    },
]


# ============================================================
# Helpers
# ============================================================

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def path_matches_model(path: Path, aliases: list[str]) -> bool:
    path_text = normalize(path.relative_to(ROOT))
    return any(alias in path_text for alias in aliases)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_test_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_csv(path)

    required_cols = {TARGET_COL, PRED_COL, SPLIT_COL}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing_cols)}\n"
            f"Available columns: {list(df.columns)}"
        )

    test_df = df[df[SPLIT_COL].astype(str).str.lower() == "test"].copy()

    if test_df.empty:
        raise ValueError(f"No test rows found in {path}")

    test_df[TARGET_COL] = pd.to_numeric(test_df[TARGET_COL], errors="coerce")
    test_df[PRED_COL] = pd.to_numeric(test_df[PRED_COL], errors="coerce")

    test_df = test_df.dropna(subset=[TARGET_COL, PRED_COL])

    if test_df.empty:
        raise ValueError(f"No valid numeric test predictions found in {path}")

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
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    }


def find_result_dir(model_config: dict) -> Path:
    label = model_config["label"]
    aliases = model_config["aliases"]

    candidates = []

    for metrics_path in RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"):
        path_text = normalize(metrics_path.relative_to(ROOT))

        if not path_matches_model(metrics_path, aliases):
            continue

        # Avoid selecting Random Forest accidentally for NN
        if label == "Neural Network":
            if "random" in path_text or "forest" in path_text:
                continue

        result_dir = metrics_path.parent
        predictions_path = result_dir / "predictions.csv"

        if not predictions_path.exists():
            continue

        metrics = load_json(metrics_path)

        if metrics.get("uses_lag_features") is not True:
            continue

        if not all(key in metrics for key in ["test_mae", "test_rmse", "test_r2"]):
            continue

        candidates.append(
            {
                "result_dir": result_dir,
                "metrics": metrics,
                "path": metrics_path,
            }
        )

    if not candidates:
        print(f"\nNo valid result directory found for {label}.")
        print("Available with_lag metrics files:")
        for path in RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"):
            print("-", path.relative_to(ROOT))
        raise FileNotFoundError(f"No valid result directory found for {label}")

    # If duplicates exist, use best test RMSE for that selected model
    candidates = sorted(candidates, key=lambda x: x["metrics"]["test_rmse"])
    return candidates[0]["result_dir"]


def load_selected_model_data() -> list[dict]:
    plot_data = []

    for model_config in SELECTED_MODELS:
        label = model_config["label"]
        result_dir = find_result_dir(model_config)

        predictions_path = result_dir / "predictions.csv"
        metrics_path = result_dir / "metrics.json"

        test_df = load_test_predictions(predictions_path)
        stored_metrics = load_json(metrics_path)
        computed_metrics = compute_metrics(test_df)

        print(f"\n{label}")
        print(f"Result dir: {result_dir.relative_to(ROOT)}")
        print(
            "Stored metrics:   "
            f"RMSE={stored_metrics['test_rmse']:.2f}, "
            f"MAE={stored_metrics['test_mae']:.2f}, "
            f"R²={stored_metrics['test_r2']:.3f}"
        )
        print(
            "Computed metrics: "
            f"RMSE={computed_metrics['test_rmse']:.2f}, "
            f"MAE={computed_metrics['test_mae']:.2f}, "
            f"R²={computed_metrics['test_r2']:.3f}"
        )

        plot_data.append(
            {
                "label": label,
                "test_df": test_df,
                "metrics": stored_metrics,
            }
        )

    return plot_data


# ============================================================
# Plot
# ============================================================

def main() -> None:
    plot_data = load_selected_model_data()

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

        ax.text(
            0.05,
            0.95,
            f"RMSE: {metrics['test_rmse']:.2f}\n"
            f"MAE: {metrics['test_mae']:.2f}\n"
            f"$R^2$: {metrics['test_r2']:.3f}",
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

    png_path = OUTPUT_DIR / "selected_final_models_actual_vs_predicted_scatter.png"
    pdf_path = OUTPUT_DIR / "selected_final_models_actual_vs_predicted_scatter.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved plots:")
    print(f"- {png_path.relative_to(ROOT)}")
    print(f"- {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()