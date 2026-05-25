from pathlib import Path
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
OUTPUT_DIR = ROOT / "presentation_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Optional manual override:
# Put exact file paths here if automatic search does not find the right files.
MANUAL_FILES = {
    "Gradient Boosting": None,
    "Random Forest": None,
    "Lasso": None,
}

MODELS = [
    {
        "label": "Gradient Boosting",
        "aliases": ["gradient_boost", "gradient_boosting", "gradientboost", "gb"],
    },
    {
        "label": "Random Forest",
        "aliases": ["random_forest", "randomforest", "rf"],
    },
    {
        "label": "Lasso",
        "aliases": ["lasso"],
    },
]

TRUE_COL_CANDIDATES = [
    "y_true",
    "true",
    "actual",
    "actuals",
    "observed",
    "target",
    "total_rentals",
    "total_rentals_true",
    "y_test",
]

PRED_COL_CANDIDATES = [
    "y_pred",
    "pred",
    "prediction",
    "predictions",
    "predicted",
    "predicted_total_rentals",
    "total_rentals_pred",
    "y_hat",
]


# ============================================================
# Helper functions
# ============================================================

def normalize_text(text: str) -> str:
    """Lowercase and replace non-alphanumeric characters with underscores."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a matching column by normalized column names."""
    normalized_map = {normalize_text(col): col for col in df.columns}

    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        if candidate_norm in normalized_map:
            return normalized_map[candidate_norm]

    return None


def has_prediction_columns(csv_path: Path) -> bool:
    """Quickly test whether a CSV probably contains actual and predicted values."""
    try:
        df_head = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return False

    true_col = find_column(df_head, TRUE_COL_CANDIDATES)
    pred_col = find_column(df_head, PRED_COL_CANDIDATES)

    return true_col is not None and pred_col is not None


def score_candidate_file(csv_path: Path, aliases: list[str]) -> int:
    """Score candidate files so the script prefers test predictions with lag."""
    path_norm = normalize_text(str(csv_path.relative_to(ROOT)))

    if not any(alias in path_norm for alias in aliases):
        return -999

    score = 0

    # Strongly prefer prediction files
    if "prediction" in path_norm or "predictions" in path_norm:
        score += 10

    # Prefer test data
    if "test" in path_norm:
        score += 8

    # Prefer lag / with_lag variant
    if "with_lag" in path_norm or "lag_yes" in path_norm or "lag" in path_norm:
        score += 4

    # Penalize no-lag variants
    if "no_lag" in path_norm or "without_lag" in path_norm:
        score -= 12

    # Penalize training or validation predictions
    if "train" in path_norm:
        score -= 8
    if "validation" in path_norm or "_val_" in path_norm:
        score -= 6

    # Penalize metric/ranking files
    bad_keywords = ["metrics", "ranking", "comparison", "summary", "feature_importance"]
    if any(keyword in path_norm for keyword in bad_keywords):
        score -= 15

    return score


def find_prediction_file(model_label: str, aliases: list[str]) -> Path:
    """Find the best matching prediction file for a model."""
    manual_path = MANUAL_FILES.get(model_label)

    if manual_path:
        path = ROOT / manual_path
        if not path.exists():
            raise FileNotFoundError(f"Manual file for {model_label} not found: {path}")
        return path

    csv_files = list(ROOT.rglob("*.csv"))

    candidates = []
    for csv_path in csv_files:
        if not has_prediction_columns(csv_path):
            continue

        score = score_candidate_file(csv_path, aliases)
        if score > -999:
            candidates.append((score, csv_path))

    if not candidates:
        print("\nAvailable CSV files:")
        for csv_path in csv_files:
            print(" -", csv_path.relative_to(ROOT))

        raise FileNotFoundError(
            f"\nNo prediction CSV found for {model_label}.\n"
            f"Either rename your prediction files clearly or set MANUAL_FILES in the script."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = candidates[0]

    print(f"{model_label}: {best_path.relative_to(ROOT)}  [score={best_score}]")
    return best_path


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load actual and predicted values from a CSV file."""
    df = pd.read_csv(csv_path)

    true_col = find_column(df, TRUE_COL_CANDIDATES)
    pred_col = find_column(df, PRED_COL_CANDIDATES)

    if true_col is None or pred_col is None:
        raise ValueError(
            f"Could not identify true/prediction columns in {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    y_true = pd.to_numeric(df[true_col], errors="coerce")
    y_pred = pd.to_numeric(df[pred_col], errors="coerce")

    clean = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()

    return clean["y_true"].to_numpy(), clean["y_pred"].to_numpy()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 without external sklearn dependency."""
    errors = y_true - y_pred

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ============================================================
# Main plotting
# ============================================================

def main() -> None:
    model_data = []

    for model in MODELS:
        csv_path = find_prediction_file(model["label"], model["aliases"])
        y_true, y_pred = load_predictions(csv_path)
        metrics = regression_metrics(y_true, y_pred)

        model_data.append(
            {
                "label": model["label"],
                "path": csv_path,
                "y_true": y_true,
                "y_pred": y_pred,
                "metrics": metrics,
            }
        )

    # Shared axes across all plots
    all_true = np.concatenate([item["y_true"] for item in model_data])
    all_pred = np.concatenate([item["y_pred"] for item in model_data])

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())

    padding = 0.05 * (max_val - min_val)
    axis_min = max(0, min_val - padding)
    axis_max = max_val + padding

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)

    for ax, item in zip(axes, model_data):
        y_true = item["y_true"]
        y_pred = item["y_pred"]
        metrics = item["metrics"]

        ax.scatter(
            y_true,
            y_pred,
            s=14,
            alpha=0.35,
            color=GREEN,
            edgecolors="none",
        )

        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color=DARK_GREY,
            linewidth=1.4,
            linestyle="--",
        )

        ax.set_title(
            item["label"],
            fontsize=15,
            fontweight="bold",
            color=DARK_GREY,
            pad=10,
        )

        metric_text = (
            f"RMSE: {metrics['RMSE']:.2f}\n"
            f"MAE: {metrics['MAE']:.2f}\n"
            f"$R^2$: {metrics['R2']:.3f}"
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
                alpha=0.90,
            ),
        )

        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.grid(True, color=LIGHT_GREY, linewidth=0.6, alpha=0.7)

        ax.tick_params(axis="both", labelsize=9, colors=DARK_GREY)

    axes[0].set_ylabel("Predicted rentals", fontsize=12, color=DARK_GREY)
    for ax in axes:
        ax.set_xlabel("Actual rentals", fontsize=12, color=DARK_GREY)

    fig.suptitle(
        "Actual vs. Predicted Daily Rentals on Test Data",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        y=1.03,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "top3_actual_vs_predicted_scatter.png"
    pdf_path = OUTPUT_DIR / "top3_actual_vs_predicted_scatter.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print("\nSaved plots:")
    print(" -", png_path.relative_to(ROOT))
    print(" -", pdf_path.relative_to(ROOT))


if __name__ == "__main__":
    main()