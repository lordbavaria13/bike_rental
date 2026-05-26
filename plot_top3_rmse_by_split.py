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
MID_GREY = "#8C8C8C"
LIGHT_GREY = "#D9D9D9"
VERY_LIGHT_GREY = "#EDEDED"

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "modelling"

OUTPUT_DIR = ROOT / "presentation_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENT = "with_lag"

DISPLAY_NAME_MAP = {
    "LinearRegression": "Linear",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "DecisionTreeRegressor": "Decision Tree",
    "KNeighborsRegressor": "KNN",
    "RandomForestRegressor": "Random Forest",
    "GradientBoostingRegressor": "Gradient Boosting",
    "MLPRegressor": "Neural Network",
    "NeuralNetwork": "Neural Network",
    "Neural Network": "Neural Network",
}


# ============================================================
# Helper functions
# ============================================================

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def get_display_name(metrics: dict, path: Path) -> str:
    model_name = metrics.get("model_name", "")

    if model_name in DISPLAY_NAME_MAP:
        return DISPLAY_NAME_MAP[model_name]

    path_text = normalize(path)

    if "gradient" in path_text or "boost" in path_text:
        return "Gradient Boosting"
    if "random" in path_text or "forest" in path_text:
        return "Random Forest"
    if "lasso" in path_text:
        return "Lasso"
    if "ridge" in path_text:
        return "Ridge"
    if "linear" in path_text:
        return "Linear"
    if "decision" in path_text or "tree" in path_text:
        return "Decision Tree"
    if "knn" in path_text or "nearest" in path_text:
        return "KNN"
    if "neural" in path_text or "mlp" in path_text:
        return "Neural Network"

    return model_name or path.parent.parent.parent.name


def load_top3_metrics() -> pd.DataFrame:
    rows = []

    metrics_files = sorted(RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"))

    if not metrics_files:
        raise FileNotFoundError(
            f"No {EXPERIMENT}/metrics.json files found under {RESULTS_ROOT}"
        )

    for metrics_path in metrics_files:
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        required_keys = ["train_rmse", "validation_rmse", "test_rmse"]

        if not all(key in metrics for key in required_keys):
            continue

        if metrics.get("uses_lag_features") is not True:
            continue

        model_label = get_display_name(metrics, metrics_path)

        if "dummy" in normalize(model_label) or "dummy" in normalize(metrics_path):
            continue

        rows.append(
            {
                "Model": model_label,
                "Train RMSE": float(metrics["train_rmse"]),
                "Validation RMSE": float(metrics["validation_rmse"]),
                "Test RMSE": float(metrics["test_rmse"]),
                "Path": str(metrics_path.relative_to(ROOT)),
            }
        )

    if not rows:
        raise ValueError("No valid with-lag model metrics found.")

    df = pd.DataFrame(rows)

    df = (
        df.sort_values("Test RMSE", ascending=True)
        .drop_duplicates(subset=["Model"], keep="first")
    )

    top3 = df.sort_values("Test RMSE", ascending=True).head(3)

    return top3


# ============================================================
# Plot
# ============================================================

def main() -> None:
    df = load_top3_metrics()

    print("\nTop 3 models by Test RMSE:")
    print(
        df[
            ["Model", "Train RMSE", "Validation RMSE", "Test RMSE", "Path"]
        ].to_string(index=False)
    )

    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    train_bars = ax.bar(
        x - width,
        df["Train RMSE"],
        width,
        label="Train",
        color=VERY_LIGHT_GREY,
        edgecolor=LIGHT_GREY,
        linewidth=0.8,
    )

    validation_bars = ax.bar(
        x,
        df["Validation RMSE"],
        width,
        label="Validation",
        color=MID_GREY,
    )

    test_bars = ax.bar(
        x + width,
        df["Test RMSE"],
        width,
        label="Test",
        color=GREEN,
    )

    ax.set_title(
        "RMSE by Split for the Top 3 Models",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=42,
    )

    ax.set_xlabel("Model", fontsize=12, color=DARK_GREY)
    ax.set_ylabel("RMSE Score", fontsize=12, color=DARK_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    ax.tick_params(axis="x", labelsize=10, colors=DARK_GREY)
    ax.tick_params(axis="y", labelsize=10, colors=DARK_GREY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)

    ymax = max(
        df["Train RMSE"].max(),
        df["Validation RMSE"].max(),
        df["Test RMSE"].max(),
    ) * 1.28

    ax.set_ylim(0, ymax)

    for bars in [train_bars, validation_bars, test_bars]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.012,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=DARK_GREY,
                fontweight="bold",
            )

    # Legend above the plot, not inside the chart area
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GREY,
        fontsize=11,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "top3_rmse_train_validation_test.png"
    pdf_path = OUTPUT_DIR / "top3_rmse_train_validation_test.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved plots:")
    print(f"- {png_path.relative_to(ROOT)}")
    print(f"- {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()