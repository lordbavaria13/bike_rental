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

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "modelling"

OUTPUT_DIR = ROOT / "presentation_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_ORDER = [
    "Linear",
    "Ridge",
    "Lasso",
    "Tree",
    "KNN",
    "RF",
    "GB",
    "NN",
]

DISPLAY_NAME_MAP = {
    "LinearRegression": "Linear",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "DecisionTreeRegressor": "Tree",
    "KNeighborsRegressor": "KNN",
    "RandomForestRegressor": "RF",
    "GradientBoostingRegressor": "GB",
    "MLPRegressor": "NN",
    "NeuralNetwork": "NN",
    "Neural Network": "NN",
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

    if "linear" in path_text:
        return "Linear"
    if "ridge" in path_text:
        return "Ridge"
    if "lasso" in path_text:
        return "Lasso"
    if "decision" in path_text or "tree" in path_text:
        return "Tree"
    if "knn" in path_text or "nearest" in path_text:
        return "KNN"
    if "random" in path_text or "forest" in path_text:
        return "RF"
    if "gradient" in path_text or "boost" in path_text:
        return "GB"
    if "neural" in path_text or "mlp" in path_text:
        return "NN"

    return model_name or path.parent.parent.parent.name


def load_metrics_for_experiment(experiment: str) -> pd.DataFrame:
    rows = []

    metrics_files = sorted(RESULTS_ROOT.rglob(f"{experiment}/metrics.json"))

    if not metrics_files:
        raise FileNotFoundError(
            f"No {experiment}/metrics.json files found under modelling/."
        )

    for metrics_path in metrics_files:
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        if "test_rmse" not in metrics:
            continue

        if experiment == "with_lag" and metrics.get("uses_lag_features") is not True:
            continue

        if experiment == "without_lag" and metrics.get("uses_lag_features") is True:
            continue

        model_label = get_display_name(metrics, metrics_path)

        if "dummy" in normalize(model_label) or "dummy" in normalize(metrics_path):
            continue

        rows.append(
            {
                "Model": model_label,
                experiment: float(metrics["test_rmse"]),
                "Path": str(metrics_path.relative_to(ROOT)),
            }
        )

    if not rows:
        raise ValueError(f"No valid metrics found for {experiment}.")

    df = pd.DataFrame(rows)

    # If duplicate model runs exist, keep the best test RMSE
    df = (
        df.sort_values(experiment, ascending=True)
        .drop_duplicates(subset=["Model"], keep="first")
    )

    return df[["Model", experiment]]


def load_lag_comparison() -> pd.DataFrame:
    without_df = load_metrics_for_experiment("without_lag")
    with_df = load_metrics_for_experiment("with_lag")

    df = without_df.merge(with_df, on="Model", how="inner")

    df["Order"] = df["Model"].apply(
        lambda model: MODEL_ORDER.index(model) if model in MODEL_ORDER else 999
    )

    df = df.sort_values("Order").drop(columns="Order")

    return df


# ============================================================
# Plot
# ============================================================

def main() -> None:
    df = load_lag_comparison()

    print("\nLoaded test RMSE comparison:")
    print(df.to_string(index=False))

    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12.5, 6.4))

    without_bars = ax.bar(
        x - width / 2,
        df["without_lag"],
        width,
        label="Without lag",
        color=MID_GREY,
    )

    with_bars = ax.bar(
        x + width / 2,
        df["with_lag"],
        width,
        label="With lag",
        color=GREEN,
    )

    ax.set_title(
        "Test RMSE Comparison With and Without Lag Features",
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

    ymax = max(df["without_lag"].max(), df["with_lag"].max()) * 1.22
    ax.set_ylim(0, ymax)

    # Value labels
    for bars in [without_bars, with_bars]:
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

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GREY,
        fontsize=11,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / "lag_vs_without_lag_test_rmse.png"
    pdf_path = OUTPUT_DIR / "lag_vs_without_lag_test_rmse.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved plots:")
    print(f"- {png_path.relative_to(ROOT)}")
    print(f"- {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()