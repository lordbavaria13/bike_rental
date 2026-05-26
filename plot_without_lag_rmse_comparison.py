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
OUTPUT_DIR = ROOT / "presentation_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_ROOT = ROOT / "modelling"

# Change this to "with_lag" if you later want the same plot after lag features
EXPERIMENT = "without_lag"

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
# Helpers
# ============================================================

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def display_name_from_metrics(metrics: dict, path: Path) -> str:
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


def load_rmse_metrics() -> pd.DataFrame:
    rows = []

    metrics_files = sorted(RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"))

    if not metrics_files:
        raise FileNotFoundError(
            f"No {EXPERIMENT}/metrics.json files found under modelling/."
        )

    for metrics_path in metrics_files:
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        required = ["train_rmse", "validation_rmse", "test_rmse"]
        if not all(key in metrics for key in required):
            continue

        if EXPERIMENT == "without_lag" and metrics.get("uses_lag_features") is True:
            continue

        if EXPERIMENT == "with_lag" and metrics.get("uses_lag_features") is False:
            continue

        model_label = display_name_from_metrics(metrics, metrics_path)

        # Skip Dummy because the slide compares real models
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
        raise ValueError(f"No valid {EXPERIMENT} model metrics found.")

    df = pd.DataFrame(rows)

    # If duplicates exist, keep the model with the lowest test RMSE
    df = (
        df.sort_values("Test RMSE", ascending=True)
        .drop_duplicates(subset=["Model"], keep="first")
    )

    df["Order"] = df["Model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999
    )

    df = df.sort_values(["Order", "Test RMSE"]).drop(columns="Order")

    return df


# ============================================================
# Plot
# ============================================================

def main() -> None:
    df = load_rmse_metrics()

    print(f"\nLoaded {EXPERIMENT} RMSE results:")
    print(
        df[
            ["Model", "Train RMSE", "Validation RMSE", "Test RMSE", "Path"]
        ].to_string(index=False)
    )

    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12.5, 6.4))

    train_bars = ax.bar(
        x - width,
        df["Train RMSE"],
        width,
        label="Train",
        color=VERY_LIGHT_GREY,
        edgecolor=LIGHT_GREY,
        linewidth=0.8,
    )

    val_bars = ax.bar(
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

    title_suffix = "Without Lag Features" if EXPERIMENT == "without_lag" else "With Lag Features"

    ax.set_title(
        f"RMSE by Split: {title_suffix}",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=18,
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
    ) * 1.18

    ax.set_ylim(0, ymax)

    # Label only the Test bars to avoid visual overload
    for bar, value in zip(test_bars, df["Test RMSE"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=DARK_GREY,
            fontweight="bold",
        )

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GREY,
        fontsize=11,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{EXPERIMENT}_rmse_train_validation_test.png"
    pdf_path = OUTPUT_DIR / f"{EXPERIMENT}_rmse_train_validation_test.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved plots:")
    print(f"- {png_path.relative_to(ROOT)}")
    print(f"- {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()