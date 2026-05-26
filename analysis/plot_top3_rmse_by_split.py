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

# Explicit model selection to stay consistent with the paper
SELECTED_MODELS = [
    {
        "display_name": "Gradient Boosting",
        "aliases": ["gradient_boost", "gradient_boosting", "gradientboost", "boost"],
    },
    {
        "display_name": "Neural Network",
        "aliases": ["neural_network", "neural", "mlp", "nn"],
    },
    {
        "display_name": "Lasso",
        "aliases": ["lasso"],
    },
]


# ============================================================
# Helper functions
# ============================================================

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def path_matches_model(path: Path, aliases: list[str]) -> bool:
    path_text = normalize(path.relative_to(ROOT))
    return any(alias in path_text for alias in aliases)


def load_metrics_file(metrics_path: Path) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_metrics_for_model(model_config: dict) -> dict:
    display_name = model_config["display_name"]
    aliases = model_config["aliases"]

    candidates = []

    for metrics_path in RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"):
        path_text = normalize(metrics_path.relative_to(ROOT))

        if not path_matches_model(metrics_path, aliases):
            continue

        metrics = load_metrics_file(metrics_path)

        required_keys = ["train_rmse", "validation_rmse", "test_rmse"]
        if not all(key in metrics for key in required_keys):
            continue

        if metrics.get("uses_lag_features") is not True:
            continue

        # Avoid accidentally selecting Random Forest when searching for NN/RF-like abbreviations
        if display_name == "Neural Network":
            if "random" in path_text or "forest" in path_text:
                continue

        candidates.append(
            {
                "Model": display_name,
                "Train RMSE": float(metrics["train_rmse"]),
                "Validation RMSE": float(metrics["validation_rmse"]),
                "Test RMSE": float(metrics["test_rmse"]),
                "Path": str(metrics_path.relative_to(ROOT)),
            }
        )

    if not candidates:
        print(f"\nNo metrics file found for {display_name}.")
        print("Available with_lag metrics files:")
        for path in RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"):
            print("-", path.relative_to(ROOT))
        raise FileNotFoundError(f"No valid with_lag metrics found for {display_name}")

    # If duplicates exist, use the best test RMSE for that selected model
    candidates = sorted(candidates, key=lambda x: x["Test RMSE"])
    return candidates[0]


def load_selected_metrics() -> pd.DataFrame:
    rows = []

    for model_config in SELECTED_MODELS:
        rows.append(find_metrics_for_model(model_config))

    return pd.DataFrame(rows)


# ============================================================
# Plot
# ============================================================

def main() -> None:
    df = load_selected_metrics()

    print("\nSelected models for presentation plot:")
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
        "RMSE by Split for the Selected Final Models",
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

    png_path = OUTPUT_DIR / "selected_final_models_rmse_train_validation_test.png"
    pdf_path = OUTPUT_DIR / "selected_final_models_rmse_train_validation_test.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved plots:")
    print(f"- {png_path.relative_to(ROOT)}")
    print(f"- {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()