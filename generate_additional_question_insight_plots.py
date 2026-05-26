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

OUTPUT_DIR = ROOT / "presentation_figures" / "additional_question_insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "with_lag"

TARGET_COL = "total_rentals"
PRED_COL = "prediction"
SPLIT_COL = "split"

SELECTED_MODELS = [
    {
        "key": "GB",
        "label": "Gradient Boosting",
        "aliases": ["gradient_boost", "gradient_boosting", "gradientboost", "boost"],
    },
    {
        "key": "NN",
        "label": "Neural Network",
        "aliases": ["neural_network", "neural", "mlp"],
    },
    {
        "key": "Lasso",
        "label": "Lasso",
        "aliases": ["lasso"],
    },
]

MODEL_COLORS = {
    "Gradient Boosting": GREEN,
    "Neural Network": MID_GREY,
    "Lasso": VERY_LIGHT_GREY,
}

MODEL_EDGE_COLORS = {
    "Gradient Boosting": GREEN,
    "Neural Network": MID_GREY,
    "Lasso": LIGHT_GREY,
}


# ============================================================
# General helpers
# ============================================================

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_plot(fig, filename_base: str) -> None:
    png_path = OUTPUT_DIR / f"{filename_base}.png"
    pdf_path = OUTPUT_DIR / f"{filename_base}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path.relative_to(ROOT)}")
    print(f"Saved: {pdf_path.relative_to(ROOT)}")


def style_axis(ax, grid_axis="y") -> None:
    ax.grid(axis=grid_axis, color=LIGHT_GREY, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors=DARK_GREY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)


def path_matches_alias(path: Path, aliases: list[str]) -> bool:
    path_text = normalize(path.relative_to(ROOT))
    return any(alias in path_text for alias in aliases)


# ============================================================
# Locate selected model result folders
# ============================================================

def find_result_dir(model_config: dict) -> Path:
    key = model_config["key"]
    label = model_config["label"]
    aliases = model_config["aliases"]

    candidates = []

    for metrics_path in RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"):
        path_text = normalize(metrics_path.relative_to(ROOT))

        if not path_matches_alias(metrics_path, aliases):
            continue

        # Prevent matching KNN or Random Forest when searching for Neural Network
        if key == "NN":
            if "knn" in path_text or "random" in path_text or "forest" in path_text:
                continue

        result_dir = metrics_path.parent
        predictions_path = result_dir / "predictions.csv"

        if not predictions_path.exists():
            continue

        metrics = load_json(metrics_path)

        if metrics.get("uses_lag_features") is not True:
            continue

        if "test_rmse" not in metrics:
            continue

        candidates.append(
            {
                "result_dir": result_dir,
                "metrics": metrics,
                "test_rmse": float(metrics["test_rmse"]),
            }
        )

    if not candidates:
        print(f"\nNo valid result directory found for {label}.")
        print("Available with_lag metrics files:")
        for path in RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"):
            print("-", path.relative_to(ROOT))
        raise FileNotFoundError(f"No valid result directory found for {label}")

    # If duplicates exist, choose the best test RMSE for that model
    candidates = sorted(candidates, key=lambda item: item["test_rmse"])
    return candidates[0]["result_dir"]


def load_selected_models() -> list[dict]:
    selected = []

    for model_config in SELECTED_MODELS:
        result_dir = find_result_dir(model_config)

        metrics = load_json(result_dir / "metrics.json")
        predictions = load_test_predictions(result_dir / "predictions.csv")

        selected.append(
            {
                "key": model_config["key"],
                "label": model_config["label"],
                "result_dir": result_dir,
                "metrics": metrics,
                "predictions": predictions,
            }
        )

        print(f"\n{model_config['label']}")
        print(f"Result dir: {result_dir.relative_to(ROOT)}")
        print(
            f"Test RMSE={metrics['test_rmse']:.2f}, "
            f"MAE={metrics['test_mae']:.2f}, "
            f"R²={metrics['test_r2']:.3f}"
        )

    return selected


# ============================================================
# Predictions and residual helpers
# ============================================================

def load_test_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = {TARGET_COL, PRED_COL}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    if SPLIT_COL in df.columns:
        df = df[df[SPLIT_COL].astype(str).str.lower() == "test"].copy()
    else:
        print(f"WARNING: No split column in {path}; using all rows.")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL, PRED_COL])

    if df.empty:
        raise ValueError(f"No valid prediction rows found in {path}")

    df["residual"] = df[PRED_COL] - df[TARGET_COL]
    df["abs_error"] = np.abs(df["residual"])

    return df


def add_demand_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, np.inf]
    labels = ["0–25", "25–50", "50–75", "75–100", "100–125", "125–150", "150–175", "175–200", "200+"]

    out = df.copy()
    out["demand_bin"] = pd.cut(
        out[TARGET_COL],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    return out


# ============================================================
# Feature score / importance helpers
# ============================================================

def detect_feature_score_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str]:
    normalized = {normalize(col): col for col in df.columns}

    feature_candidates = [
        "feature",
        "feature_name",
        "variable",
        "name",
        "predictor",
    ]

    score_candidates = [
        "importance",
        "feature_importance",
        "permutation_importance",
        "permutation_importance_mean",
        "mean_importance",
        "abs_coefficient",
        "absolute_coefficient",
        "coefficient",
        "coef",
        "value",
    ]

    feature_col = None
    score_col = None

    for candidate in feature_candidates:
        if candidate in normalized:
            feature_col = normalized[candidate]
            break

    for candidate in score_candidates:
        if candidate in normalized:
            score_col = normalized[candidate]
            break

    if score_col is None or feature_col is None:
        return None, None, "unknown"

    score_col_norm = normalize(score_col)

    if "coef" in score_col_norm or "coefficient" in score_col_norm:
        score_type = "absolute coefficient"
    elif "permutation" in score_col_norm:
        score_type = "permutation importance"
    else:
        score_type = "feature importance"

    return feature_col, score_col, score_type


def find_feature_score_file(result_dir: Path) -> Path | None:
    preferred_names = [
        "feature_importance.csv",
        "permutation_importance.csv",
        "feature_importances.csv",
        "coefficients.csv",
        "model_coefficients.csv",
        "lasso_coefficients.csv",
    ]

    for name in preferred_names:
        candidate = result_dir / name
        if candidate.exists():
            return candidate

    csv_candidates = []

    for path in result_dir.rglob("*.csv"):
        name = normalize(path.name)

        if "prediction" in name or "metric" in name or "hyperparameter" in name:
            continue

        if (
            "importance" in name
            or "coefficient" in name
            or "coef" in name
            or "permutation" in name
        ):
            csv_candidates.append(path)

    if not csv_candidates:
        return None

    return sorted(csv_candidates)[0]


def load_feature_scores(model_item: dict) -> pd.DataFrame | None:
    result_dir = model_item["result_dir"]
    feature_file = find_feature_score_file(result_dir)

    if feature_file is None:
        print(
            f"Skipped feature score plot for {model_item['label']}: "
            "no feature_importance / coefficients / permutation_importance file found."
        )
        return None

    df = pd.read_csv(feature_file)
    feature_col, score_col, score_type = detect_feature_score_columns(df)

    if feature_col is None or score_col is None:
        print(
            f"Skipped feature score plot for {model_item['label']}: "
            f"could not detect feature/score columns in {feature_file.relative_to(ROOT)}"
        )
        print(f"Columns found: {list(df.columns)}")
        return None

    scores = pd.to_numeric(df[score_col], errors="coerce")

    out = pd.DataFrame(
        {
            "feature": df[feature_col].astype(str),
            "score": scores,
        }
    ).dropna()

    out = out[~out["feature"].str.lower().isin(["intercept", "bias", "const"])]

    if out.empty:
        print(f"Skipped feature score plot for {model_item['label']}: no valid scores.")
        return None

    # Coefficients can be negative, use magnitude for ranking
    out["score_abs"] = out["score"].abs()

    out = out.sort_values("score_abs", ascending=False).head(15)
    out = out.sort_values("score_abs", ascending=True)

    out["model"] = model_item["label"]
    out["score_type"] = score_type
    out["source_file"] = str(feature_file.relative_to(ROOT))

    print(
        f"Loaded feature scores for {model_item['label']} "
        f"from {feature_file.relative_to(ROOT)} as {score_type}."
    )

    return out


# ============================================================
# Plot 1: feature scores for selected final models
# ============================================================

def plot_feature_scores_selected_models(selected_models: list[dict]) -> None:
    score_tables = []

    for model_item in selected_models:
        scores = load_feature_scores(model_item)
        if scores is not None:
            score_tables.append(scores)

    if not score_tables:
        print("No feature score files found for selected models. Skipping feature score plot.")
        return

    available_models = [df["model"].iloc[0] for df in score_tables]
    n_models = len(score_tables)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(6.2 * n_models, 7.0),
        squeeze=False,
    )

    axes = axes[0]

    for ax, scores in zip(axes, score_tables):
        model = scores["model"].iloc[0]
        score_type = scores["score_type"].iloc[0]

        ax.barh(
            scores["feature"],
            scores["score_abs"],
            color=MODEL_COLORS.get(model, GREEN),
            edgecolor=MODEL_EDGE_COLORS.get(model, GREEN),
        )

        ax.set_title(
            f"{model}\nTop feature scores",
            fontsize=14,
            fontweight="bold",
            color=DARK_GREY,
            pad=12,
        )

        ax.set_xlabel(score_type.title(), fontsize=11, color=DARK_GREY)
        ax.set_ylabel("Feature", fontsize=11, color=DARK_GREY)

        style_axis(ax, grid_axis="x")

    fig.suptitle(
        "Top Feature Scores for Selected Final Models",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        y=1.03,
    )

    fig.tight_layout()

    save_plot(fig, "01_new_feature_scores_selected_final_models")

    combined = pd.concat(score_tables, ignore_index=True)
    csv_path = OUTPUT_DIR / "01_new_feature_scores_selected_final_models.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")


# ============================================================
# Plot 2: MAE by actual demand bin
# ============================================================

def plot_mae_by_demand_bin(selected_models: list[dict]) -> None:
    rows = []

    for model_item in selected_models:
        df = add_demand_bins(model_item["predictions"])

        grouped = (
            df.groupby("demand_bin", observed=False)
            .agg(
                mae=("abs_error", "mean"),
                n=(TARGET_COL, "size"),
            )
            .reset_index()
        )

        grouped["model"] = model_item["label"]
        rows.append(grouped)

    result = pd.concat(rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "02_new_mae_by_actual_demand_bin.csv"
    result.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")

    pivot = result.pivot(index="demand_bin", columns="model", values="mae")
    pivot = pivot.dropna(how="all")

    x = np.arange(len(pivot.index))
    width = 0.24

    fig, ax = plt.subplots(figsize=(13.0, 6.4))

    model_labels = [item["label"] for item in selected_models]
    offsets = np.linspace(-width, width, len(model_labels))

    for offset, model in zip(offsets, model_labels):
        if model not in pivot.columns:
            continue

        ax.bar(
            x + offset,
            pivot[model],
            width,
            label=model,
            color=MODEL_COLORS.get(model, GREEN),
            edgecolor=MODEL_EDGE_COLORS.get(model, None),
            linewidth=0.8,
        )

    ax.set_title(
        "Mean Absolute Error by Actual Demand Level",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=42,
    )

    ax.set_xlabel("Actual rentals bin", fontsize=12, color=DARK_GREY)
    ax.set_ylabel("MAE", fontsize=12, color=DARK_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in pivot.index], rotation=0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GREY,
        fontsize=11,
    )

    style_axis(ax)
    fig.tight_layout()
    save_plot(fig, "02_new_mae_by_actual_demand_bin")


# ============================================================
# Plot 3: prediction bias by demand bin
# ============================================================

def plot_prediction_bias_by_demand_bin(selected_models: list[dict]) -> None:
    rows = []

    for model_item in selected_models:
        df = add_demand_bins(model_item["predictions"])

        grouped = (
            df.groupby("demand_bin", observed=False)
            .agg(
                mean_residual=("residual", "mean"),
                n=(TARGET_COL, "size"),
            )
            .reset_index()
        )

        grouped["model"] = model_item["label"]
        rows.append(grouped)

    result = pd.concat(rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "03_new_prediction_bias_by_actual_demand_bin.csv"
    result.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")

    pivot = result.pivot(index="demand_bin", columns="model", values="mean_residual")
    pivot = pivot.dropna(how="all")

    x = np.arange(len(pivot.index))

    fig, ax = plt.subplots(figsize=(13.0, 6.4))

    for model_item in selected_models:
        model = model_item["label"]

        if model not in pivot.columns:
            continue

        ax.plot(
            x,
            pivot[model],
            marker="o",
            linewidth=2.0,
            label=model,
            color=MODEL_COLORS.get(model, GREEN),
        )

    ax.axhline(
        0,
        color=DARK_GREY,
        linewidth=1.2,
        linestyle="--",
    )

    ax.set_title(
        "Prediction Bias by Actual Demand Level",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=42,
    )

    ax.set_xlabel("Actual rentals bin", fontsize=12, color=DARK_GREY)
    ax.set_ylabel("Mean residual: predicted − actual", fontsize=12, color=DARK_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels([str(label) for label in pivot.index], rotation=0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GREY,
        fontsize=11,
    )

    style_axis(ax)
    fig.tight_layout()
    save_plot(fig, "03_new_prediction_bias_by_actual_demand_bin")


# ============================================================
# Plot 4: residual distributions
# ============================================================

def plot_residual_distributions(selected_models: list[dict]) -> None:
    fig, axes = plt.subplots(
        1,
        len(selected_models),
        figsize=(5.6 * len(selected_models), 4.8),
        sharey=True,
    )

    if len(selected_models) == 1:
        axes = [axes]

    all_residuals = np.concatenate(
        [item["predictions"]["residual"].to_numpy() for item in selected_models]
    )

    max_abs = np.nanpercentile(np.abs(all_residuals), 99)
    bins = np.linspace(-max_abs, max_abs, 40)

    for ax, model_item in zip(axes, selected_models):
        model = model_item["label"]
        residuals = model_item["predictions"]["residual"].to_numpy()

        ax.hist(
            residuals,
            bins=bins,
            color=MODEL_COLORS.get(model, GREEN),
            edgecolor="white",
            alpha=0.85,
        )

        ax.axvline(
            0,
            color=DARK_GREY,
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_title(
            model,
            fontsize=14,
            fontweight="bold",
            color=DARK_GREY,
            pad=10,
        )

        ax.set_xlabel("Residual: predicted − actual", fontsize=11, color=DARK_GREY)
        style_axis(ax)

    axes[0].set_ylabel("Frequency", fontsize=11, color=DARK_GREY)

    fig.suptitle(
        "Residual Distributions on Test Data",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        y=1.04,
    )

    fig.tight_layout()
    save_plot(fig, "04_new_residual_distributions_selected_models")


# ============================================================
# Plot 5: compact error summary table
# ============================================================

def plot_error_summary_table(selected_models: list[dict]) -> None:
    rows = []

    for model_item in selected_models:
        df = model_item["predictions"]
        metrics = model_item["metrics"]

        high_demand = df[df[TARGET_COL] >= 150]

        rows.append(
            {
                "Model": model_item["label"],
                "RMSE": f"{metrics['test_rmse']:.2f}",
                "MAE": f"{metrics['test_mae']:.2f}",
                "R²": f"{metrics['test_r2']:.3f}",
                "MAPE": f"{metrics['test_mape'] * 100:.1f}%",
                "Mean bias": f"{df['residual'].mean():.2f}",
                "90% abs. error": f"{df['abs_error'].quantile(0.90):.2f}",
                "MAE ≥150 rentals": (
                    f"{high_demand['abs_error'].mean():.2f}"
                    if not high_demand.empty
                    else "n/a"
                ),
            }
        )

    summary = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "05_new_selected_model_error_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")

    fig, ax = plt.subplots(figsize=(13.5, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(LIGHT_GREY)
        cell.set_linewidth(0.8)

        if row == 0:
            cell.set_facecolor(GREEN)
            cell.set_text_props(weight="bold", color=DARK_GREY)
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=DARK_GREY)

    fig.suptitle(
        "Additional Test Error Summary for Selected Models",
        fontsize=17,
        fontweight="bold",
        color=DARK_GREY,
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_plot(fig, "05_new_selected_model_error_summary_table")


# ============================================================
# Main
# ============================================================

def main() -> None:
    selected_models = load_selected_models()

    print("\nCreating only additional / new insight graphics...")
    print(f"Output folder: {OUTPUT_DIR.relative_to(ROOT)}")

    plot_feature_scores_selected_models(selected_models)
    plot_mae_by_demand_bin(selected_models)
    plot_prediction_bias_by_demand_bin(selected_models)
    plot_residual_distributions(selected_models)
    plot_error_summary_table(selected_models)

    print("\nDone.")
    print("Created only additional question-support graphics, no existing presentation plots.")


if __name__ == "__main__":
    main()