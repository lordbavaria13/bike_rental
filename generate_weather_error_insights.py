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
DATA_ROOT = ROOT / "data" / "processed" / "with_lag"

OUTPUT_DIR = ROOT / "presentation_figures" / "weather_question_insights"
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

WEATHER_COLUMNS = [
    "tempmax",
    "humidity",
    "precip",
    "precipcover",
    "cloudcover",
    "windspeed",
    "visibility",
    "sealevelpressure",
    "uvindex",
    "snow",
    "snowdepth",
]


# ============================================================
# Basic helpers
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


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=LIGHT_GREY, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors=DARK_GREY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)


# ============================================================
# Locate model result folders
# ============================================================

def path_matches_alias(path: Path, aliases: list[str]) -> bool:
    path_text = normalize(path.relative_to(ROOT))
    return any(alias in path_text for alias in aliases)


def find_result_dir(model_config: dict) -> Path:
    label = model_config["label"]
    aliases = model_config["aliases"]

    candidates = []

    for metrics_path in RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"):
        path_text = normalize(metrics_path.relative_to(ROOT))

        if not path_matches_alias(metrics_path, aliases):
            continue

        # Avoid accidental matches like KNN when searching Neural Network
        if label == "Neural Network":
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

    candidates = sorted(candidates, key=lambda item: item["test_rmse"])
    return candidates[0]["result_dir"]


# ============================================================
# Load context/weather data
# ============================================================

def load_context_data() -> pd.DataFrame:
    split_files = [
        DATA_ROOT / "encoded_train.csv",
        DATA_ROOT / "encoded_validation.csv",
        DATA_ROOT / "encoded_test.csv",
    ]

    available_split_files = [path for path in split_files if path.exists()]

    if available_split_files:
        frames = []
        for path in available_split_files:
            df = pd.read_csv(path)
            frames.append(df)
        context = pd.concat(frames, ignore_index=True)
    else:
        fallback = DATA_ROOT / "daily_rentals_top20_reduced_with_lag.csv"

        if not fallback.exists():
            raise FileNotFoundError(
                "Could not find encoded split files or daily_rentals_top20_reduced_with_lag.csv"
            )

        context = pd.read_csv(fallback)

    if "start_station_id_raw" in context.columns:
        context["start_station_id"] = context["start_station_id_raw"]

    required = {"time_idx", "start_station_id", TARGET_COL}
    missing = required - set(context.columns)

    if missing:
        raise ValueError(
            f"Context data is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(context.columns)}"
        )

    present_weather_cols = [col for col in WEATHER_COLUMNS if col in context.columns]

    if not present_weather_cols:
        raise ValueError(
            "No weather columns found in context data. "
            "Expected columns such as tempmax, precip, humidity, windspeed."
        )

    context["time_idx"] = pd.to_numeric(context["time_idx"], errors="coerce")
    context["start_station_id"] = pd.to_numeric(context["start_station_id"], errors="coerce")
    context[TARGET_COL] = pd.to_numeric(context[TARGET_COL], errors="coerce")

    for col in present_weather_cols:
        context[col] = pd.to_numeric(context[col], errors="coerce")

    return context


# ============================================================
# Load predictions and merge with context
# ============================================================

def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"time_idx", "start_station_id", TARGET_COL, PRED_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file is missing required columns: {sorted(missing)}\n"
            f"File: {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    if SPLIT_COL in df.columns:
        df = df[df[SPLIT_COL].astype(str).str.lower() == "test"].copy()

    df["time_idx"] = pd.to_numeric(df["time_idx"], errors="coerce")
    df["start_station_id"] = pd.to_numeric(df["start_station_id"], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors="coerce")

    df = df.dropna(subset=["time_idx", "start_station_id", TARGET_COL, PRED_COL])

    df["residual"] = df[PRED_COL] - df[TARGET_COL]
    df["abs_error"] = df["residual"].abs()

    return df


def merge_predictions_with_context(pred: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    present_weather_cols = [col for col in WEATHER_COLUMNS if col in context.columns]

    key_candidates = [
        ["time_idx", "start_station_id", TARGET_COL, SPLIT_COL],
        ["time_idx", "start_station_id", SPLIT_COL],
        ["time_idx", "start_station_id", TARGET_COL],
        ["time_idx", "start_station_id"],
    ]

    best_merge = None
    best_coverage = -1
    best_keys = None

    for keys in key_candidates:
        keys = [key for key in keys if key in pred.columns and key in context.columns]

        if not keys:
            continue

        context_cols = keys + present_weather_cols
        context_unique = context[context_cols].drop_duplicates(subset=keys)

        merged = pred.merge(
            context_unique,
            on=keys,
            how="left",
            suffixes=("", "_context"),
        )

        coverage = merged[present_weather_cols].notna().mean().mean()

        if coverage > best_coverage:
            best_coverage = coverage
            best_merge = merged
            best_keys = keys

    if best_merge is None:
        raise ValueError("Could not merge predictions with context data.")

    print(f"Merge keys used: {best_keys}")
    print(f"Weather coverage after merge: {best_coverage:.1%}")

    if best_coverage < 0.95:
        print("WARNING: Weather merge coverage is below 95%. Check keys and processed data.")

    return best_merge


def add_weather_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    precip = out["precip"] if "precip" in out.columns else 0
    precipcover = out["precipcover"] if "precipcover" in out.columns else 0
    windspeed = out["windspeed"] if "windspeed" in out.columns else 0
    tempmax = out["tempmax"] if "tempmax" in out.columns else np.nan
    cloudcover = out["cloudcover"] if "cloudcover" in out.columns else 0
    snow = out["snow"] if "snow" in out.columns else 0
    snowdepth = out["snowdepth"] if "snowdepth" in out.columns else 0

    # Coarse operational weather categories.
    # They are not labels from the dataset, but diagnostic bins for analysis.
    difficult_weather = (
        (precip > 5)
        | (precipcover >= 20)
        | (windspeed >= 30)
        | (tempmax < 5)
        | (tempmax > 35)
        | (snow > 0)
        | (snowdepth > 0)
    )

    favorable_weather = (
        (precip == 0)
        & (precipcover == 0)
        & (windspeed < 20)
        & (tempmax >= 10)
        & (tempmax <= 30)
        & (cloudcover < 75)
        & (snow == 0)
        & (snowdepth == 0)
    )

    out["weather_condition"] = np.select(
        [favorable_weather, difficult_weather],
        ["Favorable", "Difficult"],
        default="Mixed",
    )

    out["weather_condition"] = pd.Categorical(
        out["weather_condition"],
        categories=["Favorable", "Mixed", "Difficult"],
        ordered=True,
    )

    if "tempmax" in out.columns:
        out["temperature_bin"] = pd.cut(
            out["tempmax"],
            bins=[-np.inf, 5, 10, 15, 20, 25, 30, 35, np.inf],
            labels=["<5", "5–10", "10–15", "15–20", "20–25", "25–30", "30–35", "35+"],
        )

    if "precip" in out.columns:
        out["precipitation_bin"] = pd.cut(
            out["precip"],
            bins=[-0.001, 0, 2, 8, np.inf],
            labels=["No rain", "Light", "Medium", "Heavy"],
            include_lowest=True,
        )

    if "windspeed" in out.columns:
        out["wind_bin"] = pd.cut(
            out["windspeed"],
            bins=[-np.inf, 15, 25, 35, np.inf],
            labels=["Low", "Moderate", "Windy", "Very windy"],
        )

    if "humidity" in out.columns:
        out["humidity_bin"] = pd.cut(
            out["humidity"],
            bins=[-np.inf, 50, 70, 85, np.inf],
            labels=["Low", "Medium", "High", "Very high"],
        )

    return out


def load_selected_model_data() -> list[dict]:
    context = load_context_data()
    selected = []

    for model_config in SELECTED_MODELS:
        result_dir = find_result_dir(model_config)
        metrics = load_json(result_dir / "metrics.json")
        predictions = load_predictions(result_dir / "predictions.csv")
        merged = merge_predictions_with_context(predictions, context)
        merged = add_weather_categories(merged)
        merged["model"] = model_config["label"]

        selected.append(
            {
                "key": model_config["key"],
                "label": model_config["label"],
                "result_dir": result_dir,
                "metrics": metrics,
                "data": merged,
            }
        )

        print(f"\n{model_config['label']}")
        print(f"Result dir: {result_dir.relative_to(ROOT)}")
        print(
            f"Test RMSE={metrics['test_rmse']:.2f}, "
            f"MAE={metrics['test_mae']:.2f}, "
            f"R²={metrics['test_r2']:.3f}"
        )

    combined = pd.concat([item["data"] for item in selected], ignore_index=True)
    combined_path = OUTPUT_DIR / "weather_enriched_test_predictions_selected_models.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved enriched prediction data: {combined_path.relative_to(ROOT)}")

    return selected


# ============================================================
# Aggregation helpers
# ============================================================

def aggregate_by_category(selected_models: list[dict], category_col: str) -> pd.DataFrame:
    rows = []

    for item in selected_models:
        df = item["data"].copy()

        if category_col not in df.columns:
            continue

        grouped = (
            df.groupby(category_col, observed=False)
            .agg(
                mae=("abs_error", "mean"),
                rmse=("residual", lambda x: np.sqrt(np.mean(np.square(x)))),
                mean_residual=("residual", "mean"),
                mean_actual=(TARGET_COL, "mean"),
                mean_prediction=(PRED_COL, "mean"),
                n=(TARGET_COL, "size"),
            )
            .reset_index()
        )

        grouped["model"] = item["label"]
        rows.append(grouped)

    if not rows:
        raise ValueError(f"No data available for category column: {category_col}")

    result = pd.concat(rows, ignore_index=True)

    csv_path = OUTPUT_DIR / f"{category_col}_weather_error_summary.csv"
    result.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")

    return result


# ============================================================
# Plot 1: MAE by overall weather condition
# ============================================================

def plot_mae_by_weather_condition(selected_models: list[dict]) -> None:
    result = aggregate_by_category(selected_models, "weather_condition")

    pivot = result.pivot(index="weather_condition", columns="model", values="mae")
    pivot = pivot.loc[["Favorable", "Mixed", "Difficult"]]

    x = np.arange(len(pivot.index))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 6.2))

    for offset, model in zip([-width, 0, width], [item["label"] for item in selected_models]):
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
        "MAE by Weather Condition",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=42,
    )

    ax.set_xlabel("Weather condition", fontsize=12, color=DARK_GREY)
    ax.set_ylabel("MAE", fontsize=12, color=DARK_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)

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
    save_plot(fig, "01_mae_by_weather_condition")


# ============================================================
# Plot 2: Prediction bias by weather condition
# ============================================================

def plot_bias_by_weather_condition(selected_models: list[dict]) -> None:
    result = aggregate_by_category(selected_models, "weather_condition")

    pivot = result.pivot(index="weather_condition", columns="model", values="mean_residual")
    pivot = pivot.loc[["Favorable", "Mixed", "Difficult"]]

    x = np.arange(len(pivot.index))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.8, 6.2))

    for offset, model in zip([-width, 0, width], [item["label"] for item in selected_models]):
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

    ax.axhline(0, color=DARK_GREY, linestyle="--", linewidth=1.2)

    ax.set_title(
        "Prediction Bias by Weather Condition",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=42,
    )

    ax.set_xlabel("Weather condition", fontsize=12, color=DARK_GREY)
    ax.set_ylabel("Mean residual: predicted − actual", fontsize=12, color=DARK_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)

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
    save_plot(fig, "02_prediction_bias_by_weather_condition")


# ============================================================
# Plot 3: Mean actual vs predicted demand by weather condition
# ============================================================

def plot_actual_vs_predicted_by_weather(selected_models: list[dict]) -> None:
    result = aggregate_by_category(selected_models, "weather_condition")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)

    for ax, item in zip(axes, selected_models):
        model = item["label"]
        df = result[result["model"] == model].copy()
        df = df.set_index("weather_condition").loc[["Favorable", "Mixed", "Difficult"]].reset_index()

        x = np.arange(len(df))
        width = 0.34

        ax.bar(
            x - width / 2,
            df["mean_actual"],
            width,
            label="Actual",
            color=MID_GREY,
        )

        ax.bar(
            x + width / 2,
            df["mean_prediction"],
            width,
            label="Predicted",
            color=GREEN,
        )

        ax.set_title(
            model,
            fontsize=14,
            fontweight="bold",
            color=DARK_GREY,
            pad=10,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(df["weather_condition"])
        ax.set_xlabel("Weather condition", fontsize=11, color=DARK_GREY)
        style_axis(ax)

    axes[0].set_ylabel("Mean rentals", fontsize=11, color=DARK_GREY)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GREY,
        fontsize=11,
    )

    fig.suptitle(
        "Mean Actual vs. Predicted Demand by Weather Condition",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        y=1.15,
    )

    fig.tight_layout()
    save_plot(fig, "03_actual_vs_predicted_by_weather_condition")


# ============================================================
# Generic category MAE plots
# ============================================================

def plot_mae_by_category(selected_models: list[dict], category_col: str, title: str, filename: str) -> None:
    result = aggregate_by_category(selected_models, category_col)

    pivot = result.pivot(index=category_col, columns="model", values="mae")
    pivot = pivot.dropna(how="all")

    x = np.arange(len(pivot.index))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    for offset, model in zip([-width, 0, width], [item["label"] for item in selected_models]):
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
        title,
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=42,
    )

    ax.set_xlabel(category_col.replace("_", " ").title(), fontsize=12, color=DARK_GREY)
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
    save_plot(fig, filename)


# ============================================================
# Plot 7: weather-error correlation diagnostics
# ============================================================

def plot_weather_error_correlations(selected_models: list[dict]) -> None:
    present_weather_cols = []

    for col in WEATHER_COLUMNS:
        if any(col in item["data"].columns for item in selected_models):
            present_weather_cols.append(col)

    if not present_weather_cols:
        print("No weather columns available for correlation diagnostics.")
        return

    rows = []

    for item in selected_models:
        df = item["data"].copy()

        for col in present_weather_cols:
            if col not in df.columns:
                continue

            pair = df[[col, "abs_error"]].dropna()

            if len(pair) < 10:
                continue

            corr = pair[col].corr(pair["abs_error"], method="spearman")

            if pd.isna(corr):
                continue

            rows.append(
                {
                    "model": item["label"],
                    "weather_variable": col,
                    "spearman_corr_with_abs_error": corr,
                    "abs_corr": abs(corr),
                }
            )

    if not rows:
        print("No valid weather-error correlations could be computed.")
        return

    corr_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "weather_error_correlations.csv"
    corr_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8), sharex=True)

    for ax, item in zip(axes, selected_models):
        model = item["label"]
        df = corr_df[corr_df["model"] == model].copy()
        df = df.sort_values("abs_corr", ascending=False).head(8)
        df = df.sort_values("abs_corr", ascending=True)

        ax.barh(
            df["weather_variable"],
            df["abs_corr"],
            color=MODEL_COLORS.get(model, GREEN),
            edgecolor=MODEL_EDGE_COLORS.get(model, None),
        )

        ax.set_title(
            model,
            fontsize=14,
            fontweight="bold",
            color=DARK_GREY,
            pad=10,
        )

        ax.set_xlabel("|Spearman correlation|", fontsize=11, color=DARK_GREY)
        ax.set_ylabel("Weather variable", fontsize=11, color=DARK_GREY)

        style_axis(ax, grid_axis="x")

    fig.suptitle(
        "Weather Variables Most Associated with Prediction Error",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        y=1.04,
    )

    fig.tight_layout()
    save_plot(fig, "07_weather_error_correlation_diagnostics")


# ============================================================
# Create markdown report with automatically derived insights
# ============================================================

def write_weather_insight_report(selected_models: list[dict]) -> None:
    summary = aggregate_by_category(selected_models, "weather_condition")

    lines = []
    lines.append("# Weather Error Insight Report\n")
    lines.append("This report is generated from test predictions merged with processed weather features.\n")
    lines.append("Weather categories are diagnostic bins, not original dataset labels.\n")

    for item in selected_models:
        model = item["label"]
        model_summary = summary[summary["model"] == model].set_index("weather_condition")

        lines.append(f"## {model}\n")

        if "Favorable" in model_summary.index and "Difficult" in model_summary.index:
            favorable_mae = model_summary.loc["Favorable", "mae"]
            difficult_mae = model_summary.loc["Difficult", "mae"]
            delta = difficult_mae - favorable_mae

            lines.append(f"- Favorable weather MAE: {favorable_mae:.2f}")
            lines.append(f"- Difficult weather MAE: {difficult_mae:.2f}")
            lines.append(f"- Difference difficult minus favorable: {delta:.2f}")

        if "Difficult" in model_summary.index:
            difficult_bias = model_summary.loc["Difficult", "mean_residual"]

            if difficult_bias < 0:
                bias_text = "underprediction"
            elif difficult_bias > 0:
                bias_text = "overprediction"
            else:
                bias_text = "almost no average bias"

            lines.append(
                f"- Mean residual in difficult weather: {difficult_bias:.2f} "
                f"({bias_text})"
            )

        lines.append("")

    lines.append("## How to use this in Q&A\n")
    lines.append("- Use MAE by weather condition to discuss whether bad weather increases errors.")
    lines.append("- Use bias by weather condition to discuss systematic over- or underprediction.")
    lines.append("- Use temperature, precipitation, wind, and humidity plots to identify specific weak spots.")
    lines.append("- Use correlation diagnostics only as descriptive evidence, not as causal proof.\n")

    report_path = OUTPUT_DIR / "weather_insight_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {report_path.relative_to(ROOT)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    selected_models = load_selected_model_data()

    print("\nCreating weather-specific diagnostic insights...")
    print(f"Output folder: {OUTPUT_DIR.relative_to(ROOT)}")

    plot_mae_by_weather_condition(selected_models)
    plot_bias_by_weather_condition(selected_models)
    plot_actual_vs_predicted_by_weather(selected_models)

    plot_mae_by_category(
        selected_models,
        "temperature_bin",
        "MAE by Maximum Temperature",
        "04_mae_by_temperature_bin",
    )

    plot_mae_by_category(
        selected_models,
        "precipitation_bin",
        "MAE by Precipitation Level",
        "05_mae_by_precipitation_bin",
    )

    plot_mae_by_category(
        selected_models,
        "wind_bin",
        "MAE by Wind Level",
        "06_mae_by_wind_bin",
    )

    plot_mae_by_category(
        selected_models,
        "humidity_bin",
        "MAE by Humidity Level",
        "07_mae_by_humidity_bin",
    )

    plot_weather_error_correlations(selected_models)
    write_weather_insight_report(selected_models)

    print("\nDone.")
    print("Created weather-specific diagnostic plots and a markdown insight report.")


if __name__ == "__main__":
    main()