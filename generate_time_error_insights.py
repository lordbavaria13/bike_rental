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

OUTPUT_DIR = ROOT / "presentation_figures" / "time_question_insights_detailed"
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
        "color": GREEN,
        "edge_color": GREEN,
    },
    {
        "key": "NN",
        "label": "Neural Network",
        "aliases": ["neural_network", "neural", "mlp"],
        "color": MID_GREY,
        "edge_color": MID_GREY,
    },
    {
        "key": "Lasso",
        "label": "Lasso",
        "aliases": ["lasso"],
        "color": VERY_LIGHT_GREY,
        "edge_color": LIGHT_GREY,
    },
]

MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

MONTH_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


# ============================================================
# Basic helpers
# ============================================================

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def safe_filename(text: str) -> str:
    return normalize(text).replace("_", "-")


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


def path_matches_alias(path: Path, aliases: list[str]) -> bool:
    path_text = normalize(path.relative_to(ROOT))
    return any(alias in path_text for alias in aliases)


# ============================================================
# Locate selected model result folders
# ============================================================

def find_result_dir(model_config: dict) -> Path:
    label = model_config["label"]
    aliases = model_config["aliases"]

    candidates = []

    for metrics_path in RESULTS_ROOT.rglob(f"{EXPERIMENT}/metrics.json"):
        path_text = normalize(metrics_path.relative_to(ROOT))

        if not path_matches_alias(metrics_path, aliases):
            continue

        # Prevent accidental matches when searching for Neural Network
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
# Load context data
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
                "Could not find encoded split files or "
                "daily_rentals_top20_reduced_with_lag.csv"
            )

        context = pd.read_csv(fallback)

    if "start_station_id_raw" in context.columns:
        context["start_station_id"] = context["start_station_id_raw"]

    required = {
        "time_idx",
        "start_station_id",
        TARGET_COL,
        "weekday",
        "month",
        "year",
    }

    missing = required - set(context.columns)

    if missing:
        raise ValueError(
            f"Context data is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(context.columns)}"
        )

    numeric_cols = [
        "time_idx",
        "start_station_id",
        TARGET_COL,
        "weekday",
        "month",
        "year",
    ]

    for col in numeric_cols:
        context[col] = pd.to_numeric(context[col], errors="coerce")

    return context


# ============================================================
# Load predictions and merge context
# ============================================================

def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "time_idx",
        "start_station_id",
        TARGET_COL,
        PRED_COL,
    }

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

    df = df.dropna(
        subset=[
            "time_idx",
            "start_station_id",
            TARGET_COL,
            PRED_COL,
        ]
    )

    df["residual"] = df[PRED_COL] - df[TARGET_COL]
    df["abs_error"] = df["residual"].abs()

    return df


def merge_predictions_with_context(
    pred: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:

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

        use_cols = list(dict.fromkeys(keys + ["weekday", "month", "year"]))

        context_unique = context[use_cols].drop_duplicates(subset=keys)

        merged = pred.merge(
            context_unique,
            on=keys,
            how="left",
            suffixes=("", "_context"),
        )

        coverage = merged[["weekday", "month", "year"]].notna().mean().mean()

        if coverage > best_coverage:
            best_coverage = coverage
            best_merge = merged
            best_keys = keys

    if best_merge is None:
        raise ValueError("Could not merge predictions with time context data.")

    print(f"Merge keys used: {best_keys}")
    print(f"Time context coverage after merge: {best_coverage:.1%}")

    if best_coverage < 0.95:
        print("WARNING: Time context merge coverage is below 95%. Check keys and processed data.")

    return best_merge


# ============================================================
# Add detailed temporal categories
# ============================================================

def add_temporal_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Reconstruct calendar date.
    # Preferred: use existing date column if present.
    date_col = None

    for candidate in ["date", "dteday", "day", "datetime"]:
        if candidate in out.columns:
            date_col = candidate
            break

    if date_col:
        out["date_reconstructed"] = pd.to_datetime(out[date_col], errors="coerce")
    else:
        # In this project time_idx starts from 2020-05-01.
        start_date = pd.Timestamp("2020-05-01")
        out["date_reconstructed"] = start_date + pd.to_timedelta(
            out["time_idx"],
            unit="D",
        )

    out["day_of_month"] = out["date_reconstructed"].dt.day
    out["month_from_date"] = out["date_reconstructed"].dt.month
    out["year_from_date"] = out["date_reconstructed"].dt.year

    # Prefer the existing month feature if present, otherwise use reconstructed date.
    out["month_numeric"] = pd.to_numeric(
        out["month"].fillna(out["month_from_date"]),
        errors="coerce",
    )

    out["month_name"] = out["month_numeric"].round().astype("Int64").map(MONTH_MAP)

    out["month_name"] = pd.Categorical(
        out["month_name"],
        categories=MONTH_ORDER,
        ordered=True,
    )

    # Keep day 1-31 as ordered integer category.
    out["day_of_month"] = pd.to_numeric(out["day_of_month"], errors="coerce")
    out["day_of_month_int"] = out["day_of_month"].round().astype("Int64")

    return out


def load_selected_model_data() -> list[dict]:
    context = load_context_data()
    selected = []

    for model_config in SELECTED_MODELS:
        result_dir = find_result_dir(model_config)
        metrics = load_json(result_dir / "metrics.json")
        predictions = load_predictions(result_dir / "predictions.csv")

        merged = merge_predictions_with_context(predictions, context)
        merged = add_temporal_categories(merged)
        merged["model"] = model_config["label"]

        selected.append(
            {
                "key": model_config["key"],
                "label": model_config["label"],
                "color": model_config["color"],
                "edge_color": model_config["edge_color"],
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

    combined_path = OUTPUT_DIR / "time_enriched_test_predictions_selected_models.csv"
    combined.to_csv(combined_path, index=False)

    print(f"\nSaved enriched prediction data: {combined_path.relative_to(ROOT)}")

    return selected


# ============================================================
# Aggregation
# ============================================================

def aggregate_single_model_by_category(
    model_item: dict,
    category_col: str,
    full_index,
) -> pd.DataFrame:

    df = model_item["data"].copy()

    if category_col not in df.columns:
        raise ValueError(f"Column {category_col} not found for {model_item['label']}")

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

    grouped = grouped.set_index(category_col).reindex(full_index).reset_index()
    grouped["model"] = model_item["label"]

    return grouped


# ============================================================
# Plot 1: one plot per model for each day of month
# ============================================================

def plot_day_of_month_for_each_model(selected_models: list[dict]) -> None:
    all_rows = []
    full_index = list(range(1, 32))

    for model_item in selected_models:
        model_name = model_item["label"]
        model_slug = safe_filename(model_name)

        grouped = aggregate_single_model_by_category(
            model_item=model_item,
            category_col="day_of_month_int",
            full_index=full_index,
        )

        all_rows.append(grouped)

        fig, ax = plt.subplots(figsize=(13.5, 6.2))

        bars = ax.bar(
            grouped["day_of_month_int"].astype(str),
            grouped["mae"],
            color=model_item["color"],
            edgecolor=model_item["edge_color"],
            linewidth=0.8,
            width=0.72,
        )

        ax.set_title(
            f"{model_name}: MAE by Day of Month",
            fontsize=18,
            fontweight="bold",
            color=DARK_GREY,
            pad=18,
        )

        ax.set_xlabel("Day of month", fontsize=12, color=DARK_GREY)
        ax.set_ylabel("MAE", fontsize=12, color=DARK_GREY)

        ax.set_xticks(np.arange(len(full_index)))
        ax.set_xticklabels([str(day) for day in full_index], fontsize=9)

        ymax = grouped["mae"].max(skipna=True) * 1.18

        if pd.isna(ymax) or ymax <= 0:
            ymax = 1

        ax.set_ylim(0, ymax)

        for bar, value in zip(bars, grouped["mae"]):
            if pd.isna(value):
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.012,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=DARK_GREY,
                fontweight="bold",
                rotation=90,
            )

        style_axis(ax)
        fig.tight_layout()

        save_plot(fig, f"01_{model_slug}_mae_by_each_day_of_month")

    combined = pd.concat(all_rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "01_mae_by_each_day_of_month_all_selected_models.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")


# ============================================================
# Plot 2: one plot per model for each month of year
# ============================================================

def plot_month_of_year_for_each_model(selected_models: list[dict]) -> None:
    all_rows = []
    full_index = MONTH_ORDER

    for model_item in selected_models:
        model_name = model_item["label"]
        model_slug = safe_filename(model_name)

        grouped = aggregate_single_model_by_category(
            model_item=model_item,
            category_col="month_name",
            full_index=full_index,
        )

        all_rows.append(grouped)

        fig, ax = plt.subplots(figsize=(12.5, 6.2))

        bars = ax.bar(
            grouped["month_name"].astype(str),
            grouped["mae"],
            color=model_item["color"],
            edgecolor=model_item["edge_color"],
            linewidth=0.8,
            width=0.68,
        )

        ax.set_title(
            f"{model_name}: MAE by Month of Year",
            fontsize=18,
            fontweight="bold",
            color=DARK_GREY,
            pad=18,
        )

        ax.set_xlabel("Month", fontsize=12, color=DARK_GREY)
        ax.set_ylabel("MAE", fontsize=12, color=DARK_GREY)

        ymax = grouped["mae"].max(skipna=True) * 1.18

        if pd.isna(ymax) or ymax <= 0:
            ymax = 1

        ax.set_ylim(0, ymax)

        for bar, value in zip(bars, grouped["mae"]):
            if pd.isna(value):
                continue

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

        style_axis(ax)
        fig.tight_layout()

        save_plot(fig, f"02_{model_slug}_mae_by_each_month_of_year")

    combined = pd.concat(all_rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "02_mae_by_each_month_of_year_all_selected_models.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")


# ============================================================
# Optional additional line plots: residual bias by day/month
# Useful for Q&A, but still one plot per model
# ============================================================

def plot_bias_by_day_of_month_for_each_model(selected_models: list[dict]) -> None:
    all_rows = []
    full_index = list(range(1, 32))

    for model_item in selected_models:
        model_name = model_item["label"]
        model_slug = safe_filename(model_name)

        grouped = aggregate_single_model_by_category(
            model_item=model_item,
            category_col="day_of_month_int",
            full_index=full_index,
        )

        all_rows.append(grouped)

        fig, ax = plt.subplots(figsize=(13.5, 6.2))

        ax.plot(
            grouped["day_of_month_int"],
            grouped["mean_residual"],
            marker="o",
            linewidth=2.0,
            color=model_item["color"],
        )

        ax.axhline(
            0,
            color=DARK_GREY,
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_title(
            f"{model_name}: Prediction Bias by Day of Month",
            fontsize=18,
            fontweight="bold",
            color=DARK_GREY,
            pad=18,
        )

        ax.set_xlabel("Day of month", fontsize=12, color=DARK_GREY)
        ax.set_ylabel("Mean residual: predicted − actual", fontsize=12, color=DARK_GREY)

        ax.set_xticks(full_index)
        ax.set_xticklabels([str(day) for day in full_index], fontsize=9)

        style_axis(ax)
        fig.tight_layout()

        save_plot(fig, f"03_{model_slug}_bias_by_each_day_of_month")

    combined = pd.concat(all_rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "03_bias_by_each_day_of_month_all_selected_models.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")


def plot_bias_by_month_of_year_for_each_model(selected_models: list[dict]) -> None:
    all_rows = []
    full_index = MONTH_ORDER

    for model_item in selected_models:
        model_name = model_item["label"]
        model_slug = safe_filename(model_name)

        grouped = aggregate_single_model_by_category(
            model_item=model_item,
            category_col="month_name",
            full_index=full_index,
        )

        all_rows.append(grouped)

        fig, ax = plt.subplots(figsize=(12.5, 6.2))

        ax.plot(
            grouped["month_name"].astype(str),
            grouped["mean_residual"],
            marker="o",
            linewidth=2.0,
            color=model_item["color"],
        )

        ax.axhline(
            0,
            color=DARK_GREY,
            linestyle="--",
            linewidth=1.2,
        )

        ax.set_title(
            f"{model_name}: Prediction Bias by Month of Year",
            fontsize=18,
            fontweight="bold",
            color=DARK_GREY,
            pad=18,
        )

        ax.set_xlabel("Month", fontsize=12, color=DARK_GREY)
        ax.set_ylabel("Mean residual: predicted − actual", fontsize=12, color=DARK_GREY)

        style_axis(ax)
        fig.tight_layout()

        save_plot(fig, f"04_{model_slug}_bias_by_each_month_of_year")

    combined = pd.concat(all_rows, ignore_index=True)
    csv_path = OUTPUT_DIR / "04_bias_by_each_month_of_year_all_selected_models.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(ROOT)}")


# ============================================================
# Report
# ============================================================

def write_detailed_time_insight_report(selected_models: list[dict]) -> None:
    lines = []

    lines.append("# Detailed Time Error Insight Report\n")
    lines.append("This report is generated from test predictions and reconstructed calendar information.\n")
    lines.append("It evaluates whether errors differ by individual day of month and individual month of year.\n")

    for model_item in selected_models:
        model = model_item["label"]

        day_summary = aggregate_single_model_by_category(
            model_item=model_item,
            category_col="day_of_month_int",
            full_index=list(range(1, 32)),
        )

        month_summary = aggregate_single_model_by_category(
            model_item=model_item,
            category_col="month_name",
            full_index=MONTH_ORDER,
        )

        day_summary_clean = day_summary.dropna(subset=["mae"])
        month_summary_clean = month_summary.dropna(subset=["mae"])

        lines.append(f"## {model}\n")

        if not day_summary_clean.empty:
            best_day = day_summary_clean.loc[day_summary_clean["mae"].idxmin()]
            worst_day = day_summary_clean.loc[day_summary_clean["mae"].idxmax()]

            lines.append(
                f"- Best day of month by MAE: day {int(best_day['day_of_month_int'])} "
                f"with MAE {best_day['mae']:.2f}."
            )
            lines.append(
                f"- Worst day of month by MAE: day {int(worst_day['day_of_month_int'])} "
                f"with MAE {worst_day['mae']:.2f}."
            )

        if not month_summary_clean.empty:
            best_month = month_summary_clean.loc[month_summary_clean["mae"].idxmin()]
            worst_month = month_summary_clean.loc[month_summary_clean["mae"].idxmax()]

            lines.append(
                f"- Best month by MAE: {best_month['month_name']} "
                f"with MAE {best_month['mae']:.2f}."
            )
            lines.append(
                f"- Worst month by MAE: {worst_month['month_name']} "
                f"with MAE {worst_month['mae']:.2f}."
            )

        lines.append("")

    lines.append("## How to use this in Q&A\n")
    lines.append("- Use day-of-month plots to discuss whether beginning, middle, or end of month contains harder cases.")
    lines.append("- Use month-of-year plots to discuss seasonal effects more precisely than the earlier Beginning/Middle/End-of-year plot.")
    lines.append("- Use bias plots to explain whether a model systematically overpredicts or underpredicts in specific time periods.")
    lines.append("- These diagnostics are descriptive and should not be interpreted as causal proof.\n")

    report_path = OUTPUT_DIR / "detailed_time_insight_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {report_path.relative_to(ROOT)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    selected_models = load_selected_model_data()

    print("\nCreating detailed time-specific diagnostic insights...")
    print(f"Output folder: {OUTPUT_DIR.relative_to(ROOT)}")

    # Required detailed plots:
    plot_day_of_month_for_each_model(selected_models)
    plot_month_of_year_for_each_model(selected_models)

    # Additional useful bias plots:
    plot_bias_by_day_of_month_for_each_model(selected_models)
    plot_bias_by_month_of_year_for_each_model(selected_models)

    write_detailed_time_insight_report(selected_models)

    print("\nDone.")
    print("Created individual day-of-month and month-of-year plots for each selected model.")


if __name__ == "__main__":
    main()