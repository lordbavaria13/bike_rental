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

OUTPUT_DIR = ROOT / "presentation_figures" / "question_backup_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "total_rentals"
PRED_COL = "prediction"
SPLIT_COL = "split"

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

FULL_NAME_MAP = {
    "Linear": "Linear Regression",
    "Ridge": "Ridge",
    "Lasso": "Lasso",
    "Tree": "Decision Tree",
    "KNN": "KNN",
    "RF": "Random Forest",
    "GB": "Gradient Boosting",
    "NN": "Neural Network",
}

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

SELECTED_FINAL_MODELS = ["GB", "NN", "Lasso"]


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

    path_text = normalize(path.relative_to(ROOT))

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


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def order_models(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order"] = df["Model"].apply(
        lambda model: MODEL_ORDER.index(model) if model in MODEL_ORDER else 999
    )
    return df.sort_values(["order", "Model"]).drop(columns="order")


def load_all_metrics() -> pd.DataFrame:
    rows = []

    for metrics_path in sorted(RESULTS_ROOT.rglob("metrics.json")):
        path_text = normalize(metrics_path.relative_to(ROOT))

        if "99_model_comparison" in path_text:
            continue

        try:
            metrics = load_json(metrics_path)
        except Exception:
            continue

        required = [
            "train_rmse",
            "validation_rmse",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_mape",
        ]

        if not all(key in metrics for key in required):
            continue

        model = get_display_name(metrics, metrics_path)

        if "dummy" in normalize(model) or "dummy" in path_text:
            continue

        experiment = "with_lag" if metrics.get("uses_lag_features") else "without_lag"

        rows.append(
            {
                "Model": model,
                "Model full": FULL_NAME_MAP.get(model, model),
                "Experiment": experiment,
                "Train RMSE": float(metrics["train_rmse"]),
                "Validation RMSE": float(metrics["validation_rmse"]),
                "Test RMSE": float(metrics["test_rmse"]),
                "Test MAE": float(metrics["test_mae"]),
                "Test R2": float(metrics["test_r2"]),
                "Test MAPE": float(metrics["test_mape"]),
                "n_features": metrics.get("n_features", np.nan),
                "Path": str(metrics_path.relative_to(ROOT)),
            }
        )

    if not rows:
        raise ValueError("No valid model metrics were found under modelling/.")

    df = pd.DataFrame(rows)

    # If duplicate runs exist, keep the best test RMSE per model and experiment
    df = (
        df.sort_values("Test RMSE", ascending=True)
        .drop_duplicates(subset=["Model", "Experiment"], keep="first")
    )

    return order_models(df)


def get_experiment_df(metrics_df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    df = metrics_df[metrics_df["Experiment"] == experiment].copy()
    return order_models(df)


def save_plot(fig, filename_base: str) -> None:
    png_path = OUTPUT_DIR / f"{filename_base}.png"
    pdf_path = OUTPUT_DIR / f"{filename_base}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path.relative_to(ROOT)}")
    print(f"Saved: {pdf_path.relative_to(ROOT)}")


def style_axis(ax) -> None:
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors=DARK_GREY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)


# ============================================================
# Plot 1: lag vs no-lag test RMSE
# ============================================================

def plot_lag_vs_no_lag(metrics_df: pd.DataFrame) -> None:
    no_lag = get_experiment_df(metrics_df, "without_lag")[["Model", "Test RMSE"]]
    with_lag = get_experiment_df(metrics_df, "with_lag")[["Model", "Test RMSE"]]

    df = no_lag.merge(
        with_lag,
        on="Model",
        suffixes=(" without lag", " with lag"),
        how="inner",
    )
    df = order_models(df)

    x = np.arange(len(df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12.5, 6.4))

    bars_no_lag = ax.bar(
        x - width / 2,
        df["Test RMSE without lag"],
        width,
        label="Without lag",
        color=MID_GREY,
    )

    bars_with_lag = ax.bar(
        x + width / 2,
        df["Test RMSE with lag"],
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
    ax.set_xticklabels(df["Model"])

    ymax = max(df["Test RMSE without lag"].max(), df["Test RMSE with lag"].max()) * 1.22
    ax.set_ylim(0, ymax)

    for bars in [bars_no_lag, bars_with_lag]:
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

    style_axis(ax)
    fig.tight_layout()
    save_plot(fig, "01_lag_vs_without_lag_test_rmse")


# ============================================================
# Plot 2 and 3: RMSE by split for all models
# ============================================================

def plot_rmse_by_split_all_models(metrics_df: pd.DataFrame, experiment: str) -> None:
    df = get_experiment_df(metrics_df, experiment)

    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12.8, 6.4))

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

    title_suffix = "With Lag Features" if experiment == "with_lag" else "Without Lag Features"

    ax.set_title(
        f"RMSE by Split: {title_suffix}",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=42,
    )

    ax.set_xlabel("Model", fontsize=12, color=DARK_GREY)
    ax.set_ylabel("RMSE Score", fontsize=12, color=DARK_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ymax = max(df["Train RMSE"].max(), df["Validation RMSE"].max(), df["Test RMSE"].max()) * 1.25
    ax.set_ylim(0, ymax)

    # Label test bars only to avoid visual overload
    for bar in test_bars:
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

    style_axis(ax)
    fig.tight_layout()

    prefix = "02" if experiment == "without_lag" else "03"
    save_plot(fig, f"{prefix}_{experiment}_rmse_train_validation_test_all_models")


# ============================================================
# Plot 4: selected final models RMSE by split
# ============================================================

def plot_selected_final_models_rmse(metrics_df: pd.DataFrame) -> None:
    df = get_experiment_df(metrics_df, "with_lag")
    df = df[df["Model"].isin(SELECTED_FINAL_MODELS)].copy()
    df = df.set_index("Model").loc[SELECTED_FINAL_MODELS].reset_index()

    models = df["Model full"].tolist()
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

    ymax = max(df["Train RMSE"].max(), df["Validation RMSE"].max(), df["Test RMSE"].max()) * 1.28
    ax.set_ylim(0, ymax)

    for bars in [train_bars, val_bars, test_bars]:
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

    style_axis(ax)
    fig.tight_layout()
    save_plot(fig, "04_selected_final_models_rmse_train_validation_test")


# ============================================================
# Plot 5: scatterplots for selected final models
# ============================================================

def find_result_dir_for_model(model: str) -> Path:
    aliases = {
        "GB": ["gradient_boost", "gradient_boosting", "gradientboost", "boost"],
        "NN": ["neural_network", "neural", "mlp", "nn"],
        "Lasso": ["lasso"],
    }[model]

    candidates = []

    for metrics_path in RESULTS_ROOT.rglob("with_lag/metrics.json"):
        path_text = normalize(metrics_path.relative_to(ROOT))

        if not any(alias in path_text for alias in aliases):
            continue

        if model == "NN" and ("random" in path_text or "forest" in path_text):
            continue

        result_dir = metrics_path.parent
        predictions_path = result_dir / "predictions.csv"

        if not predictions_path.exists():
            continue

        metrics = load_json(metrics_path)

        if metrics.get("uses_lag_features") is not True:
            continue

        candidates.append((float(metrics["test_rmse"]), result_dir, metrics))

    if not candidates:
        raise FileNotFoundError(f"No valid with_lag result directory found for {model}")

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def load_test_predictions(result_dir: Path) -> pd.DataFrame:
    path = result_dir / "predictions.csv"

    df = pd.read_csv(path)

    required = {TARGET_COL, PRED_COL, SPLIT_COL}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    test_df = df[df[SPLIT_COL].astype(str).str.lower() == "test"].copy()
    test_df[TARGET_COL] = pd.to_numeric(test_df[TARGET_COL], errors="coerce")
    test_df[PRED_COL] = pd.to_numeric(test_df[PRED_COL], errors="coerce")
    test_df = test_df.dropna(subset=[TARGET_COL, PRED_COL])

    if test_df.empty:
        raise ValueError(f"No valid test rows found in {path}")

    return test_df


def plot_selected_scatter() -> None:
    plot_items = []

    for model in SELECTED_FINAL_MODELS:
        result_dir = find_result_dir_for_model(model)
        metrics = load_json(result_dir / "metrics.json")
        test_df = load_test_predictions(result_dir)

        plot_items.append(
            {
                "Model": model,
                "Model full": FULL_NAME_MAP.get(model, model),
                "metrics": metrics,
                "test_df": test_df,
            }
        )

    all_actual = np.concatenate([item["test_df"][TARGET_COL].to_numpy() for item in plot_items])
    all_pred = np.concatenate([item["test_df"][PRED_COL].to_numpy() for item in plot_items])

    axis_min = 0
    axis_max = max(all_actual.max(), all_pred.max()) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)

    for ax, item in zip(axes, plot_items):
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
            item["Model full"],
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
    save_plot(fig, "05_selected_final_models_actual_vs_predicted_scatter")


# ============================================================
# Plot 6: lag improvement delta
# ============================================================

def plot_lag_delta(metrics_df: pd.DataFrame) -> None:
    no_lag = get_experiment_df(metrics_df, "without_lag")[["Model", "Test RMSE"]]
    with_lag = get_experiment_df(metrics_df, "with_lag")[["Model", "Test RMSE"]]

    df = no_lag.merge(with_lag, on="Model", suffixes=(" without lag", " with lag"))
    df["RMSE improvement"] = df["Test RMSE without lag"] - df["Test RMSE with lag"]
    df = order_models(df)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    bars = ax.bar(
        df["Model"],
        df["RMSE improvement"],
        color=GREEN,
        width=0.62,
    )

    ax.set_title(
        "Test RMSE Improvement from Lag Features",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=18,
    )

    ax.set_xlabel("Model", fontsize=12, color=DARK_GREY)
    ax.set_ylabel("RMSE reduction", fontsize=12, color=DARK_GREY)

    ymax = df["RMSE improvement"].max() * 1.22
    ax.set_ylim(0, ymax)

    for bar, value in zip(bars, df["RMSE improvement"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.012,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=DARK_GREY,
            fontweight="bold",
        )

    style_axis(ax)
    fig.tight_layout()
    save_plot(fig, "06_lag_feature_rmse_improvement_delta")


# ============================================================
# Plot 7: feature importance for Gradient Boosting
# ============================================================

def find_gradient_boosting_result_dir() -> Path | None:
    try:
        return find_result_dir_for_model("GB")
    except FileNotFoundError:
        return None


def plot_gradient_boosting_feature_importance() -> None:
    result_dir = find_gradient_boosting_result_dir()

    if result_dir is None:
        print("Skipped feature importance: Gradient Boosting result dir not found.")
        return

    importance_path = result_dir / "feature_importance.csv"

    if not importance_path.exists():
        print(f"Skipped feature importance: {importance_path.relative_to(ROOT)} not found.")
        return

    df = pd.read_csv(importance_path)

    # Detect columns flexibly
    columns_norm = {normalize(col): col for col in df.columns}

    feature_col = None
    importance_col = None

    for candidate in ["feature", "feature_name", "variable"]:
        if candidate in columns_norm:
            feature_col = columns_norm[candidate]
            break

    for candidate in ["importance", "feature_importance", "value"]:
        if candidate in columns_norm:
            importance_col = columns_norm[candidate]
            break

    if feature_col is None or importance_col is None:
        print(f"Skipped feature importance: could not detect columns in {importance_path}")
        print(f"Columns found: {list(df.columns)}")
        return

    df[importance_col] = pd.to_numeric(df[importance_col], errors="coerce")
    df = df.dropna(subset=[importance_col])
    df = df.sort_values(importance_col, ascending=False).head(15)
    df = df.sort_values(importance_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    bars = ax.barh(
        df[feature_col],
        df[importance_col],
        color=GREEN,
    )

    ax.set_title(
        "Top Gradient Boosting Feature Importances",
        fontsize=18,
        fontweight="bold",
        color=DARK_GREY,
        pad=18,
    )

    ax.set_xlabel("Importance", fontsize=12, color=DARK_GREY)
    ax.set_ylabel("Feature", fontsize=12, color=DARK_GREY)

    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors=DARK_GREY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color(LIGHT_GREY)
    ax.spines["bottom"].set_color(LIGHT_GREY)

    fig.tight_layout()
    save_plot(fig, "07_gradient_boosting_feature_importance_top15")


# ============================================================
# Export tables useful for questions
# ============================================================

def export_question_tables(metrics_df: pd.DataFrame) -> None:
    metrics_out = OUTPUT_DIR / "all_metrics_summary.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Saved: {metrics_out.relative_to(ROOT)}")

    selected = get_experiment_df(metrics_df, "with_lag")
    selected = selected[selected["Model"].isin(SELECTED_FINAL_MODELS)].copy()
    selected = selected.set_index("Model").loc[SELECTED_FINAL_MODELS].reset_index()

    selected_out = OUTPUT_DIR / "selected_final_models_summary.csv"
    selected.to_csv(selected_out, index=False)
    print(f"Saved: {selected_out.relative_to(ROOT)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading metrics...")
    metrics_df = load_all_metrics()

    print("\nAll loaded metrics:")
    print(
        metrics_df[
            [
                "Model",
                "Experiment",
                "Train RMSE",
                "Validation RMSE",
                "Test RMSE",
                "Test MAE",
                "Test R2",
                "Test MAPE",
                "Path",
            ]
        ].to_string(index=False)
    )

    export_question_tables(metrics_df)

    print("\nGenerating plots...")
    plot_lag_vs_no_lag(metrics_df)
    plot_rmse_by_split_all_models(metrics_df, "without_lag")
    plot_rmse_by_split_all_models(metrics_df, "with_lag")
    plot_selected_final_models_rmse(metrics_df)
    plot_selected_scatter()
    plot_lag_delta(metrics_df)
    plot_gradient_boosting_feature_importance()

    print("\nDone.")
    print(f"All outputs saved in: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()