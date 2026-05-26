from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "presentation_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

GREEN = "#92D401"
DARK_GREY = "#555555"
MID_GREY = "#8C8C8C"
LIGHT_GREY = "#D9D9D9"
WHITE = "#FFFFFF"

OUR_METRICS_PATH = (
    ROOT
    / "modelling"
    / "07_gradient_boosting"
    / "results"
    / "with_lag"
    / "metrics.json"
)


# ============================================================
# Load our final model metrics
# ============================================================

def load_our_metrics() -> dict:
    """
    Reads our final Gradient Boosting metrics from the result folder.
    Falls back to the values reported in the final paper if the file is missing.
    """

    fallback = {
        "test_mae": 16.92,
        "test_rmse": 22.78,
        "test_r2": 0.663,
        "test_mape": 0.253,
    }

    if not OUR_METRICS_PATH.exists():
        print("WARNING: Own metrics.json not found. Using paper values as fallback.")
        return fallback

    with open(OUR_METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    required_keys = ["test_mae", "test_rmse", "test_r2", "test_mape"]

    if not all(key in metrics for key in required_keys):
        print("WARNING: Own metrics.json incomplete. Using paper values as fallback.")
        return fallback

    return metrics


# ============================================================
# Build comparison table
# ============================================================

def build_comparison_table() -> pd.DataFrame:
    our = load_our_metrics()

    rows = [
        {
            "Work": "Our project",
            "Task / dataset": "Capital Bikeshare, station-day, top 20 stations",
            "Best reported model": "Gradient Boosting + lag",
            "RMSE": f"{our['test_rmse']:.2f}",
            "MAE": f"{our['test_mae']:.2f}",
            "R²": f"{our['test_r2']:.3f}",
            "MAPE": f"{our['test_mape'] * 100:.1f}%",
            "Main comparison point": "Stricter station-level setup; local noise remains visible",
        },
        {
            "Work": "Sathishkumar & Cho",
            "Task / dataset": "Seoul Bike and Capital Bikeshare, system-level demand",
            "Best reported model": "CUBIST",
            "RMSE": "n/a",
            "MAE": "n/a",
            "R²": "0.95 / 0.89",
            "MAPE": "n/a",
            "Main comparison point": "Higher R², but system-level demand is easier than station-level demand",
        },
        {
            "Work": "Sathishkumar et al.",
            "Task / dataset": "Hourly Seoul Bike demand",
            "Best reported model": "Gradient Boosting",
            "RMSE": "n/a",
            "MAE": "n/a",
            "R²": "0.92",
            "MAPE": "n/a",
            "Main comparison point": "Hourly data includes strong time-of-day signal",
        },
        {
            "Work": "Choi",
            "Task / dataset": "Hourly Seoul Bike demand",
            "Best reported model": "Random Forest",
            "RMSE": "282.63",
            "MAE": "169.57",
            "R²": "0.77",
            "MAPE": "n/a",
            "Main comparison point": "Absolute errors are larger because target scale is larger",
        },
        {
            "Work": "Xin",
            "Task / dataset": "UCI-DC, one-hour-ahead forecasting",
            "Best reported model": "N-BEATS",
            "RMSE": "75.88",
            "MAE": "51.33",
            "R²": "n/a",
            "MAPE": "26.7%",
            "Main comparison point": "MAPE is close to our relative error level",
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# Plot table as figure
# ============================================================

def plot_table(df: pd.DataFrame) -> None:
    csv_path = OUTPUT_DIR / "related_work_comparison.csv"
    png_path = OUTPUT_DIR / "related_work_comparison_table.png"
    pdf_path = OUTPUT_DIR / "related_work_comparison_table.pdf"

    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(17, 5.6))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 2.0)

    # Column widths tuned for a presentation slide
    col_widths = {
        0: 0.12,  # Work
        1: 0.25,  # Task / dataset
        2: 0.16,  # Model
        3: 0.07,  # RMSE
        4: 0.07,  # MAE
        5: 0.07,  # R2
        6: 0.07,  # MAPE
        7: 0.29,  # Interpretation
    }

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(LIGHT_GREY)
        cell.set_linewidth(0.8)

        if col in col_widths:
            cell.set_width(col_widths[col])

        if row == 0:
            cell.set_facecolor(GREEN)
            cell.set_text_props(weight="bold", color=DARK_GREY)
        else:
            cell.set_facecolor(WHITE)
            cell.set_text_props(color=DARK_GREY)

        # Highlight our own project row
        if row == 1:
            cell.set_facecolor("#F3FBE8")
            cell.set_text_props(weight="bold", color=DARK_GREY)

    title = (
        "Comparison to Related Bike-Sharing Demand Forecasting Studies"
    )

    subtitle = (
        "Metrics are not fully comparable because studies differ in city, aggregation level, "
        "target scale, and prediction horizon."
    )

    fig.suptitle(
        title,
        fontsize=17,
        fontweight="bold",
        color=DARK_GREY,
        y=0.98,
    )

    ax.set_title(
        subtitle,
        fontsize=10.5,
        color=MID_GREY,
        pad=18,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(f"- {csv_path.relative_to(ROOT)}")
    print(f"- {png_path.relative_to(ROOT)}")
    print(f"- {pdf_path.relative_to(ROOT)}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    df = build_comparison_table()

    print("\nRelated work comparison:")
    print(df.to_string(index=False))

    plot_table(df)


if __name__ == "__main__":
    main()