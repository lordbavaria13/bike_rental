import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Settings
# =========================
DATA_PATH = Path("data/processed/daily_rentals_top20_reduced.csv")
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

STATION_COL = "start_station_id"
RENTALS_COL = "total_rentals"

OUTPUT_FILE = OUTPUT_DIR / "total_rentals_per_station_id.png"

# Optional: show only the top N stations if there are too many
TOP_N = 30   # set to None to show all stations

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# Check columns
# =========================
required_cols = [STATION_COL, RENTALS_COL]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

# =========================
# Calculate total rentals per station ID
# =========================
rentals_per_station = (
    df.groupby(STATION_COL)[RENTALS_COL]
    .sum()
    .sort_values(ascending=False)
)

if TOP_N is not None:
    rentals_per_station = rentals_per_station.head(TOP_N)

# =========================
# Create bar chart
# =========================
plt.figure(figsize=(14, 7))

bars = plt.bar(
    rentals_per_station.index.astype(str),
    rentals_per_station.values
)

plt.title(
    "Total Rentals per Station ID in the Train Dataset",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Station ID", fontsize=12)
plt.ylabel("Total Rentals", fontsize=12)

plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Add values above the bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

# =========================
# Save and show chart
# =========================
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

print(f"Chart saved to: {OUTPUT_FILE}")