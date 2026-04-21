from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
DATA_PATH = DATA_DIR / "daily_rentals_top20_reduced.csv"

TARGET_COL = "total_rentals"
TIME_COL = "time_idx"
STATION_COL = "start_station_id"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

FIGSIZE = (8, 5)
DPI = 150
TITLE_SIZE = 13
LABEL_SIZE = 11
