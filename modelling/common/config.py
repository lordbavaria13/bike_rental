from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

# Experiment names used everywhere in the project.
EXPERIMENT_WITHOUT_LAG = "without_lag"
EXPERIMENT_WITH_LAG = "with_lag"
EXPERIMENTS = (EXPERIMENT_WITHOUT_LAG, EXPERIMENT_WITH_LAG)
DEFAULT_EXPERIMENT = EXPERIMENT_WITHOUT_LAG

TARGET_COL = "total_rentals"
TIME_COL = "time_idx"
STATION_COL = "start_station_id"
RAW_STATION_CONTEXT_COL = "start_station_id_raw"
CONTEXT_COLUMNS = (RAW_STATION_CONTEXT_COL,)

# Lag setup. These features are created station-specifically and only from the past.
LAG_FEATURES = (1, 7)
LAG_FEATURE_NAMES = tuple(f"{TARGET_COL}_lag_{lag}" for lag in LAG_FEATURES)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

FIGSIZE = (8, 5)
DPI = 150
TITLE_SIZE = 13
LABEL_SIZE = 11


def get_experiment_dir(experiment: str) -> Path:
    """Return data/processed/<experiment> after validating the experiment name."""
    validate_experiment(experiment)
    return DATA_DIR / experiment


def get_experiment_paths(experiment: str) -> dict[str, Path]:
    """Return all dataset paths for one experiment variant."""
    experiment_dir = get_experiment_dir(experiment)
    dataset_name = f"daily_rentals_top20_reduced_{experiment}.csv"

    return {
        "experiment_dir": experiment_dir,
        "data_path": experiment_dir / dataset_name,
        "encoded_train_path": experiment_dir / "encoded_train.csv",
        "encoded_val_path": experiment_dir / "encoded_validation.csv",
        "encoded_test_path": experiment_dir / "encoded_test.csv",
        "encoded_features_path": experiment_dir / "encoded_feature_names.csv",
    }


def validate_experiment(experiment: str) -> None:
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{experiment}'. Expected one of: {list(EXPERIMENTS)}"
        )


# Backward-compatible default paths. Older scripts can still import these names,
# but all updated model scripts use get_experiment_paths(...).
DATA_PATH = DATA_DIR / "daily_rentals_top20_reduced.csv"
ENCODED_TRAIN_PATH = get_experiment_paths(DEFAULT_EXPERIMENT)["encoded_train_path"]
ENCODED_VAL_PATH = get_experiment_paths(DEFAULT_EXPERIMENT)["encoded_val_path"]
ENCODED_TEST_PATH = get_experiment_paths(DEFAULT_EXPERIMENT)["encoded_test_path"]
ENCODED_FEATURES_PATH = get_experiment_paths(DEFAULT_EXPERIMENT)["encoded_features_path"]
