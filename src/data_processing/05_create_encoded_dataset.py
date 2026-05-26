from __future__ import annotations

"""
Create two comparable encoded modelling datasets:

1. without_lag
   Same feature set as before, but restricted to the same station-day rows that
   are available in the lag experiment. This makes the comparison fair.

2. with_lag
   Adds station-specific total_rentals_lag_1 and total_rentals_lag_7.

Important methodological decisions implemented here:
- Lag features are station-specific.
- Lag features use exact time_idx differences, not just the previous available row.
- Rows with missing lag values are removed.
- The non-lag dataset is filtered to exactly the same station-day rows as the lag dataset.
- Rolling or current-target leakage is avoided. Lag values only come from past time_idx.
- Station IDs are one-hot encoded after the chronological split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "daily_rentals_top20_reduced.csv"
OUTPUT_BASE_DIR = BASE_DIR / "data" / "processed"

TARGET_COL = "total_rentals"
TIME_COL = "time_idx"
STATION_COL = "start_station_id"
RAW_STATION_CONTEXT_COL = "start_station_id_raw"
LAGS = (1, 7)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EXPERIMENTS = ("without_lag", "with_lag")


def chronological_split(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically by unique time values."""
    if round(train_ratio + val_ratio + test_ratio, 10) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    unique_time = np.sort(df[time_col].dropna().unique())
    n_time = len(unique_time)

    if n_time < 10:
        raise ValueError("Not enough unique time points for chronological split.")

    train_end = int(n_time * train_ratio)
    val_end = int(n_time * (train_ratio + val_ratio))

    train_times = unique_time[:train_end]
    val_times = unique_time[train_end:val_end]
    test_times = unique_time[val_end:]

    train_df = df[df[time_col].isin(train_times)].copy()
    val_df = df[df[time_col].isin(val_times)].copy()
    test_df = df[df[time_col].isin(test_times)].copy()

    return train_df, val_df, test_df


def validate_input_columns(df: pd.DataFrame) -> None:
    required_cols = {TARGET_COL, TIME_COL, STATION_COL}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Input dataset is missing required columns: {missing_cols}")


def collapse_duplicate_station_days(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure one row per station-day before creating exact lag features.

    Duplicates are rare in the current reduced dataset. For duplicate station-day
    rows, weather/calendar features are identical or should describe the same day,
    while total_rentals is additive demand. Therefore we sum the target and keep
    the first value for all other columns.
    """
    duplicate_count = int(df.duplicated([STATION_COL, TIME_COL]).sum())
    if duplicate_count == 0:
        return df.copy()

    print(f"Found {duplicate_count} duplicate station-day rows. Collapsing them before lag creation...")
    aggregation = {col: "first" for col in df.columns if col not in {STATION_COL, TIME_COL, TARGET_COL}}
    aggregation[TARGET_COL] = "sum"

    collapsed = (
        df.groupby([STATION_COL, TIME_COL], as_index=False)
        .agg(aggregation)
        .sort_values([TIME_COL, STATION_COL])
        .reset_index(drop=True)
    )
    return collapsed


def add_exact_station_lags(df: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    """Add exact station-specific lag columns using time_idx - lag lookups."""
    validate_input_columns(df)

    out = df.copy()
    out[STATION_COL] = out[STATION_COL].astype(str)

    source = out[[STATION_COL, TIME_COL, TARGET_COL]].copy()

    for lag in lags:
        lag_col = f"{TARGET_COL}_lag_{lag}"
        lookup = source.rename(
            columns={
                TIME_COL: f"{TIME_COL}_lookup_{lag}",
                TARGET_COL: lag_col,
            }
        )

        out[f"{TIME_COL}_lookup_{lag}"] = out[TIME_COL] - lag
        out = out.merge(
            lookup,
            how="left",
            left_on=[STATION_COL, f"{TIME_COL}_lookup_{lag}"],
            right_on=[STATION_COL, f"{TIME_COL}_lookup_{lag}"],
        )
        out = out.drop(columns=[f"{TIME_COL}_lookup_{lag}"])

    return out


def build_comparable_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build without_lag and with_lag datasets on identical station-day rows."""
    df_base = collapse_duplicate_station_days(df)
    df_base[STATION_COL] = df_base[STATION_COL].astype(str)

    lagged_df = add_exact_station_lags(df_base, LAGS)
    lag_cols = [f"{TARGET_COL}_lag_{lag}" for lag in LAGS]

    with_lag_df = lagged_df.dropna(subset=lag_cols).copy()

    # Keep exactly the same station-day rows for the no-lag baseline.
    valid_keys = with_lag_df[[STATION_COL, TIME_COL]].drop_duplicates()
    without_lag_df = df_base.merge(
        valid_keys,
        how="inner",
        on=[STATION_COL, TIME_COL],
    )

    # Sort both variants identically for reproducibility.
    sort_cols = [TIME_COL, STATION_COL]
    without_lag_df = without_lag_df.sort_values(sort_cols).reset_index(drop=True)
    with_lag_df = with_lag_df.sort_values(sort_cols).reset_index(drop=True)

    # Safety check: both variants must contain the same keys in the same order.
    no_lag_keys = without_lag_df[[STATION_COL, TIME_COL]].reset_index(drop=True)
    lag_keys = with_lag_df[[STATION_COL, TIME_COL]].reset_index(drop=True)
    if not no_lag_keys.equals(lag_keys):
        raise RuntimeError("Lag and no-lag datasets do not contain identical station-day keys.")

    return without_lag_df, with_lag_df


def get_numeric_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
) -> list[str]:
    """Return all numeric feature columns except target and categorical columns."""
    if categorical_cols is None:
        categorical_cols = []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    return [
        col for col in numeric_cols
        if col != target_col and col not in categorical_cols
    ]


def _clean_feature_names(raw_feature_names: list[str]) -> list[str]:
    cleaned_names: list[str] = []

    for name in raw_feature_names:
        if name.startswith("num__"):
            cleaned_names.append(name.replace("num__", "", 1))
        elif name.startswith("cat__start_station_id_"):
            cleaned_names.append(name.replace("cat__start_station_id_", "station_", 1))
        else:
            cleaned_names.append(name)

    return cleaned_names


def encode_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """One-hot encode station IDs and keep context columns for residual analysis."""
    if categorical_cols is None:
        categorical_cols = [STATION_COL]

    numeric_cols = get_numeric_feature_columns(
        train_df,
        target_col=target_col,
        categorical_cols=categorical_cols,
    )

    X_train = train_df[numeric_cols + categorical_cols].copy()
    X_val = val_df[numeric_cols + categorical_cols].copy()
    X_test = test_df[numeric_cols + categorical_cols].copy()

    y_train = train_df[[target_col]].copy()
    y_val = val_df[[target_col]].copy()
    y_test = test_df[[target_col]].copy()

    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    X_train_ready = preprocessor.fit_transform(X_train)
    X_val_ready = preprocessor.transform(X_val)
    X_test_ready = preprocessor.transform(X_test)

    raw_feature_names = preprocessor.get_feature_names_out().tolist()
    feature_names = _clean_feature_names(raw_feature_names)

    train_encoded = pd.DataFrame(X_train_ready, columns=feature_names)
    val_encoded = pd.DataFrame(X_val_ready, columns=feature_names)
    test_encoded = pd.DataFrame(X_test_ready, columns=feature_names)

    def add_metadata(encoded_df: pd.DataFrame, original_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        encoded_df[target_col] = y_train[target_col].values if split_name == "train" else (
            y_val[target_col].values if split_name == "validation" else y_test[target_col].values
        )
        encoded_df["split"] = split_name
        encoded_df[RAW_STATION_CONTEXT_COL] = original_df[STATION_COL].astype(str).values
        return encoded_df

    train_encoded = add_metadata(train_encoded, train_df, "train")
    val_encoded = add_metadata(val_encoded, val_df, "validation")
    test_encoded = add_metadata(test_encoded, test_df, "test")

    return train_encoded, val_encoded, test_encoded, feature_names


def save_experiment_dataset(experiment: str, df: pd.DataFrame) -> None:
    output_dir = OUTPUT_BASE_DIR / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"daily_rentals_top20_reduced_{experiment}.csv"
    df.to_csv(raw_path, index=False)

    train_df, val_df, test_df = chronological_split(
        df=df,
        time_col=TIME_COL,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    train_encoded, val_encoded, test_encoded, feature_names = encode_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=TARGET_COL,
        categorical_cols=[STATION_COL],
    )

    train_encoded.to_csv(output_dir / "encoded_train.csv", index=False)
    val_encoded.to_csv(output_dir / "encoded_validation.csv", index=False)
    test_encoded.to_csv(output_dir / "encoded_test.csv", index=False)
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        output_dir / "encoded_feature_names.csv",
        index=False,
    )

    print(f"\nExperiment: {experiment}")
    print(f"  Raw dataset: {raw_path}")
    print(f"  Raw shape: {df.shape}")
    print(f"  Train encoded shape: {train_encoded.shape}")
    print(f"  Validation encoded shape: {val_encoded.shape}")
    print(f"  Test encoded shape: {test_encoded.shape}")
    print(f"  Number of final features: {len(feature_names)}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print("Loading reduced dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    validate_input_columns(df)
    print(f"Input dataset shape: {df.shape}")

    print("Creating lag and no-lag experiment datasets...")
    without_lag_df, with_lag_df = build_comparable_datasets(df)

    save_experiment_dataset("without_lag", without_lag_df)
    save_experiment_dataset("with_lag", with_lag_df)

    print("\nDone. Comparable encoded datasets were created.")


if __name__ == "__main__":
    main()
