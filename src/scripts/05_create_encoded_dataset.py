from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "daily_rentals_top20_reduced.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

TARGET_COL = "total_rentals"
TIME_COL = "time_idx"
STATION_COL = "start_station_id"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def chronological_split(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically."""
    import numpy as np
    
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


def get_numeric_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
) -> list[str]:
    """Get all numeric feature columns."""
    if categorical_cols is None:
        categorical_cols = []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    return [
        col for col in numeric_cols
        if col != target_col and col not in categorical_cols
    ]


def _clean_feature_names(raw_feature_names: list[str]) -> list[str]:
    """Clean feature names returned by ColumnTransformer."""
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
    """
    Encode features (one-hot encode categorical columns).
    
    Returns:
    - encoded train dataframe
    - encoded validation dataframe  
    - encoded test dataframe
    - list of final feature names
    """
    if categorical_cols is None:
        categorical_cols = ["start_station_id"]

    # Get numeric features
    numeric_cols = get_numeric_feature_columns(
        train_df,
        target_col=target_col,
        categorical_cols=categorical_cols,
    )

    # Build raw X dataframes
    X_train = train_df[numeric_cols + categorical_cols].copy()
    X_val = val_df[numeric_cols + categorical_cols].copy()
    X_test = test_df[numeric_cols + categorical_cols].copy()

    # Extract targets
    y_train = train_df[[target_col]].copy()
    y_val = val_df[[target_col]].copy()
    y_test = test_df[[target_col]].copy()

    # Convert categorical ID columns to string
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Build preprocessing pipeline
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

    # Fit only on training data
    X_train_ready = preprocessor.fit_transform(X_train)
    X_val_ready = preprocessor.transform(X_val)
    X_test_ready = preprocessor.transform(X_test)

    # Get cleaned feature names
    raw_feature_names = preprocessor.get_feature_names_out().tolist()
    feature_names = _clean_feature_names(raw_feature_names)

    # Create dataframes with encoded features
    train_encoded = pd.DataFrame(X_train_ready, columns=feature_names)
    val_encoded = pd.DataFrame(X_val_ready, columns=feature_names)
    test_encoded = pd.DataFrame(X_test_ready, columns=feature_names)

    # Add target column
    train_encoded[target_col] = y_train[target_col].values
    val_encoded[target_col] = y_val[target_col].values
    test_encoded[target_col] = y_test[target_col].values

    # Add split identifier
    train_encoded["split"] = "train"
    val_encoded["split"] = "validation"
    test_encoded["split"] = "test"

    return train_encoded, val_encoded, test_encoded, feature_names


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print("Loading reduced dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Dataset shape: {df.shape}")

    print("Performing chronological split...")
    train_df, val_df, test_df = chronological_split(
        df=df,
        time_col=TIME_COL,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    print("Encoding features (one-hot encoding of station IDs)...")
    train_encoded, val_encoded, test_encoded, feature_names = encode_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=TARGET_COL,
        categorical_cols=["start_station_id"],
    )

    print(f"Final number of features: {len(feature_names)}")
    print(f"Train encoded shape: {train_encoded.shape}")
    print(f"Validation encoded shape: {val_encoded.shape}")
    print(f"Test encoded shape: {test_encoded.shape}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save encoded datasets
    train_path = OUTPUT_DIR / "encoded_train.csv"
    val_path = OUTPUT_DIR / "encoded_validation.csv"
    test_path = OUTPUT_DIR / "encoded_test.csv"
    features_path = OUTPUT_DIR / "encoded_feature_names.csv"

    train_encoded.to_csv(train_path, index=False)
    val_encoded.to_csv(val_path, index=False)
    test_encoded.to_csv(test_path, index=False)

    pd.DataFrame({"feature": feature_names}).to_csv(features_path, index=False)

    print(f"\nSaved encoded train dataset: {train_path}")
    print(f"Saved encoded validation dataset: {val_path}")
    print(f"Saved encoded test dataset: {test_path}")
    print(f"Saved feature names: {features_path}")

    print("\nFirst few rows of encoded train dataset:")
    print(train_encoded.head())

    print("\nDone.")


if __name__ == "__main__":
    main()
