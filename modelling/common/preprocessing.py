from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelling.common.config import (
    CONTEXT_COLUMNS,
    RAW_STATION_CONTEXT_COL,
    STATION_COL,
    TARGET_COL,
    TIME_COL,
    get_experiment_paths,
)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a modelling dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def get_numeric_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
) -> list[str]:
    """Return numeric input features, excluding target, categorical and context columns."""
    if categorical_cols is None:
        categorical_cols = []

    excluded = {target_col, *categorical_cols, *CONTEXT_COLUMNS, "split"}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    return [col for col in numeric_cols if col not in excluded]


def split_X_y(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
):
    """Split a dataframe into X and y."""
    X = df[feature_cols].copy()
    y = df[target_col].to_numpy()
    return X, y


def scale_features(X_train, X_val, X_test):
    """Scale train/validation/test feature matrices with one StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def _clean_feature_names(raw_feature_names: list[str]) -> list[str]:
    """Clean ColumnTransformer feature names for readable outputs."""
    cleaned_names: list[str] = []

    for name in raw_feature_names:
        if name.startswith("num__"):
            cleaned_names.append(name.replace("num__", "", 1))
        elif name.startswith("cat__start_station_id_"):
            cleaned_names.append(name.replace("cat__start_station_id_", "station_", 1))
        else:
            cleaned_names.append(name)

    return cleaned_names


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
    scale_numeric: bool = False,
):
    """
    Prepare raw split dataframes for modelling.

    This is kept for fallback use. In the main pipeline, encoded datasets from
    src/scripts/05_create_encoded_dataset.py are preferred.
    """
    if categorical_cols is None:
        categorical_cols = [STATION_COL]

    missing_categorical_cols = [col for col in categorical_cols if col not in train_df.columns]
    if missing_categorical_cols:
        raise ValueError(f"Missing categorical columns in training data: {missing_categorical_cols}")

    numeric_cols = get_numeric_feature_columns(
        train_df,
        target_col=target_col,
        categorical_cols=categorical_cols,
    )

    X_train = train_df[numeric_cols + categorical_cols].copy()
    X_val = val_df[numeric_cols + categorical_cols].copy()
    X_test = test_df[numeric_cols + categorical_cols].copy()

    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    X_train_ready = preprocessor.fit_transform(X_train)
    X_val_ready = preprocessor.transform(X_val)
    X_test_ready = preprocessor.transform(X_test)

    feature_names = _clean_feature_names(preprocessor.get_feature_names_out().tolist())

    return (
        preprocessor,
        feature_names,
        X_train_ready,
        X_val_ready,
        X_test_ready,
        y_train,
        y_val,
        y_test,
    )


def load_encoded_datasets(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load already encoded train/validation/test datasets."""
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Encoded dataset not found: {path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if target_col not in df.columns:
            raise ValueError(f"{name} encoded dataset is missing target column: {target_col}")

    return train_df, val_df, test_df


def load_encoded_datasets_for_experiment(
    experiment: str,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load encoded train/validation/test datasets for with_lag or without_lag."""
    paths = get_experiment_paths(experiment)
    return load_encoded_datasets(
        train_path=paths["encoded_train_path"],
        val_path=paths["encoded_val_path"],
        test_path=paths["encoded_test_path"],
        target_col=target_col,
    )


def prepare_encoded_feature_matrices_for_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    feature_names: list[str] | None = None,
    scale_numeric: bool = False,
) -> tuple[StandardScaler | None, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare X matrices from already one-hot encoded datasets."""
    if feature_names is None:
        exclude_cols = {target_col, "split", *CONTEXT_COLUMNS}
        feature_names = [col for col in train_df.columns if col not in exclude_cols]

    missing_features = [col for col in feature_names if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing encoded feature columns: {missing_features}")

    X_train = train_df[feature_names].values
    X_val = val_df[feature_names].values
    X_test = test_df[feature_names].values

    if scale_numeric:
        scaler, X_train_ready, X_val_ready, X_test_ready = scale_features(X_train, X_val, X_test)
    else:
        scaler = None
        X_train_ready = X_train
        X_val_ready = X_val
        X_test_ready = X_test

    return scaler, feature_names, X_train_ready, X_val_ready, X_test_ready


def extract_prediction_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return context columns used in predictions.csv."""
    context = pd.DataFrame(index=df.index)

    if TIME_COL in df.columns:
        context[TIME_COL] = df[TIME_COL].values
    else:
        context[TIME_COL] = pd.NA

    if RAW_STATION_CONTEXT_COL in df.columns:
        context[STATION_COL] = df[RAW_STATION_CONTEXT_COL].astype(str).values
    elif STATION_COL in df.columns:
        context[STATION_COL] = df[STATION_COL].astype(str).values
    else:
        context[STATION_COL] = pd.NA

    if TARGET_COL in df.columns:
        context[TARGET_COL] = df[TARGET_COL].values

    return context.reset_index(drop=True)
