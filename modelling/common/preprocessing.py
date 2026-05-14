from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the final modelling dataset from disk.

    We raise an error immediately if the file does not exist,
    because all model scripts depend on this file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, low_memory=False)


def get_numeric_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
) -> list[str]:
    """
    Return all numeric feature columns that should be used as model inputs.

    Important:
    - the target column is removed
    - categorical columns like start_station_id are removed here,
      because they will be encoded separately
    """
    if categorical_cols is None:
        categorical_cols = []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    return [
        col for col in numeric_cols
        if col != target_col and col not in categorical_cols
    ]


def split_X_y(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
):
    """
    Simple helper to split a dataframe into X and y.

    We keep this function because some older scripts may still use it.
    """
    X = df[feature_cols].copy()
    y = df[target_col].to_numpy()
    return X, y


def scale_features(X_train, X_val, X_test):
    """
    Scale three feature matrices with one shared scaler.

    The scaler is fitted only on the training data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def _clean_feature_names(raw_feature_names: list[str]) -> list[str]:
    """
    Clean feature names returned by ColumnTransformer.

    This makes plots and saved files easier to read.
    """
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
    Prepare final feature matrices for model training.

    What this function does:
    1. Split target from features
    2. Use numeric columns as normal model inputs
    3. Treat start_station_id as a categorical feature
    4. Fit preprocessing only on the training set
    5. Transform validation and test with the same fitted preprocessor

    Why this is important:
    - start_station_id is an ID, not a real numeric quantity
    - therefore we use one-hot encoding instead of raw numeric values
    - fitting only on train keeps the setup methodologically clean

    Parameters
    ----------
    train_df, val_df, test_df:
        Chronologically split dataframes.

    target_col:
        Name of the prediction target column.

    categorical_cols:
        Columns that should be one-hot encoded.
        Default: ["start_station_id"]

    scale_numeric:
        If True, numeric features are scaled with StandardScaler.
        This should be used for models like:
        - Linear Regression
        - Ridge
        - Lasso
        - KNN
        - Neural Network

        For tree-based models this should usually stay False.

    Returns
    -------
    preprocessor:
        Fitted ColumnTransformer

    feature_names:
        Final transformed feature names after encoding

    X_train_ready, X_val_ready, X_test_ready:
        Prepared feature matrices

    y_train, y_val, y_test:
        Target arrays
    """
    if categorical_cols is None:
        categorical_cols = ["start_station_id"]

    # Check that all categorical columns really exist.
    missing_categorical_cols = [
        col for col in categorical_cols if col not in train_df.columns
    ]
    if missing_categorical_cols:
        raise ValueError(
            f"Missing categorical columns in training data: {missing_categorical_cols}"
        )

    # Get all numeric features except the target and the categorical columns.
    numeric_cols = get_numeric_feature_columns(
        train_df,
        target_col=target_col,
        categorical_cols=categorical_cols,
    )

    # Build raw X dataframes.
    # We copy them to avoid modifying the original train/val/test dataframes.
    X_train = train_df[numeric_cols + categorical_cols].copy()
    X_val = val_df[numeric_cols + categorical_cols].copy()
    X_test = test_df[numeric_cols + categorical_cols].copy()

    # Extract targets.
    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    # Convert categorical ID columns to string before encoding.
    # This avoids treating station IDs like regular numeric values.
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Scale numeric features only if the model type needs it.
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    # Build the preprocessing pipeline.
    # Numeric columns are either scaled or passed through.
    # Station IDs are one-hot encoded.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
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

    # Fit only on the training data.
    X_train_ready = preprocessor.fit_transform(X_train)
    X_val_ready = preprocessor.transform(X_val)
    X_test_ready = preprocessor.transform(X_test)

    # Get readable names for all transformed features.
    raw_feature_names = preprocessor.get_feature_names_out().tolist()
    feature_names = _clean_feature_names(raw_feature_names)

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