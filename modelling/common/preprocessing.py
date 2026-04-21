from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def get_numeric_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [col for col in numeric_cols if col != target_col]


def split_X_y(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    X = df[feature_cols].copy()
    y = df[target_col].to_numpy()
    return X, y


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled
