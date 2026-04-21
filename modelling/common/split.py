from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
