from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
        f"{prefix}_median_ae": float(median_absolute_error(y_true, y_pred)),
        f"{prefix}_explained_variance": float(explained_variance_score(y_true, y_pred)),
        f"{prefix}_mape": safe_mape(y_true, y_pred),
    }
