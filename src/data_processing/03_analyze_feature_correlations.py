from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "daily_rentals_top20.csv.gz"
OUT_DIR = BASE_DIR / "data" / "processed" / "analysis"

CORR_THRESHOLD = 0.85

ID_COLS = [
    "date",
    "start_station_id",
    "start_station_name",
    "station_name_final",
]

# These are not target variables, but we may want to keep them even if correlated
PREFERRED_FEATURES = {
    "year",
    "month",
    "day",
    "weekday",
    "is_weekend",
    "sunrise_minutes",
    "sunset_minutes",
    "daylight_minutes",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH, low_memory=False)


def parse_time_to_minutes(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), errors="coerce")
    return parsed.dt.hour * 60 + parsed.dt.minute


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "sunrise" in df.columns:
        df["sunrise_minutes"] = parse_time_to_minutes(df["sunrise"])

    if "sunset" in df.columns:
        df["sunset_minutes"] = parse_time_to_minutes(df["sunset"])

    if "sunrise_minutes" in df.columns and "sunset_minutes" in df.columns:
        df["daylight_minutes"] = df["sunset_minutes"] - df["sunrise_minutes"]

    return df


def detect_target_columns(columns: Iterable[str]) -> list[str]:
    targets = []
    for col in columns:
        if col == "total_rentals" or col.endswith("_count"):
            targets.append(col)
    return sorted(targets)


def get_numeric_predictors(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = set(target_cols) | set(ID_COLS)
    predictors = [col for col in numeric_cols if col not in excluded]
    return sorted(predictors)


def build_high_corr_pairs(corr_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    abs_corr = corr_df.abs()
    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        for idx in upper.index:
            value = upper.loc[idx, col]
            if pd.notna(value) and value >= threshold:
                pairs.append(
                    {
                        "feature_1": idx,
                        "feature_2": col,
                        "abs_correlation": float(value),
                        "correlation": float(corr_df.loc[idx, col]),
                    }
                )

    pairs_df = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False)
    return pairs_df


def choose_drop_recommendations(
    pairs_df: pd.DataFrame,
    predictor_target_corr: pd.Series,
) -> pd.DataFrame:
    recommendations = []
    dropped = set()

    for _, row in pairs_df.iterrows():
        f1 = row["feature_1"]
        f2 = row["feature_2"]

        if f1 in dropped or f2 in dropped:
            continue

        score1 = float(predictor_target_corr.get(f1, 0.0))
        score2 = float(predictor_target_corr.get(f2, 0.0))

        # Prefer to keep calendar/time-engineered features if possible
        if f1 in PREFERRED_FEATURES and f2 not in PREFERRED_FEATURES:
            drop_col = f2
            keep_col = f1
            reason = "kept preferred feature"
        elif f2 in PREFERRED_FEATURES and f1 not in PREFERRED_FEATURES:
            drop_col = f1
            keep_col = f2
            reason = "kept preferred feature"
        else:
            # Keep the feature with stronger relation to targets
            if score1 >= score2:
                keep_col = f1
                drop_col = f2
            else:
                keep_col = f2
                drop_col = f1
            reason = "kept feature with stronger target correlation"

        dropped.add(drop_col)

        recommendations.append(
            {
                "drop_feature": drop_col,
                "keep_feature": keep_col,
                "abs_feature_feature_corr": float(row["abs_correlation"]),
                "drop_feature_target_score": float(predictor_target_corr.get(drop_col, 0.0)),
                "keep_feature_target_score": float(predictor_target_corr.get(keep_col, 0.0)),
                "reason": reason,
            }
        )

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "drop_feature",
                "keep_feature",
                "abs_feature_feature_corr",
                "drop_feature_target_score",
                "keep_feature_target_score",
                "reason",
            ]
        )

    return pd.DataFrame(recommendations)


def save_heatmap(corr_df: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_df, cmap="coolwarm", center=0)
    plt.title("Predictor Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "predictor_correlation_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()

    print("Loading processed dataset...")
    df = load_dataset()
    df = add_time_features(df)

    print(f"Dataset shape: {df.shape}")

    target_cols = detect_target_columns(df.columns)
    predictor_cols = get_numeric_predictors(df, target_cols)

    if not predictor_cols:
        raise ValueError("No numeric predictor columns found.")

    print(f"Detected target columns: {len(target_cols)}")
    print(f"Detected numeric predictor columns: {len(predictor_cols)}")

    # Summary of predictors
    predictor_summary = pd.DataFrame(
        {
            "feature": predictor_cols,
            "dtype": [str(df[col].dtype) for col in predictor_cols],
            "missing_values": [int(df[col].isna().sum()) for col in predictor_cols],
            "n_unique": [int(df[col].nunique(dropna=True)) for col in predictor_cols],
        }
    )
    predictor_summary.to_csv(OUT_DIR / "predictor_summary.csv", index=False)

    # Predictor-predictor correlation
    predictor_df = df[predictor_cols].copy()
    corr_df = predictor_df.corr(method="pearson")
    corr_df.to_csv(OUT_DIR / "predictor_correlation_matrix.csv")

    save_heatmap(corr_df)

    # High-correlation pairs
    pairs_df = build_high_corr_pairs(corr_df, CORR_THRESHOLD)
    pairs_df.to_csv(OUT_DIR / "high_correlation_pairs.csv", index=False)

    # Predictor-target correlation
    predictor_target_rows = []
    if target_cols:
        for target in target_cols:
            if pd.api.types.is_numeric_dtype(df[target]):
                merged = df[predictor_cols + [target]].corr(method="pearson")[target].drop(target)
                for feature, corr_value in merged.items():
                    predictor_target_rows.append(
                        {
                            "target": target,
                            "feature": feature,
                            "correlation": float(corr_value) if pd.notna(corr_value) else np.nan,
                            "abs_correlation": abs(float(corr_value)) if pd.notna(corr_value) else np.nan,
                        }
                    )

    predictor_target_df = pd.DataFrame(predictor_target_rows)

    if not predictor_target_df.empty:
        predictor_target_df = predictor_target_df.sort_values(
            ["target", "abs_correlation"],
            ascending=[True, False],
        )
        predictor_target_df.to_csv(OUT_DIR / "predictor_target_correlations.csv", index=False)

        predictor_target_score = (
            predictor_target_df.groupby("feature")["abs_correlation"].max().fillna(0.0)
        )
    else:
        predictor_target_score = pd.Series(0.0, index=predictor_cols, dtype=float)
        predictor_target_df.to_csv(OUT_DIR / "predictor_target_correlations.csv", index=False)

    # Recommended drops
    recommendations_df = choose_drop_recommendations(pairs_df, predictor_target_score)
    recommendations_df.to_csv(OUT_DIR / "recommended_drop_columns.csv", index=False)

    print("\nDone.")
    print(f"Outputs written to: {OUT_DIR}")

    print("\nTop highly correlated predictor pairs:")
    if pairs_df.empty:
        print("  None found above threshold.")
    else:
        print(pairs_df.head(15).to_string(index=False))

    print("\nRecommended columns to drop:")
    if recommendations_df.empty:
        print("  No drop recommendations.")
    else:
        print(recommendations_df.to_string(index=False))


if __name__ == "__main__":
    main()