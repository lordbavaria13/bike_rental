from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "daily_rentals_top20.csv.gz"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

STATION_ID_COL = "start_station_id"
NAME_COLS = ["start_station_name", "station_name_final"]
TARGET_COL = "total_rentals"

FEATURE_COLS = [
    "start_station_id",
    "tempmax",
    "humidity",
    "precip",
    "precipcover",
    "cloudcover",
    "windspeed",
    "visibility",
    "sealevelpressure",
    "uvindex",
    "sunset_minutes",
    "month",
    "weekday",
    "year",
    "time_idx",
    "snow",
    "snowdepth",
]


def parse_time_to_minutes(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), errors="coerce")
    return parsed.dt.hour * 60 + parsed.dt.minute


def add_numeric_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "sunset" in df.columns and "sunset_minutes" not in df.columns:
        df["sunset_minutes"] = parse_time_to_minutes(df["sunset"])

    min_date = df["date"].min()
    df["time_idx"] = (df["date"] - min_date).dt.days.astype("float")

    return df


def build_station_mapping(df: pd.DataFrame) -> pd.DataFrame:
    available_name_cols = [col for col in NAME_COLS if col in df.columns]

    mapping_cols = [STATION_ID_COL] + available_name_cols
    mapping_df = (
        df[mapping_cols]
        .drop_duplicates(subset=[STATION_ID_COL])
        .sort_values(STATION_ID_COL)
        .reset_index(drop=True)
    )

    mapping_df[STATION_ID_COL] = pd.to_numeric(mapping_df[STATION_ID_COL], errors="coerce")
    return mapping_df


def build_reduced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = add_numeric_time_features(df)

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    reduced_df = df[FEATURE_COLS + [TARGET_COL]].copy()

    reduced_df[STATION_ID_COL] = pd.to_numeric(reduced_df[STATION_ID_COL], errors="coerce")

    numeric_cols = reduced_df.select_dtypes(include=["number"]).columns.tolist()
    reduced_df = reduced_df[numeric_cols].copy()

    final_cols = [col for col in FEATURE_COLS if col in reduced_df.columns] + [TARGET_COL]
    reduced_df = reduced_df[final_cols]

    reduced_df = reduced_df.dropna(subset=[STATION_ID_COL, TARGET_COL]).reset_index(drop=True)

    return reduced_df


def save_outputs(
    reduced_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    original_columns: list[str],
    reduced_columns: list[str],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reduced_csv_gz_path = OUTPUT_DIR / "daily_rentals_top20_reduced.csv.gz"
    reduced_csv_path = OUTPUT_DIR / "daily_rentals_top20_reduced.csv"
    mapping_csv_path = OUTPUT_DIR / "station_id_name_mapping.csv"
    removed_cols_path = OUTPUT_DIR / "daily_rentals_top20_removed_columns.csv"
    summary_path = OUTPUT_DIR / "daily_rentals_top20_reduced_summary.csv"

    reduced_df.to_csv(reduced_csv_gz_path, index=False, compression="gzip")
    reduced_df.to_csv(reduced_csv_path, index=False)
    mapping_df.to_csv(mapping_csv_path, index=False)

    try:
        reduced_parquet_path = OUTPUT_DIR / "daily_rentals_top20_reduced.parquet"
        reduced_df.to_parquet(reduced_parquet_path, index=False)
        print(f"Saved parquet: {reduced_parquet_path}")
    except Exception as exc:
        print(f"Parquet export skipped: {exc}")

    removed_columns = [col for col in original_columns if col not in reduced_columns]
    removed_df = pd.DataFrame({"removed_column": removed_columns})
    removed_df.to_csv(removed_cols_path, index=False)

    summary_df = pd.DataFrame(
        {
            "metric": [
                "n_rows",
                "n_original_columns",
                "n_reduced_columns",
                "n_removed_columns",
            ],
            "value": [
                len(reduced_df),
                len(original_columns),
                len(reduced_columns),
                len(removed_columns),
            ],
        }
    )
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved reduced dataset (gz): {reduced_csv_gz_path}")
    print(f"Saved reduced dataset (csv): {reduced_csv_path}")
    print(f"Saved station mapping: {mapping_csv_path}")
    print(f"Saved removed-column list: {removed_cols_path}")
    print(f"Saved summary: {summary_path}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print("Loading processed dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"Original shape: {df.shape}")

    mapping_df = build_station_mapping(df)
    reduced_df = build_reduced_dataset(df)

    print(f"Reduced shape: {reduced_df.shape}")

    save_outputs(
        reduced_df=reduced_df,
        mapping_df=mapping_df,
        original_columns=df.columns.tolist(),
        reduced_columns=reduced_df.columns.tolist(),
    )

    print("\nFinal feature columns:")
    print(pd.Series(FEATURE_COLS).to_string(index=False))

    print("\nFinal target column:")
    print(TARGET_COL)

    print("\nPreview:")
    print(reduced_df.head(10).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()