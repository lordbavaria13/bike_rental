from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RENT_FILE = RAW_DIR / "daily_rent_detail.csv"
STATION_FILE = RAW_DIR / "station_list.csv"
WEATHER_FILE = RAW_DIR / "weather.csv"

CHUNK_SIZE = 200_000
TOP_N = 20

RENT_USECOLS = [
    "started_at",
    "start_station_id",
    "start_station_name",
    "rideable_type",
    "member_casual",
]

WEATHER_USECOLS = [
    "datetime",
    "tempmax",
    "tempmin",
    "temp",
    "feelslikemax",
    "feelslikemin",
    "feelslike",
    "dew",
    "humidity",
    "precip",
    "precipprob",
    "precipcover",
    "snow",
    "snowdepth",
    "windgust",
    "windspeed",
    "winddir",
    "sealevelpressure",
    "cloudcover",
    "visibility",
    "solarradiation",
    "solarenergy",
    "uvindex",
    "severerisk",
    "conditions",
    "icon",
    "sunrise",
    "sunset",
]


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def get_top_n_start_stations() -> pd.DataFrame:
    station_counter: Counter[str] = Counter()
    station_name_lookup: dict[str, str] = {}

    for chunk in pd.read_csv(
        RENT_FILE,
        usecols=["start_station_id", "start_station_name"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk = chunk.dropna(subset=["start_station_id"])
        chunk["start_station_id"] = clean_string(chunk["start_station_id"])
        chunk["start_station_name"] = clean_string(chunk["start_station_name"])

        station_counter.update(chunk["start_station_id"].dropna())

        name_map = (
            chunk[["start_station_id", "start_station_name"]]
            .dropna()
            .drop_duplicates(subset=["start_station_id"])
        )
        for _, row in name_map.iterrows():
            station_name_lookup[str(row["start_station_id"])] = str(row["start_station_name"])

    top_df = (
        pd.DataFrame(station_counter.items(), columns=["start_station_id", "total_rentals"])
        .sort_values("total_rentals", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    top_df.insert(0, "rank", range(1, len(top_df) + 1))
    top_df["start_station_name"] = top_df["start_station_id"].map(station_name_lookup)
    return top_df


def load_station_list() -> pd.DataFrame:
    if not STATION_FILE.exists():
        return pd.DataFrame(columns=["station_id", "station_name"])

    df = pd.read_csv(STATION_FILE, low_memory=False)
    df["station_id"] = clean_string(df["station_id"])
    df["station_name"] = clean_string(df["station_name"])
    return df.drop_duplicates(subset=["station_id"])


def load_weather() -> pd.DataFrame:
    if not WEATHER_FILE.exists():
        return pd.DataFrame()

    weather = pd.read_csv(WEATHER_FILE, usecols=WEATHER_USECOLS, low_memory=False)
    weather["date"] = pd.to_datetime(weather["datetime"], errors="coerce").dt.date
    weather = weather.drop(columns=["datetime"])
    weather = weather.drop_duplicates(subset=["date"]).reset_index(drop=True)
    return weather


def build_daily_dataset(top_station_ids: set[str]) -> pd.DataFrame:
    grouped_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        RENT_FILE,
        usecols=RENT_USECOLS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk = chunk.dropna(
            subset=["started_at", "start_station_id", "rideable_type", "member_casual"]
        )

        chunk["start_station_id"] = clean_string(chunk["start_station_id"])
        chunk["start_station_name"] = clean_string(chunk["start_station_name"])
        chunk["rideable_type"] = clean_string(chunk["rideable_type"])
        chunk["member_casual"] = clean_string(chunk["member_casual"])

        chunk = chunk[chunk["start_station_id"].isin(top_station_ids)]
        if chunk.empty:
            continue

        chunk["started_at"] = pd.to_datetime(chunk["started_at"], errors="coerce")
        chunk = chunk.dropna(subset=["started_at"])
        if chunk.empty:
            continue

        chunk["date"] = chunk["started_at"].dt.date

        grouped = (
            chunk.groupby(
                ["date", "start_station_id", "start_station_name", "rideable_type", "member_casual"],
                as_index=False
            )
            .size()
            .rename(columns={"size": "rentals_count"})
        )

        grouped_parts.append(grouped)

    if not grouped_parts:
        return pd.DataFrame()

    long_df = pd.concat(grouped_parts, ignore_index=True)

    long_df = (
        long_df.groupby(
            ["date", "start_station_id", "start_station_name", "rideable_type", "member_casual"],
            as_index=False
        )["rentals_count"]
        .sum()
    )

    base_df = (
        long_df.groupby(
            ["date", "start_station_id", "start_station_name"],
            as_index=False
        )["rentals_count"]
        .sum()
        .rename(columns={"rentals_count": "total_rentals"})
    )

    long_df["combo_col"] = (
        long_df["rideable_type"].str.replace(" ", "_", regex=False)
        + "_"
        + long_df["member_casual"].str.replace(" ", "_", regex=False)
        + "_count"
    )

    combo_wide = (
        long_df.pivot_table(
            index=["date", "start_station_id", "start_station_name"],
            columns="combo_col",
            values="rentals_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    combo_wide.columns.name = None

    bike_wide = (
        long_df.pivot_table(
            index=["date", "start_station_id", "start_station_name"],
            columns="rideable_type",
            values="rentals_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    bike_wide.columns.name = None
    bike_wide = bike_wide.rename(
        columns={
            col: f"{str(col).replace(' ', '_')}_count"
            for col in bike_wide.columns
            if col not in ["date", "start_station_id", "start_station_name"]
        }
    )

    user_wide = (
        long_df.pivot_table(
            index=["date", "start_station_id", "start_station_name"],
            columns="member_casual",
            values="rentals_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    user_wide.columns.name = None
    user_wide = user_wide.rename(
        columns={
            col: f"{str(col).replace(' ', '_')}_count"
            for col in user_wide.columns
            if col not in ["date", "start_station_id", "start_station_name"]
        }
    )

    daily_df = base_df.merge(
        combo_wide,
        on=["date", "start_station_id", "start_station_name"],
        how="left",
    )
    daily_df = daily_df.merge(
        bike_wide,
        on=["date", "start_station_id", "start_station_name"],
        how="left",
    )
    daily_df = daily_df.merge(
        user_wide,
        on=["date", "start_station_id", "start_station_name"],
        how="left",
    )

    count_cols = [
        col for col in daily_df.columns
        if col.endswith("_count") or col == "total_rentals"
    ]
    for col in count_cols:
        daily_df[col] = daily_df[col].fillna(0).astype(int)

    daily_df = daily_df.sort_values(["start_station_id", "date"]).reset_index(drop=True)
    return daily_df


def save_outputs(top20_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    station_list = load_station_list()
    weather = load_weather()

    if not station_list.empty:
        daily_df = daily_df.merge(
            station_list,
            left_on="start_station_id",
            right_on="station_id",
            how="left",
        )
        daily_df = daily_df.drop(columns=["station_id"], errors="ignore")
        daily_df["station_name_final"] = daily_df["station_name"].fillna(daily_df["start_station_name"])
        daily_df = daily_df.drop(columns=["station_name"], errors="ignore")
    else:
        daily_df["station_name_final"] = daily_df["start_station_name"]

    if not weather.empty:
        daily_df = daily_df.merge(weather, on="date", how="left")

    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df["year"] = daily_df["date"].dt.year
    daily_df["month"] = daily_df["date"].dt.month
    daily_df["day"] = daily_df["date"].dt.day
    daily_df["weekday"] = daily_df["date"].dt.weekday
    daily_df["is_weekend"] = daily_df["weekday"].isin([5, 6]).astype(int)

    top20_df.to_csv(PROCESSED_DIR / "top_20_stations.csv", index=False)

    csv_path = PROCESSED_DIR / "daily_rentals_top20.csv.gz"
    daily_df.to_csv(csv_path, index=False, compression="gzip")

    try:
        parquet_path = PROCESSED_DIR / "daily_rentals_top20.parquet"
        daily_df.to_parquet(parquet_path, index=False)
        print(f"Saved parquet: {parquet_path}")
    except Exception as exc:
        print(f"Parquet export skipped: {exc}")

    print(f"Saved top-20 station list: {PROCESSED_DIR / 'top_20_stations.csv'}")
    print(f"Saved processed daily dataset: {csv_path}")
    print("\nPreview:")
    print(daily_df.head(10).to_string(index=False))


def main() -> None:
    ensure_dirs()

    print("Step 1/3: Finding top 20 start stations...")
    top20_df = get_top_n_start_stations()
    print(top20_df.to_string(index=False))

    top_station_ids = set(top20_df["start_station_id"].astype(str))

    print("\nStep 2/3: Building daily dataset for top 20 stations...")
    daily_df = build_daily_dataset(top_station_ids)

    print("\nStep 3/3: Merging station names and weather, then saving outputs...")
    save_outputs(top20_df, daily_df)

    print("\nDone.")


if __name__ == "__main__":
    main()