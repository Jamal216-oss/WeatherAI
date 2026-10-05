import os
import pandas as pd
import numpy as np

# ============================================================
# WEATHER AI - GLOBAL DATA PREPARATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "global_weather_historical.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "global_weather_clean.csv"
)

print()
print("=" * 70)
print("              WEATHER AI")
print("       GLOBAL DATA PREPARATION")
print("=" * 70)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print()
print("Loading global dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial records: {len(df):,}")

# ------------------------------------------------------------
# Convert time
# ------------------------------------------------------------

df["time"] = pd.to_datetime(
    df["time"],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=["time"])

# ------------------------------------------------------------
# Sort data
# ------------------------------------------------------------

df = df.sort_values(
    ["location", "time"]
).reset_index(drop=True)

# ------------------------------------------------------------
# Remove duplicates
# ------------------------------------------------------------

duplicates = df.duplicated(
    subset=["location", "time"]
).sum()

print(f"Duplicate records found: {duplicates:,}")

df = df.drop_duplicates(
    subset=["location", "time"]
)

# ------------------------------------------------------------
# Weather columns
# ------------------------------------------------------------

weather_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m"
]

# ------------------------------------------------------------
# Handle missing values
# ------------------------------------------------------------

print()
print("Checking missing values...")

for column in weather_columns:

    missing_before = df[column].isna().sum()

    if missing_before > 0:

        print(
            f"{column}: "
            f"{missing_before:,} missing values"
        )

        # Interpolate separately for every location
        df[column] = (
            df.groupby("location")[column]
            .transform(
                lambda x: x.interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )
        )

# ------------------------------------------------------------
# Time features
# ------------------------------------------------------------

print()
print("Creating time features...")

df["hour"] = df["time"].dt.hour

df["day"] = df["time"].dt.day

df["month"] = df["time"].dt.month

df["year"] = df["time"].dt.year

df["day_of_week"] = df["time"].dt.dayofweek

# Day of year
df["day_of_year"] = df["time"].dt.dayofyear

# ------------------------------------------------------------
# Cyclic time features
# ------------------------------------------------------------

# These help the model understand that:
# 23:00 and 00:00 are close together.
# December and January are also close in the yearly cycle.

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365
)

# ------------------------------------------------------------
# Future targets
# ------------------------------------------------------------

print()
print("Creating prediction targets...")

# Next hour temperature
df["temperature_next_hour"] = (
    df.groupby("location")["temperature_2m"]
    .shift(-1)
)

# Next hour rain
df["rain_next_hour"] = (
    df.groupby("location")["rain"]
    .shift(-1)
)

# ------------------------------------------------------------
# Remove rows without targets
# ------------------------------------------------------------

df = df.dropna(
    subset=[
        "temperature_next_hour",
        "rain_next_hour"
    ]
)

# ------------------------------------------------------------
# Final missing-value check
# ------------------------------------------------------------

remaining_missing = df.isna().sum().sum()

print()
print(
    f"Remaining missing values: "
    f"{remaining_missing:,}"
)

# ------------------------------------------------------------
# Save cleaned dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print()
print("=" * 70)
print("GLOBAL DATA PREPARATION COMPLETE")
print("=" * 70)

print(f"Final records : {len(df):,}")
print(f"Locations     : {df['location'].nunique()}")
print(f"Features      : {len(df.columns)}")

print()
print("Locations:")
print(
    df["location"]
    .value_counts()
    .sort_index()
)

print()
print("Output file:")
print(OUTPUT_FILE)

print()
print("Columns:")
print(list(df.columns))

print()
print("=" * 70)