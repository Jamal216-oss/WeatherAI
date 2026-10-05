import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "global_weather_clean.csv"
OUTPUT_FILE = BASE_DIR / "data" / "global_weather_precipitation_clean.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ADDING NEXT-HOUR PRECIPITATION TARGET")
print("=" * 70)

print(f"\nInput file: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Original records: {len(df):,}")
print(f"Original columns: {len(df.columns)}")


# ============================================================
# PREPARE TIME
# ============================================================

df["time"] = pd.to_datetime(df["time"])

# Sort separately by location and time
df = df.sort_values(
    ["location", "time"]
).reset_index(drop=True)


# ============================================================
# CREATE NEXT-HOUR PRECIPITATION
# ============================================================

print("\nCreating precipitation_next_hour...")

df["precipitation_next_hour"] = (
    df.groupby("location")["precipitation"]
    .shift(-1)
)


# ============================================================
# REMOVE LAST RECORD OF EACH LOCATION
# ============================================================

before = len(df)

df = df.dropna(
    subset=["precipitation_next_hour"]
).reset_index(drop=True)

removed = before - len(df)


# ============================================================
# VALIDATION
# ============================================================

print("\nValidation:")
print(f"Records before: {before:,}")
print(f"Records removed: {removed:,}")
print(f"Records after: {len(df):,}")

print(
    f"Locations: {df['location'].nunique()}"
)

print(
    f"Missing precipitation_next_hour: "
    f"{df['precipitation_next_hour'].isna().sum()}"
)


# Check that the target really represents the next hour
print("\nSample verification:")

sample = df[
    [
        "location",
        "time",
        "precipitation",
        "precipitation_next_hour"
    ]
].head(10)

print(sample.to_string(index=False))


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(f"\nSaved to:")
print(OUTPUT_FILE)

print(f"\nFinal records: {len(df):,}")
print(f"Final columns: {len(df.columns)}")
print("\nThe new target is:")
print("precipitation_next_hour")
print("=" * 70)