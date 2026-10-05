import os
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "data/singida_weather_historical.csv"
OUTPUT_FILE = "data/singida_weather_clean.csv"


# ==========================================
# LOAD DATA
# ==========================================

print("==========================================")
print("       WEATHER DATA PREPARATION")
print("==========================================")
print()

print("Loading dataset...")

weather_data = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print()


# ==========================================
# CONVERT TIME COLUMN
# ==========================================

print("Converting time column...")

weather_data["time"] = pd.to_datetime(
    weather_data["time"],
    errors="coerce"
)

print("Time conversion completed.")
print()


# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("Checking missing values...")

missing_before = weather_data.isnull().sum()

print(missing_before)
print()


# ==========================================
# REMOVE INVALID TIME RECORDS
# ==========================================

invalid_time = weather_data["time"].isnull().sum()

print("Invalid time records:", invalid_time)

if invalid_time > 0:
    weather_data = weather_data.dropna(
        subset=["time"]
    )

print()


# ==========================================
# REMOVE DUPLICATE RECORDS
# ==========================================

duplicates_before = weather_data.duplicated().sum()

print("Duplicate records found:", duplicates_before)

if duplicates_before > 0:
    weather_data = weather_data.drop_duplicates()

print()


# ==========================================
# SORT BY TIME
# ==========================================

print("Sorting data by time...")

weather_data = weather_data.sort_values(
    by="time"
).reset_index(drop=True)

print("Data sorted successfully.")
print()


# ==========================================
# HANDLE MISSING WEATHER VALUES
# ==========================================

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

print("Checking weather variables...")

for column in weather_columns:

    missing_count = weather_data[column].isnull().sum()

    if missing_count > 0:

        print(
            f"Filling {missing_count} missing values in {column}"
        )

        weather_data[column] = (
            weather_data[column]
            .interpolate()
            .ffill()
            .bfill()
        )

print()


# ==========================================
# CREATE TIME FEATURES
# ==========================================

print("Creating time features...")

weather_data["hour"] = (
    weather_data["time"].dt.hour
)

weather_data["day"] = (
    weather_data["time"].dt.day
)

weather_data["month"] = (
    weather_data["time"].dt.month
)

weather_data["year"] = (
    weather_data["time"].dt.year
)

weather_data["day_of_week"] = (
    weather_data["time"].dt.dayofweek
)

print("Time features created.")
print()


# ==========================================
# CREATE FUTURE WEATHER TARGETS
# ==========================================

print("Creating prediction targets...")

# Next-hour temperature
weather_data["temperature_next_hour"] = (
    weather_data["temperature_2m"].shift(-1)
)

# Next-hour rain
weather_data["rain_next_hour"] = (
    weather_data["rain"].shift(-1)
)

print("Prediction targets created.")
print()


# ==========================================
# REMOVE LAST RECORD
# ==========================================

# The last row does not have a future hour
weather_data = weather_data.dropna(
    subset=[
        "temperature_next_hour",
        "rain_next_hour"
    ]
)

print("Removed records without future targets.")
print()


# ==========================================
# SAVE CLEAN DATASET
# ==========================================

os.makedirs(
    "data",
    exist_ok=True
)

weather_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# FINAL REPORT
# ==========================================

print("==========================================")
print("       DATA PREPARATION COMPLETED")
print("==========================================")
print()

print(
    "Final number of records:",
    len(weather_data)
)

print(
    "Final number of columns:",
    len(weather_data.columns)
)

print()

print("Final columns:")

for column in weather_data.columns:
    print("-", column)

print()

print("Missing values after cleaning:")

print(
    weather_data.isnull().sum()
)

print()

print("Clean dataset saved at:")

print(
    os.path.abspath(OUTPUT_FILE)
)

print()

print("Ready for machine learning.")