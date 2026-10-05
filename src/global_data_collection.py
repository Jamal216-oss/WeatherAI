import os
import requests
import pandas as pd
import time

# ============================================================
# WEATHER AI - GLOBAL HISTORICAL DATA COLLECTION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------
# Global training locations
# ------------------------------------------------------------

locations = [
    {
        "name": "Dar es Salaam",
        "country": "Tanzania",
        "latitude": -6.8235,
        "longitude": 39.2695
    },
    {
        "name": "Nairobi",
        "country": "Kenya",
        "latitude": -1.2864,
        "longitude": 36.8172
    },
    {
        "name": "Johannesburg",
        "country": "South Africa",
        "latitude": -26.2041,
        "longitude": 28.0473
    },
    {
        "name": "Cairo",
        "country": "Egypt",
        "latitude": 30.0444,
        "longitude": 31.2357
    },
    {
        "name": "London",
        "country": "United Kingdom",
        "latitude": 51.5074,
        "longitude": -0.1278
    },
    {
        "name": "New York",
        "country": "United States",
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    {
        "name": "Tokyo",
        "country": "Japan",
        "latitude": 35.6762,
        "longitude": 139.6503
    },
    {
        "name": "Sydney",
        "country": "Australia",
        "latitude": -33.8688,
        "longitude": 151.2093
    },
    {
        "name": "Mumbai",
        "country": "India",
        "latitude": 19.0760,
        "longitude": 72.8777
    },
    {
        "name": "São Paulo",
        "country": "Brazil",
        "latitude": -23.5505,
        "longitude": -46.6333
    }
]

# ------------------------------------------------------------
# API configuration
# ------------------------------------------------------------

API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

HOURLY_VARIABLES = [
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
# Download one location
# ------------------------------------------------------------

def collect_location(location):

    print()
    print("=" * 60)
    print(f"Downloading: {location['name']}, {location['country']}")
    print("=" * 60)

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "auto"
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        if "hourly" not in data:
            print("ERROR: No hourly data returned.")
            return None

        hourly = data["hourly"]

        df = pd.DataFrame(hourly)

        # Add location information
        df["location"] = location["name"]
        df["country"] = location["country"]
        df["latitude"] = location["latitude"]
        df["longitude"] = location["longitude"]

        print(f"Records downloaded: {len(df):,}")

        return df

    except Exception as e:

        print(f"ERROR downloading {location['name']}: {e}")

        return None


# ------------------------------------------------------------
# Main collection process
# ------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("              WEATHER AI")
    print("        GLOBAL DATA COLLECTION")
    print("=" * 70)

    all_data = []

    for location in locations:

        df = collect_location(location)

        if df is not None:
            all_data.append(df)

        # Small delay between requests
        time.sleep(2)

    if not all_data:

        print()
        print("No data was collected.")
        return

    print()
    print("Combining datasets...")

    global_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Convert time
    global_df["time"] = pd.to_datetime(
        global_df["time"],
        errors="coerce"
    )

    # Remove invalid timestamps
    global_df = global_df.dropna(
        subset=["time"]
    )

    # Sort
    global_df = global_df.sort_values(
        ["location", "time"]
    ).reset_index(drop=True)

    # Save
    output_file = os.path.join(
        DATA_DIR,
        "global_weather_historical.csv"
    )

    global_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)

    print(f"Total records : {len(global_df):,}")
    print(f"Locations     : {global_df['location'].nunique()}")
    print(f"Output file   : {output_file}")

    print()
    print("Records per location:")
    print(
        global_df["location"]
        .value_counts()
        .sort_index()
    )

    print()
    print("First records:")
    print(global_df.head())

    print()
    print("Dataset columns:")
    print(list(global_df.columns))

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()