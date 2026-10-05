import os
import requests
import pandas as pd


def get_weather_data(latitude, longitude, start_date, end_date):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "pressure_msl,"
            "cloud_cover,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),
        "timezone": "auto"
    }

    print("Requesting weather data...")

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    weather_data = pd.DataFrame(data["hourly"])

    return weather_data


if __name__ == "__main__":

    # Singida coordinates
    latitude = -4.577
    longitude = 34.948

    # Long historical period
    start_date = "2020-01-01"
    end_date = "2025-12-31"

    print("======================================")
    print("      WEATHER DATA COLLECTION")
    print("======================================")
    print()

    print("Location: Singida, Tanzania")
    print("Start date:", start_date)
    print("End date:", end_date)
    print()
    print("Collecting historical weather data...")
    print("This may take some time. Please wait...")
    print()

    weather_data = get_weather_data(
        latitude,
        longitude,
        start_date,
        end_date
    )

    # Make sure data folder exists
    os.makedirs("data", exist_ok=True)

    # Save historical dataset
    output_file = os.path.join(
        "data",
        "singida_weather_historical.csv"
    )

    weather_data.to_csv(
        output_file,
        index=False
    )

    print()
    print("======================================")
    print("      COLLECTION COMPLETED")
    print("======================================")
    print()

    print("Number of records:", len(weather_data))
    print("Number of columns:", len(weather_data.columns))

    print()
    print("Columns:")
    for column in weather_data.columns:
        print("-", column)

    print()
    print("First 5 records:")
    print(weather_data.head())

    print()
    print("Last 5 records:")
    print(weather_data.tail())

    print()
    print("Data saved successfully at:")
    print(os.path.abspath(output_file))