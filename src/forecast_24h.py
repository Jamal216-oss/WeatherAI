import os
import joblib
import requests
import pandas as pd
from datetime import datetime


# ============================================================
# WEATHERAI - 24 HOUR FORECAST ENGINE
# ============================================================

LATITUDE = -4.577
LONGITUDE = 34.948

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAIN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rain_model.pkl"
)

TEMPERATURE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "temperature_model.pkl"
)

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading WeatherAI models...")

rain_model = joblib.load(RAIN_MODEL_PATH)
temperature_model = joblib.load(TEMPERATURE_MODEL_PATH)

print("Rain model loaded.")
print("Temperature model loaded.")


# ============================================================
# GET HOURLY FORECAST DATA
# ============================================================

def get_forecast_data():

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "pressure_msl",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m"
        ]),

        "forecast_days": 2,
        "timezone": "auto"
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# PREPARE HOURLY DATA
# ============================================================

def prepare_data(data):

    hourly = data["hourly"]

    df = pd.DataFrame(hourly)

    df["time"] = pd.to_datetime(df["time"])

    return df


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(df):

    # We want the next 24 hours
    now = pd.Timestamp.now(tz=df["time"].dt.tz)

    future_df = df[df["time"] >= now].head(24).copy()

    if future_df.empty:
        raise ValueError("No future forecast data available.")

    future_df["hour"] = future_df["time"].dt.hour
    future_df["day"] = future_df["time"].dt.day
    future_df["month"] = future_df["time"].dt.month
    future_df["day_of_week"] = future_df["time"].dt.dayofweek

    features = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "pressure_msl",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "hour",
        "day",
        "month",
        "day_of_week"
    ]

    X = future_df[features]

    # Temperature prediction
    temperature_predictions = temperature_model.predict(X)

    # Rain prediction
    rain_predictions = rain_model.predict(X)

    rain_probabilities = rain_model.predict_proba(X)

    future_df["predicted_temperature"] = (
        temperature_predictions
    )

    future_df["rain_prediction"] = (
        rain_predictions
    )

    future_df["rain_probability"] = (
        rain_probabilities[:, 1]
    )

    return future_df


# ============================================================
# DISPLAY FORECAST
# ============================================================

def display_forecast(df):

    print("\n")
    print("=" * 75)
    print("                 WEATHERAI")
    print("             24-HOUR FORECAST")
    print("=" * 75)

    print(
        f"{'TIME':<20}"
        f"{'TEMP':<12}"
        f"{'RAIN %':<12}"
        f"{'CONDITION'}"
    )

    print("-" * 75)

    for _, row in df.iterrows():

        time = row["time"].strftime("%Y-%m-%d %H:%M")

        temperature = row["predicted_temperature"]

        rain_probability = (
            row["rain_probability"] * 100
        )

        if row["rain_prediction"] == 1:
            condition = "RAIN LIKELY"
        else:
            condition = "NO RAIN"

        print(
            f"{time:<20}"
            f"{temperature:>6.1f} °C    "
            f"{rain_probability:>6.1f}%     "
            f"{condition}"
        )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        print("\nGetting weather forecast data...")

        data = get_forecast_data()

        print("Forecast data received.")

        df = prepare_data(data)

        forecast = generate_predictions(df)

        display_forecast(forecast)

    except Exception as error:

        print("\nERROR:")
        print(error)