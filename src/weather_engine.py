import os
import joblib
import requests
import pandas as pd
from datetime import datetime

# ============================================================
# WEATHER AI - GLOBAL CENTRAL PREDICTION ENGINE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RAIN_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "global_rain_model.pkl"
)

TEMPERATURE_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "global_temperature_model.pkl"
)

LATITUDE = -6.8235
LONGITUDE = 39.2695

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# ------------------------------------------------------------
# Features used by global models
# ------------------------------------------------------------

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "latitude",
    "longitude",
    "hour",
    "day",
    "month",
    "day_of_week",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]


# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------

print()
print("=" * 60)
print("Loading WeatherAI global models...")
print("=" * 60)

if not os.path.exists(RAIN_MODEL_FILE):
    raise FileNotFoundError(
        f"Global rain model not found:\n{RAIN_MODEL_FILE}"
    )

if not os.path.exists(TEMPERATURE_MODEL_FILE):
    raise FileNotFoundError(
        f"Global temperature model not found:\n"
        f"{TEMPERATURE_MODEL_FILE}"
    )

rain_model = joblib.load(
    RAIN_MODEL_FILE
)

temperature_model = joblib.load(
    TEMPERATURE_MODEL_FILE
)

print("Global rain model loaded.")
print("Global temperature model loaded.")


# ------------------------------------------------------------
# Get current weather
# ------------------------------------------------------------

def get_current_weather():

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": (
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

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data["current"]


# ------------------------------------------------------------
# Prepare model input
# ------------------------------------------------------------

def prepare_features(weather):

    now = pd.Timestamp.now()

    hour = now.hour
    day = now.day
    month = now.month
    day_of_week = now.dayofweek
    day_of_year = now.dayofyear

    features = {
        "temperature_2m": weather["temperature_2m"],
        "relative_humidity_2m": weather["relative_humidity_2m"],
        "precipitation": weather["precipitation"],
        "rain": weather["rain"],
        "pressure_msl": weather["pressure_msl"],
        "cloud_cover": weather["cloud_cover"],
        "wind_speed_10m": weather["wind_speed_10m"],
        "wind_direction_10m": weather["wind_direction_10m"],

        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,

        "hour_sin": __import__("math").sin(
            2 * __import__("math").pi * hour / 24
        ),

        "hour_cos": __import__("math").cos(
            2 * __import__("math").pi * hour / 24
        ),

        "day_sin": __import__("math").sin(
            2 * __import__("math").pi * day_of_year / 365
        ),

        "day_cos": __import__("math").cos(
            2 * __import__("math").pi * day_of_year / 365
        )
    }

    return pd.DataFrame(
        [features],
        columns=FEATURES
    )


# ------------------------------------------------------------
# Make prediction
# ------------------------------------------------------------

def predict():

    weather = get_current_weather()

    X = prepare_features(
        weather
    )

    # Temperature
    predicted_temperature = (
        temperature_model.predict(X)[0]
    )

    # Rain
    rain_prediction = (
        rain_model.predict(X)[0]
    )

    rain_probabilities = (
        rain_model.predict_proba(X)[0]
    )

    no_rain_probability = (
        rain_probabilities[0] * 100
    )

    rain_probability = (
        rain_probabilities[1] * 100
    )

    return {
        "current_weather": weather,

        "predicted_temperature": round(
            float(predicted_temperature),
            2
        ),

        "rain_prediction": int(
            rain_prediction
        ),

        "rain_probability": round(
            float(rain_probability),
            2
        ),

        "no_rain_probability": round(
            float(no_rain_probability),
            2
        )
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("              WEATHER AI")
    print("       GLOBAL PREDICTION ENGINE")
    print("=" * 60)

    result = predict()

    weather = result["current_weather"]

    print()
    print("CURRENT WEATHER")
    print("-" * 60)

    print(
        f"Temperature : "
        f"{weather['temperature_2m']} °C"
    )

    print(
        f"Humidity    : "
        f"{weather['relative_humidity_2m']} %"
    )

    print(
        f"Pressure    : "
        f"{weather['pressure_msl']} hPa"
    )

    print(
        f"Cloud Cover : "
        f"{weather['cloud_cover']} %"
    )

    print(
        f"Wind Speed  : "
        f"{weather['wind_speed_10m']} km/h"
    )

    print(
        f"Wind Dir.   : "
        f"{weather['wind_direction_10m']}°"
    )

    print()
    print("NEXT-HOUR AI PREDICTION")
    print("-" * 60)

    print(
        f"Temperature : "
        f"{result['predicted_temperature']:.2f} °C"
    )

    if result["rain_prediction"] == 1:
        rain_status = "RAIN LIKELY"
    else:
        rain_status = "NO RAIN LIKELY"

    print(
        f"Rain        : {rain_status}"
    )

    print(
        f"Rain Probability    : "
        f"{result['rain_probability']:.2f}%"
    )

    print(
        f"No-Rain Probability : "
        f"{result['no_rain_probability']:.2f}%"
    )

    print()
    print("Prediction Horizon: Next Hour")

    print("=" * 60)