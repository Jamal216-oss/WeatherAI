import requests
import joblib
import pandas as pd
from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_FILE = "models/rain_model.pkl"

LATITUDE = -4.577
LONGITUDE = 34.948

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


# ==========================================
# LOAD MODEL
# ==========================================

print("==========================================")
print("       LIVE WEATHER PREDICTION")
print("==========================================")
print()

print("Loading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print()


# ==========================================
# GET CURRENT WEATHER
# ==========================================

print("Getting current weather data...")
print("Please wait...")

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
    timeout=60
)

response.raise_for_status()

weather = response.json()

current = weather["current"]


print("Current weather received.")
print()


# ==========================================
# GET TIME INFORMATION
# ==========================================

current_time = datetime.fromisoformat(
    current["time"]
)

hour = current_time.hour
day = current_time.day
month = current_time.month
day_of_week = current_time.weekday()


# ==========================================
# DISPLAY CURRENT WEATHER
# ==========================================

print("==========================================")
print("          CURRENT WEATHER")
print("==========================================")
print()

print(
    f"Time: {current_time}"
)

print(
    f"Temperature: {current['temperature_2m']} °C"
)

print(
    f"Humidity: {current['relative_humidity_2m']} %"
)

print(
    f"Precipitation: {current['precipitation']} mm"
)

print(
    f"Rain: {current['rain']} mm"
)

print(
    f"Pressure: {current['pressure_msl']} hPa"
)

print(
    f"Cloud cover: {current['cloud_cover']} %"
)

print(
    f"Wind speed: {current['wind_speed_10m']} km/h"
)

print(
    f"Wind direction: {current['wind_direction_10m']}°"
)

print()


# ==========================================
# PREPARE MODEL INPUT
# ==========================================

input_data = pd.DataFrame(
    [[
        current["temperature_2m"],
        current["relative_humidity_2m"],
        current["precipitation"],
        current["rain"],
        current["pressure_msl"],
        current["cloud_cover"],
        current["wind_speed_10m"],
        current["wind_direction_10m"],
        hour,
        day,
        month,
        day_of_week
    ]],
    columns=[
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
)


# ==========================================
# MAKE PREDICTION
# ==========================================

prediction = model.predict(
    input_data
)[0]

probabilities = model.predict_proba(
    input_data
)[0]

no_rain_probability = probabilities[0]
rain_probability = probabilities[1]


# ==========================================
# DISPLAY PREDICTION
# ==========================================

print("==========================================")
print("        NEXT-HOUR RAIN PREDICTION")
print("==========================================")
print()

if prediction == 1:
    print("Prediction: RAIN LIKELY")
else:
    print("Prediction: NO RAIN LIKELY")

print()

print(
    f"Rain probability: "
    f"{rain_probability * 100:.2f}%"
)

print(
    f"No-rain probability: "
    f"{no_rain_probability * 100:.2f}%"
)

print()

print("Prediction horizon: Next hour")

print()
print("==========================================")
print("             PREDICTION DONE")
print("==========================================")