import os
import joblib
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_FILE = "models/rain_model.pkl"


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("==========================================")
print("       WEATHER PREDICTION ENGINE")
print("==========================================")
print()

print("Loading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print()


# ==========================================
# CURRENT WEATHER INPUT
# ==========================================

print("Enter current weather conditions.")
print()

temperature = float(
    input("Temperature (°C): ")
)

humidity = float(
    input("Relative humidity (%): ")
)

precipitation = float(
    input("Precipitation (mm): ")
)

rain = float(
    input("Current rain (mm): ")
)

pressure = float(
    input("Pressure (hPa): ")
)

cloud_cover = float(
    input("Cloud cover (%): ")
)

wind_speed = float(
    input("Wind speed (km/h): ")
)

wind_direction = float(
    input("Wind direction (degrees): ")
)

hour = int(
    input("Hour (0-23): ")
)

day = int(
    input("Day (1-31): ")
)

month = int(
    input("Month (1-12): ")
)

day_of_week = int(
    input("Day of week (0=Monday, 6=Sunday): ")
)


# ==========================================
# CREATE INPUT DATA
# ==========================================

input_data = pd.DataFrame(
    [[
        temperature,
        humidity,
        precipitation,
        rain,
        pressure,
        cloud_cover,
        wind_speed,
        wind_direction,
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
# DISPLAY RESULT
# ==========================================

print()
print("==========================================")
print("          WEATHER PREDICTION")
print("==========================================")
print()

if prediction == 1:

    print("Prediction: RAIN LIKELY")

else:

    print("Prediction: NO RAIN LIKELY")


print()

print(
    f"Rain probability: {rain_probability * 100:.2f}%"
)

print(
    f"No-rain probability: {no_rain_probability * 100:.2f}%"
)

print()

print("Prediction is for the next hour.")
print()

print("==========================================")