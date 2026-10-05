from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
import joblib
import os
import math


# ============================================================
# WEATHERAI API
# ============================================================

app = FastAPI(
    title="WeatherAI API",
    description="AI-powered global weather prediction API",
    version="1.1"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# MODEL PATHS
# ============================================================

RAIN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "global_rain_model.pkl"
)

TEMPERATURE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "global_temperature_model.pkl"
)

PRECIPITATION_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "global_precipitation_model.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    rain_model = joblib.load(
        RAIN_MODEL_PATH
    )

    temperature_model = joblib.load(
        TEMPERATURE_MODEL_PATH
    )

    precipitation_model = joblib.load(
        PRECIPITATION_MODEL_PATH
    )

    print(
        "All WeatherAI models loaded successfully."
    )

except Exception as e:

    print(
        "ERROR loading models:",
        e
    )

    rain_model = None
    temperature_model = None
    precipitation_model = None


# ============================================================
# REQUEST MODEL
# ============================================================

class LocationRequest(BaseModel):

    latitude: float
    longitude: float


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name": "WeatherAI",

        "version": "1.1",

        "status": "online",

        "models": {

            "temperature":
                temperature_model is not None,

            "rain_probability":
                rain_model is not None,

            "precipitation_amount":
                precipitation_model is not None
        }
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "temperature_model":
            temperature_model is not None,

        "rain_model":
            rain_model is not None,

        "precipitation_model":
            precipitation_model is not None
    }


# ============================================================
# VALIDATE LOCATION
# ============================================================

def validate_location(
    latitude,
    longitude
):

    if latitude < -90 or latitude > 90:

        raise HTTPException(
            status_code=400,
            detail="Latitude must be between -90 and 90."
        )

    if longitude < -180 or longitude > 180:

        raise HTTPException(
            status_code=400,
            detail="Longitude must be between -180 and 180."
        )


# ============================================================
# GET CURRENT WEATHER
# ============================================================

def get_current_weather(
    latitude,
    longitude
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

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
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(
    weather,
    latitude,
    longitude,
    time_value=None
):

    current = weather.get(
        "current",
        {}
    )

    if time_value is None:

        time_value = current.get(
            "time"
        )

    timestamp = pd.to_datetime(
        time_value
    )

    hour = timestamp.hour

    day = timestamp.day

    month = timestamp.month

    day_of_week = timestamp.dayofweek

    day_of_year = timestamp.dayofyear

    hour_sin = math.sin(
        2 * math.pi * hour / 24
    )

    hour_cos = math.cos(
        2 * math.pi * hour / 24
    )

    day_sin = math.sin(
        2 * math.pi *
        day_of_year / 365.25
    )

    day_cos = math.cos(
        2 * math.pi *
        day_of_year / 365.25
    )

    features = {

        "temperature_2m":
            current.get(
                "temperature_2m",
                0
            ),

        "relative_humidity_2m":
            current.get(
                "relative_humidity_2m",
                0
            ),

        "precipitation":
            current.get(
                "precipitation",
                0
            ),

        "rain":
            current.get(
                "rain",
                0
            ),

        "pressure_msl":
            current.get(
                "pressure_msl",
                0
            ),

        "cloud_cover":
            current.get(
                "cloud_cover",
                0
            ),

        "wind_speed_10m":
            current.get(
                "wind_speed_10m",
                0
            ),

        "wind_direction_10m":
            current.get(
                "wind_direction_10m",
                0
            ),

        "latitude":
            latitude,

        "longitude":
            longitude,

        "hour":
            hour,

        "day":
            day,

        "month":
            month,

        "day_of_week":
            day_of_week,

        "day_of_year":
            day_of_year,

        "hour_sin":
            hour_sin,

        "hour_cos":
            hour_cos,

        "day_sin":
            day_sin,

        "day_cos":
            day_cos
    }

    return pd.DataFrame(
        [features]
    )


# ============================================================
# RAIN INTENSITY
# ============================================================

def get_rain_intensity(
    amount
):

    if amount < 0.1:

        return "NO RAIN"

    elif amount < 2.5:

        return "LIGHT"

    elif amount < 7.6:

        return "MODERATE"

    elif amount < 50:

        return "HEAVY"

    else:

        return "VERY HEAVY"


# ============================================================
# RAIN PROBABILITY LEVEL
# ============================================================

def get_rain_probability_label(
    probability
):

    if probability < 20:

        return "VERY LOW"

    elif probability < 40:

        return "LOW"

    elif probability < 60:

        return "MODERATE"

    elif probability < 80:

        return "HIGH"

    else:

        return "VERY HIGH"


# ============================================================
# CONFIDENCE
# ============================================================

def get_prediction_confidence(
    probability
):

    distance = abs(
        probability - 50
    )

    confidence = distance * 2

    if confidence >= 80:

        level = "VERY HIGH"

    elif confidence >= 60:

        level = "HIGH"

    elif confidence >= 40:

        level = "MODERATE"

    elif confidence >= 20:

        level = "LOW"

    else:

        level = "VERY LOW"

    return (
        round(confidence, 2),
        level
    )


# ============================================================
# WEATHER CONDITION
# ============================================================

def get_weather_condition(
    temperature,
    rain_probability,
    precipitation,
    cloud_cover
):

    if (
        rain_probability >= 60
        and precipitation >= 0.1
    ):

        if precipitation >= 7.6:

            return "HEAVY RAIN"

        elif precipitation >= 2.5:

            return "MODERATE RAIN"

        else:

            return "LIGHT RAIN"

    if cloud_cover >= 85:

        return "OVERCAST"

    elif cloud_cover >= 60:

        return "CLOUDY"

    elif cloud_cover >= 30:

        return "PARTLY CLOUDY"

    else:

        return "CLEAR"


# ============================================================
# WEATHER ICON
# ============================================================

def get_weather_icon(
    condition
):

    icons = {

        "CLEAR":
            "☀️",

        "PARTLY CLOUDY":
            "⛅",

        "CLOUDY":
            "☁️",

        "OVERCAST":
            "☁️",

        "LIGHT RAIN":
            "🌦️",

        "MODERATE RAIN":
            "🌧️",

        "HEAVY RAIN":
            "⛈️"
    }

    return icons.get(
        condition,
        "🌤️"
    )


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: LocationRequest
):

    latitude = request.latitude

    longitude = request.longitude

    validate_location(
        latitude,
        longitude
    )

    if (
        rain_model is None
        or temperature_model is None
        or precipitation_model is None
    ):

        raise HTTPException(
            status_code=500,
            detail="One or more AI models could not be loaded."
        )

    try:

        weather = get_current_weather(
            latitude,
            longitude
        )

        features = create_features(
            weather,
            latitude,
            longitude
        )

        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        predicted_temperature = (
            temperature_model.predict(
                features
            )[0]
        )

        # ----------------------------------------------------
        # RAIN PROBABILITY
        # ----------------------------------------------------

        rain_probability = (
            rain_model.predict_proba(
                features
            )[0][1]
        )

        rain_probability = float(
            rain_probability * 100
        )

        # ----------------------------------------------------
        # PRECIPITATION
        # ----------------------------------------------------

        predicted_precipitation = (
            precipitation_model.predict(
                features
            )[0]
        )

        predicted_precipitation = max(
            0,
            float(
                predicted_precipitation
            )
        )

        current = weather[
            "current"
        ]

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        rain_status = (
            "RAIN LIKELY"
            if rain_probability >= 50
            else "NO RAIN LIKELY"
        )

        intensity = (
            get_rain_intensity(
                predicted_precipitation
            )
        )

        probability_level = (
            get_rain_probability_label(
                rain_probability
            )
        )

        confidence, confidence_level = (
            get_prediction_confidence(
                rain_probability
            )
        )

        condition = (
            get_weather_condition(
                current.get(
                    "temperature_2m",
                    0
                ),

                rain_probability,

                predicted_precipitation,

                current.get(
                    "cloud_cover",
                    0
                )
            )
        )

        weather_icon = (
            get_weather_icon(
                condition
            )
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "location": {

                "latitude":
                    latitude,

                "longitude":
                    longitude
            },

            "generated_at":
                current.get(
                    "time"
                ),

            "current_weather": {

                "temperature":
                    current.get(
                        "temperature_2m"
                    ),

                "humidity":
                    current.get(
                        "relative_humidity_2m"
                    ),

                "precipitation":
                    current.get(
                        "precipitation"
                    ),

                "rain":
                    current.get(
                        "rain"
                    ),

                "pressure":
                    current.get(
                        "pressure_msl"
                    ),

                "cloud_cover":
                    current.get(
                        "cloud_cover"
                    ),

                "wind_speed":
                    current.get(
                        "wind_speed_10m"
                    ),

                "wind_direction":
                    current.get(
                        "wind_direction_10m"
                    )
            },

            "ai_prediction": {

                "next_hour_temperature":
                    round(
                        float(
                            predicted_temperature
                        ),
                        2
                    ),

                "rain_probability":
                    round(
                        rain_probability,
                        2
                    ),

                "rain_probability_level":
                    probability_level,

                "rain_status":
                    rain_status,

                "predicted_precipitation_mm":
                    round(
                        predicted_precipitation,
                        2
                    ),

                "rain_intensity":
                    intensity,

                "confidence":
                    confidence,

                "confidence_level":
                    confidence_level,

                "weather_condition":
                    condition,

                "weather_icon":
                    weather_icon
            }
        }

    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"Weather service error: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# ============================================================
# 24-HOUR FORECAST
# ============================================================

@app.post("/forecast")
def forecast(
    request: LocationRequest
):

    latitude = request.latitude

    longitude = request.longitude

    validate_location(
        latitude,
        longitude
    )

    if (
        rain_model is None
        or temperature_model is None
        or precipitation_model is None
    ):

        raise HTTPException(
            status_code=500,
            detail="One or more AI models could not be loaded."
        )

    try:

        url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        params = {

            "latitude":
                latitude,

            "longitude":
                longitude,

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

            "timezone":
                "auto",

            "forecast_days":
                2
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        current_time = pd.to_datetime(
            data["current"]["time"]
        )

        hourly = data["hourly"]

        forecast_df = pd.DataFrame(
            hourly
        )

        forecast_df["time"] = (
            pd.to_datetime(
                forecast_df["time"]
            )
        )

        forecast_df = forecast_df[
            forecast_df["time"]
            > current_time
        ]

        forecast_df = forecast_df.head(
            24
        )

        results = []

        # ----------------------------------------------------
        # LOOP THROUGH FORECAST HOURS
        # ----------------------------------------------------

        for _, row in forecast_df.iterrows():

            timestamp = row["time"]

            hour = timestamp.hour

            day = timestamp.day

            month = timestamp.month

            day_of_week = (
                timestamp.dayofweek
            )

            day_of_year = (
                timestamp.dayofyear
            )

            hour_sin = math.sin(
                2 * math.pi * hour / 24
            )

            hour_cos = math.cos(
                2 * math.pi * hour / 24
            )

            day_sin = math.sin(
                2 * math.pi *
                day_of_year / 365.25
            )

            day_cos = math.cos(
                2 * math.pi *
                day_of_year / 365.25
            )

            features = pd.DataFrame([{

                "temperature_2m":
                    row["temperature_2m"],

                "relative_humidity_2m":
                    row[
                        "relative_humidity_2m"
                    ],

                "precipitation":
                    row["precipitation"],

                "rain":
                    row["rain"],

                "pressure_msl":
                    row["pressure_msl"],

                "cloud_cover":
                    row["cloud_cover"],

                "wind_speed_10m":
                    row["wind_speed_10m"],

                "wind_direction_10m":
                    row[
                        "wind_direction_10m"
                    ],

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "hour":
                    hour,

                "day":
                    day,

                "month":
                    month,

                "day_of_week":
                    day_of_week,

                "day_of_year":
                    day_of_year,

                "hour_sin":
                    hour_sin,

                "hour_cos":
                    hour_cos,

                "day_sin":
                    day_sin,

                "day_cos":
                    day_cos
            }])

            # ------------------------------------------------
            # TEMPERATURE
            # ------------------------------------------------

            ai_temperature = (
                temperature_model.predict(
                    features
                )[0]
            )

            # ------------------------------------------------
            # RAIN PROBABILITY
            # ------------------------------------------------

            rain_probability = (
                rain_model.predict_proba(
                    features
                )[0][1]
            )

            rain_probability = float(
                rain_probability * 100
            )

            # ------------------------------------------------
            # PRECIPITATION
            # ------------------------------------------------

            precipitation_amount = (
                precipitation_model.predict(
                    features
                )[0]
            )

            precipitation_amount = max(
                0,
                float(
                    precipitation_amount
                )
            )

            # ------------------------------------------------
            # LABELS
            # ------------------------------------------------

            rain_status = (
                "RAIN LIKELY"
                if rain_probability >= 50
                else "NO RAIN LIKELY"
            )

            intensity = (
                get_rain_intensity(
                    precipitation_amount
                )
            )

            probability_level = (
                get_rain_probability_label(
                    rain_probability
                )
            )

            confidence, confidence_level = (
                get_prediction_confidence(
                    rain_probability
                )
            )

            condition = (
                get_weather_condition(
                    row["temperature_2m"],
                    rain_probability,
                    precipitation_amount,
                    row["cloud_cover"]
                )
            )

            weather_icon = (
                get_weather_icon(
                    condition
                )
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            results.append({

                "time":
                    timestamp.strftime(
                        "%Y-%m-%d %H:%M"
                    ),

                "temperature":
                    round(
                        float(
                            row[
                                "temperature_2m"
                            ]
                        ),
                        1
                    ),

                "ai_temperature":
                    round(
                        float(
                            ai_temperature
                        ),
                        1
                    ),

                "rain":
                    rain_status,

                "rain_probability":
                    round(
                        rain_probability,
                        2
                    ),

                "rain_probability_level":
                    probability_level,

                "precipitation_mm":
                    round(
                        precipitation_amount,
                        2
                    ),

                "rain_intensity":
                    intensity,

                "confidence":
                    confidence,

                "confidence_level":
                    confidence_level,

                "weather_condition":
                    condition,

                "weather_icon":
                    weather_icon,

                "humidity":
                    round(
                        float(
                            row[
                                "relative_humidity_2m"
                            ]
                        ),
                        1
                    ),

                "cloud_cover":
                    round(
                        float(
                            row[
                                "cloud_cover"
                            ]
                        ),
                        1
                    ),

                "wind_speed":
                    round(
                        float(
                            row[
                                "wind_speed_10m"
                            ]
                        ),
                        1
                    )
            })

        return {

            "location": {

                "latitude":
                    latitude,

                "longitude":
                    longitude
            },

            "forecast_horizon":
                "Next 24 Hours",

            "generated_at":
                current_time.strftime(
                    "%Y-%m-%d %H:%M"
                ),

            "forecast":
                results
        }

    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"Weather service error: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Forecast error: {str(e)}"
        )