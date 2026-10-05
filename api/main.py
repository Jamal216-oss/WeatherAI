from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import pandas as pd
import numpy as np
import requests

from pathlib import Path
from datetime import datetime
from calendar import month_name


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="WeatherAI API",
    description=(
        "AI-powered global weather forecasting API "
        "for temperature, rain probability and precipitation."
    ),
    version="1.0.0"
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

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


TEMPERATURE_MODEL_PATH = (
    MODELS_DIR / "global_temperature_model_production.pkl"
)

RAIN_MODEL_PATH = (
    MODELS_DIR / "global_rain_model_production.pkl"
)

# IMPORTANT:
# V2 predicts NEXT-HOUR precipitation correctly.
PRECIPITATION_MODEL_PATH = (
    MODELS_DIR / "global_precipitation_model_production_v2.pkl"
)


# ============================================================
# LOAD MODELS
# ============================================================

def load_model(path: Path, model_name: str):

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} not found:\n{path}"
        )

    print(f"Loading {model_name}...")
    model = joblib.load(path)
    print(f"{model_name} loaded successfully.")

    return model


temperature_model = load_model(
    TEMPERATURE_MODEL_PATH,
    "Temperature model"
)

rain_model = load_model(
    RAIN_MODEL_PATH,
    "Rain model"
)

precipitation_model = load_model(
    PRECIPITATION_MODEL_PATH,
    "Precipitation V2 model"
)


# ============================================================
# FEATURES
# ============================================================

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
    "day_cos",
]


# ============================================================
# REQUEST MODEL
# ============================================================

class LocationRequest(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude between -90 and 90"
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude between -180 and 180"
    )


# ============================================================
# LOCATION VALIDATION
# ============================================================

def validate_location(latitude: float, longitude: float):

    if not -90 <= latitude <= 90:
        raise HTTPException(
            status_code=400,
            detail="Latitude must be between -90 and 90."
        )

    if not -180 <= longitude <= 180:
        raise HTTPException(
            status_code=400,
            detail="Longitude must be between -180 and 180."
        )


# ============================================================
# OPEN-METEO CURRENT WEATHER
# ============================================================

def get_current_weather(latitude: float, longitude: float):

    url = "https://api.open-meteo.com/v1/forecast"

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

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        if "current" not in data:
            raise ValueError(
                "Current weather data not available."
            )

        return data["current"]

    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=f"Weather service error: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process weather data: {str(e)}"
        )


# ============================================================
# CREATE ML FEATURES
# ============================================================

def create_features(
    weather_data: dict,
    latitude: float,
    longitude: float,
    timestamp
):

    timestamp = pd.to_datetime(timestamp)

    hour = timestamp.hour
    day = timestamp.day
    month = timestamp.month
    day_of_week = timestamp.dayofweek
    day_of_year = timestamp.dayofyear

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    day_sin = np.sin(
        2 * np.pi * day_of_year / 365
    )

    day_cos = np.cos(
        2 * np.pi * day_of_year / 365
    )

    features = {

        "temperature_2m": float(
            weather_data.get(
                "temperature_2m",
                0
            )
        ),

        "relative_humidity_2m": float(
            weather_data.get(
                "relative_humidity_2m",
                0
            )
        ),

        "precipitation": float(
            weather_data.get(
                "precipitation",
                0
            )
        ),

        "rain": float(
            weather_data.get(
                "rain",
                0
            )
        ),

        "pressure_msl": float(
            weather_data.get(
                "pressure_msl",
                0
            )
        ),

        "cloud_cover": float(
            weather_data.get(
                "cloud_cover",
                0
            )
        ),

        "wind_speed_10m": float(
            weather_data.get(
                "wind_speed_10m",
                0
            )
        ),

        "wind_direction_10m": float(
            weather_data.get(
                "wind_direction_10m",
                0
            )
        ),

        "latitude": float(latitude),

        "longitude": float(longitude),

        "hour": hour,

        "day": day,

        "month": month,

        "day_of_week": day_of_week,

        "day_of_year": day_of_year,

        "hour_sin": hour_sin,

        "hour_cos": hour_cos,

        "day_sin": day_sin,

        "day_cos": day_cos,
    }

    return pd.DataFrame(
        [[features[column] for column in FEATURES]],
        columns=FEATURES
    )


# ============================================================
# RAIN INTENSITY
# ============================================================

def get_rain_intensity(precipitation: float):

    precipitation = max(
        0,
        float(precipitation)
    )

    if precipitation < 0.1:
        return "NO RAIN"

    elif precipitation < 2.5:
        return "LIGHT RAIN"

    elif precipitation < 7.6:
        return "MODERATE RAIN"

    else:
        return "HEAVY RAIN"


# ============================================================
# RAIN PROBABILITY LABEL
# ============================================================

def get_rain_probability_label(probability: float):

    if probability >= 60:
        return "RAIN LIKELY"

    return "NO RAIN LIKELY"


# ============================================================
# PREDICTION SIGNAL
# ============================================================

def get_prediction_confidence(probability: float):

    probability = float(probability)

    distance = abs(
        probability - 50
    )

    confidence = min(
        distance * 2,
        100
    )

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

    return round(confidence, 2), level


# ============================================================
# WEATHER CONDITION
# ============================================================

def get_weather_condition(
    precipitation: float,
    rain_probability: float,
    cloud_cover: float
):

    precipitation = max(
        0,
        float(precipitation)
    )

    rain_probability = float(
        rain_probability
    )

    cloud_cover = float(
        cloud_cover
    )

    if precipitation >= 7.6:
        return "HEAVY RAIN"

    if precipitation >= 2.5:
        return "MODERATE RAIN"

    if precipitation >= 0.1:
        return "LIGHT RAIN"

    if rain_probability >= 60:
        return "RAIN LIKELY"

    if cloud_cover >= 80:
        return "CLOUDY"

    if cloud_cover >= 40:
        return "PARTLY CLOUDY"

    return "CLEAR"


# ============================================================
# WEATHER ICON
# ============================================================

def get_weather_icon(condition: str):

    icons = {

        "HEAVY RAIN": "🌧️",

        "MODERATE RAIN": "🌧️",

        "LIGHT RAIN": "🌦️",

        "RAIN LIKELY": "🌧️",

        "CLOUDY": "☁️",

        "PARTLY CLOUDY": "⛅",

        "CLEAR": "☀️",

    }

    return icons.get(
        condition,
        "🌤️"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name": "WeatherAI API",

        "version": "1.0.0",

        "status": "online",

        "models": {

            "temperature": "production",

            "rain": "production",

            "precipitation": "production_v2"

        },

        "endpoints": [

            "/health",

            "/predict",

            "/forecast",

            "/annual-weather"

        ]
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "temperature_model": "loaded",

        "rain_model": "loaded",

        "precipitation_model":
            "production_v2_loaded",

        "timestamp":
            datetime.utcnow().isoformat()
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: LocationRequest):

    latitude = request.latitude
    longitude = request.longitude

    validate_location(
        latitude,
        longitude
    )

    # --------------------------------------------------------
    # Get current weather
    # --------------------------------------------------------

    current = get_current_weather(
        latitude,
        longitude
    )

    current_time = pd.to_datetime(
        current["time"]
    )

    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    features = create_features(
        current,
        latitude,
        longitude,
        current_time
    )

    # --------------------------------------------------------
    # Temperature prediction
    # --------------------------------------------------------

    predicted_temperature = (
        temperature_model.predict(features)[0]
    )

    # --------------------------------------------------------
    # Rain prediction
    # --------------------------------------------------------

    rain_probability = (
        rain_model.predict_proba(features)[0][1]
        * 100
    )

    rain_probability = float(
        np.clip(
            rain_probability,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Precipitation V2 prediction
    # --------------------------------------------------------

    predicted_precipitation = (
        precipitation_model.predict(features)[0]
    )

    predicted_precipitation = max(
        0,
        float(predicted_precipitation)
    )

    # --------------------------------------------------------
    # Derived values
    # --------------------------------------------------------

    rain_status = get_rain_probability_label(
        rain_probability
    )

    rain_intensity = get_rain_intensity(
        predicted_precipitation
    )

    confidence, confidence_level = (
        get_prediction_confidence(
            rain_probability
        )
    )

    condition = get_weather_condition(
        predicted_precipitation,
        rain_probability,
        current["cloud_cover"]
    )

    icon = get_weather_icon(
        condition
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "location": {

            "latitude": latitude,

            "longitude": longitude

        },

        "current_weather": {

            "time": current["time"],

            "temperature": round(
                float(
                    current["temperature_2m"]
                ),
                2
            ),

            "humidity": round(
                float(
                    current[
                        "relative_humidity_2m"
                    ]
                ),
                2
            ),

            "precipitation": round(
                float(
                    current["precipitation"]
                ),
                3
            ),

            "rain": round(
                float(
                    current["rain"]
                ),
                3
            ),

            "pressure": round(
                float(
                    current["pressure_msl"]
                ),
                2
            ),

            "cloud_cover": round(
                float(
                    current["cloud_cover"]
                ),
                2
            ),

            "wind_speed": round(
                float(
                    current["wind_speed_10m"]
                ),
                2
            ),

            "wind_direction": round(
                float(
                    current["wind_direction_10m"]
                ),
                2
            )
        },

        "prediction": {

            "next_hour_temperature": round(
                float(predicted_temperature),
                2
            ),

            "rain_probability": round(
                rain_probability,
                2
            ),

            "rain_status": rain_status,

            "predicted_precipitation": round(
                predicted_precipitation,
                3
            ),

            "rain_intensity": rain_intensity,

            "confidence": confidence,

            "confidence_level": confidence_level,

            "condition": condition,

            "icon": icon
        },

        "generated_at":
            datetime.utcnow().isoformat()
    }


# ============================================================
# FORECAST DATA
# ============================================================

def get_forecast_weather(
    latitude: float,
    longitude: float
):

    url = "https://api.open-meteo.com/v1/forecast"

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

        "forecast_days": 2,

        "timezone": "auto"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:

        raise HTTPException(
            status_code=502,
            detail=(
                "Weather forecast service error: "
                f"{str(e)}"
            )
        )


# ============================================================
# FORECAST
# ============================================================

@app.post("/forecast")
def forecast(request: LocationRequest):

    latitude = request.latitude
    longitude = request.longitude

    validate_location(
        latitude,
        longitude
    )

    data = get_forecast_weather(
        latitude,
        longitude
    )

    if "hourly" not in data:

        raise HTTPException(
            status_code=502,
            detail="Hourly forecast data not available."
        )

    hourly = data["hourly"]

    # --------------------------------------------------------
    # Current time from Open-Meteo
    # --------------------------------------------------------

    if (
        "current" in data
        and "time" in data["current"]
    ):

        current_time = pd.to_datetime(
            data["current"]["time"]
        )

    else:

        current_time = pd.to_datetime(
            hourly["time"][0]
        )

    # --------------------------------------------------------
    # Convert hourly times
    # --------------------------------------------------------

    hourly_times = pd.to_datetime(
        hourly["time"]
    )

    # --------------------------------------------------------
    # Find first hour after current time
    # --------------------------------------------------------

    future_indices = np.where(
        hourly_times > current_time
    )[0]

    if len(future_indices) == 0:

        raise HTTPException(
            status_code=500,
            detail="No future hourly forecast available."
        )

    start_index = int(
        future_indices[0]
    )

    end_index = min(
        start_index + 24,
        len(hourly_times)
    )

    # --------------------------------------------------------
    # Build 24-hour forecast
    # --------------------------------------------------------

    forecast_results = []

    for index in range(
        start_index,
        end_index
    ):

        timestamp = hourly_times[index]

        # ----------------------------------------------------
        # Weather data for this hour
        # ----------------------------------------------------

        weather = {

            "temperature_2m": hourly[
                "temperature_2m"
            ][index],

            "relative_humidity_2m": hourly[
                "relative_humidity_2m"
            ][index],

            "precipitation": hourly[
                "precipitation"
            ][index],

            "rain": hourly[
                "rain"
            ][index],

            "pressure_msl": hourly[
                "pressure_msl"
            ][index],

            "cloud_cover": hourly[
                "cloud_cover"
            ][index],

            "wind_speed_10m": hourly[
                "wind_speed_10m"
            ][index],

            "wind_direction_10m": hourly[
                "wind_direction_10m"
            ][index]
        }

        # ----------------------------------------------------
        # Create features
        # ----------------------------------------------------

        features = create_features(
            weather,
            latitude,
            longitude,
            timestamp
        )

        # ----------------------------------------------------
        # Temperature prediction
        # ----------------------------------------------------

        predicted_temperature = (
            temperature_model.predict(
                features
            )[0]
        )

        # ----------------------------------------------------
        # Rain prediction
        # ----------------------------------------------------

        rain_probability = (
            rain_model.predict_proba(
                features
            )[0][1]
            * 100
        )

        rain_probability = float(
            np.clip(
                rain_probability,
                0,
                100
            )
        )

        # ----------------------------------------------------
        # Precipitation V2 prediction
        # ----------------------------------------------------

        predicted_precipitation = (
            precipitation_model.predict(
                features
            )[0]
        )

        predicted_precipitation = max(
            0,
            float(predicted_precipitation)
        )

        # ----------------------------------------------------
        # Derived values
        # ----------------------------------------------------

        rain_status = (
            get_rain_probability_label(
                rain_probability
            )
        )

        rain_intensity = (
            get_rain_intensity(
                predicted_precipitation
            )
        )

        confidence, confidence_level = (
            get_prediction_confidence(
                rain_probability
            )
        )

        condition = get_weather_condition(
            predicted_precipitation,
            rain_probability,
            weather["cloud_cover"]
        )

        icon = get_weather_icon(
            condition
        )

        # ----------------------------------------------------
        # Append result
        # ----------------------------------------------------

        forecast_results.append({

            "time": timestamp.strftime(
                "%Y-%m-%dT%H:%M"
            ),

            "input_temperature": round(
                float(
                    weather["temperature_2m"]
                ),
                2
            ),

            "predicted_temperature": round(
                float(predicted_temperature),
                2
            ),

            "rain_probability": round(
                rain_probability,
                2
            ),

            "rain_status": rain_status,

            "predicted_precipitation": round(
                predicted_precipitation,
                3
            ),

            "rain_intensity": rain_intensity,

            "confidence": confidence,

            "confidence_level": confidence_level,

            "condition": condition,

            "icon": icon
        })

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "location": {

            "latitude": latitude,

            "longitude": longitude

        },

        "forecast_horizon":
            "Next 24 Hours",

        "generated_at":
            datetime.utcnow().isoformat(),

        "forecast":
            forecast_results
    }


# ============================================================
# ANNUAL WEATHER STATISTICS
# ============================================================

@app.get("/annual-weather")
def annual_weather(
    latitude: float,
    longitude: float,
    year: int
):

    """
    Generate monthly and annual weather statistics
    for a specific location and year using
    Open-Meteo historical weather data.
    """

    # --------------------------------------------------------
    # Validate location
    # --------------------------------------------------------

    validate_location(
        latitude,
        longitude
    )

    # --------------------------------------------------------
    # Validate year
    # --------------------------------------------------------

    current_year = datetime.now().year

    if year < 1940 or year > current_year:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Year must be between "
                f"1940 and {current_year}."
            )
        )

    # --------------------------------------------------------
    # Open-Meteo historical API
    # --------------------------------------------------------

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "start_date": f"{year}-01-01",

        "end_date": f"{year}-12-31",

        "daily": (
            "temperature_2m_mean,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "rain_sum,"
            "precipitation_hours,"
            "wind_speed_10m_max"
        ),

        "timezone": "auto"
    }

    # --------------------------------------------------------
    # Request historical data
    # --------------------------------------------------------

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to retrieve historical "
                f"weather data: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # Check daily data
    # --------------------------------------------------------

    daily = data.get("daily")

    if not daily:

        raise HTTPException(
            status_code=404,
            detail=(
                "No historical weather data "
                "found for this location."
            )
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame({

        "date": daily.get(
            "time",
            []
        ),

        "temperature_mean": daily.get(
            "temperature_2m_mean",
            []
        ),

        "temperature_max": daily.get(
            "temperature_2m_max",
            []
        ),

        "temperature_min": daily.get(
            "temperature_2m_min",
            []
        ),

        "precipitation": daily.get(
            "precipitation_sum",
            []
        ),

        "rain": daily.get(
            "rain_sum",
            []
        ),

        "precipitation_hours": daily.get(
            "precipitation_hours",
            []
        ),

        "wind_speed": daily.get(
            "wind_speed_10m_max",
            []
        )
    })

    if df.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                "Historical weather dataset "
                "is empty."
            )
        )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["month"] = df[
        "date"
    ].dt.month

    # Replace missing values with zero
    # where appropriate
    df["rain"] = df[
        "rain"
    ].fillna(0)

    df["precipitation"] = df[
        "precipitation"
    ].fillna(0)

    df["precipitation_hours"] = df[
        "precipitation_hours"
    ].fillna(0)

    # --------------------------------------------------------
    # Rainy / dry day classification
    # --------------------------------------------------------

    df["rainy_day"] = (
        df["rain"] > 0
    )

    df["dry_day"] = (
        df["rain"] <= 0
    )

    # --------------------------------------------------------
    # MONTHLY STATISTICS
    # --------------------------------------------------------

    monthly = []

    for month in range(1, 13):

        month_df = df[
            df["month"] == month
        ]

        if month_df.empty:
            continue

        monthly.append({

            "month":
                month_name[month],

            "month_number":
                month,

            "average_temperature":
                round(
                    month_df[
                        "temperature_mean"
                    ].mean(),
                    2
                ),

            "maximum_temperature":
                round(
                    month_df[
                        "temperature_max"
                    ].max(),
                    2
                ),

            "minimum_temperature":
                round(
                    month_df[
                        "temperature_min"
                    ].min(),
                    2
                ),

            "rainfall_mm":
                round(
                    month_df[
                        "precipitation"
                    ].sum(),
                    2
                ),

            "rain_mm":
                round(
                    month_df[
                        "rain"
                    ].sum(),
                    2
                ),

            "rainy_days":
                int(
                    month_df[
                        "rainy_day"
                    ].sum()
                ),

            "dry_days":
                int(
                    month_df[
                        "dry_day"
                    ].sum()
                ),

            "precipitation_hours":
                round(
                    month_df[
                        "precipitation_hours"
                    ].sum(),
                    2
                ),

            "average_wind_speed":
                round(
                    month_df[
                        "wind_speed"
                    ].mean(),
                    2
                ),

            "maximum_wind_speed":
                round(
                    month_df[
                        "wind_speed"
                    ].max(),
                    2
                )
        })

    # --------------------------------------------------------
    # ANNUAL SUMMARY
    # --------------------------------------------------------

    annual_summary = {

        "average_temperature":
            round(
                df[
                    "temperature_mean"
                ].mean(),
                2
            ),

        "maximum_temperature":
            round(
                df[
                    "temperature_max"
                ].max(),
                2
            ),

        "minimum_temperature":
            round(
                df[
                    "temperature_min"
                ].min(),
                2
            ),

        "total_rainfall_mm":
            round(
                df[
                    "precipitation"
                ].sum(),
                2
            ),

        "total_rain_mm":
            round(
                df[
                    "rain"
                ].sum(),
                2
            ),

        "rainy_days":
            int(
                df[
                    "rainy_day"
                ].sum()
            ),

        "dry_days":
            int(
                df[
                    "dry_day"
                ].sum()
            ),

        "precipitation_hours":
            round(
                df[
                    "precipitation_hours"
                ].sum(),
                2
            ),

        "average_wind_speed":
            round(
                df[
                    "wind_speed"
                ].mean(),
                2
            ),

        "maximum_wind_speed":
            round(
                df[
                    "wind_speed"
                ].max(),
                2
            )
    }

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "status": "success",

        "location": {

            "latitude":
                latitude,

            "longitude":
                longitude
        },

        "year":
            year,

        "annual_summary":
            annual_summary,

        "monthly":
            monthly,

        "generated_at":
            datetime.utcnow().isoformat()
    }