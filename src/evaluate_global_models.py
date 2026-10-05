import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "global_weather_clean.csv"
)

TEMP_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "global_temperature_model.pkl"
)

RAIN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "global_rain_model.pkl"
)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Records:", len(df))
print("Locations:", df["location"].nunique())


# =========================================================
# FEATURES
# =========================================================

features = [

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


# =========================================================
# TARGETS
# =========================================================

X = df[features]

y_temperature = df[
    "temperature_next_hour"
]

y_rain = (
    df["rain_next_hour"] > 0
).astype(int)


# =========================================================
# TIME-BASED TEST SET
# =========================================================

print("\nPreparing test data...")

df["time"] = pd.to_datetime(
    df["time"]
)

# Last 20% chronologically
cutoff = int(
    len(df) * 0.8
)

test_df = df.iloc[cutoff:].copy()

X_test = test_df[features]

y_temp_test = test_df[
    "temperature_next_hour"
]

y_rain_test = (
    test_df["rain_next_hour"] > 0
).astype(int)


# =========================================================
# LOAD MODELS
# =========================================================

print("\nLoading models...")

temperature_model = joblib.load(
    TEMP_MODEL_PATH
)

rain_model = joblib.load(
    RAIN_MODEL_PATH
)


# =========================================================
# TEMPERATURE PREDICTION
# =========================================================

print("\nEvaluating temperature model...")

temp_predictions = (
    temperature_model.predict(
        X_test
    )
)


mae = mean_absolute_error(
    y_temp_test,
    temp_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_temp_test,
        temp_predictions
    )
)

r2 = r2_score(
    y_temp_test,
    temp_predictions
)


# =========================================================
# TEMPERATURE BASELINE
# =========================================================

# Simple baseline:
# next-hour temperature = current temperature

baseline_predictions = test_df[
    "temperature_2m"
].values


baseline_mae = mean_absolute_error(
    y_temp_test,
    baseline_predictions
)


# =========================================================
# RAIN PREDICTION
# =========================================================

print("\nEvaluating rain model...")

rain_predictions = (
    rain_model.predict(
        X_test
    )
)


rain_accuracy = accuracy_score(
    y_rain_test,
    rain_predictions
)

rain_precision = precision_score(
    y_rain_test,
    rain_predictions,
    zero_division=0
)

rain_recall = recall_score(
    y_rain_test,
    rain_predictions,
    zero_division=0
)

rain_f1 = f1_score(
    y_rain_test,
    rain_predictions,
    zero_division=0
)


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("WEATHERAI MODEL EVALUATION")
print("=" * 60)

print("\nTEMPERATURE MODEL")
print("-" * 60)

print(
    f"AI MAE       : {mae:.4f} °C"
)

print(
    f"AI RMSE      : {rmse:.4f} °C"
)

print(
    f"AI R²        : {r2:.4f}"
)

print(
    f"Baseline MAE : {baseline_mae:.4f} °C"
)


print("\nRAIN MODEL")
print("-" * 60)

print(
    f"Accuracy     : {rain_accuracy:.4f}"
)

print(
    f"Precision    : {rain_precision:.4f}"
)

print(
    f"Recall       : {rain_recall:.4f}"
)

print(
    f"F1 Score     : {rain_f1:.4f}"
)


print("\n")
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)