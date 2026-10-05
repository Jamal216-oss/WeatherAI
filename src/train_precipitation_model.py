import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "global_precipitation_model.pkl"
)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Records:", len(df))


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
# TARGET
# =========================================================

# Next-hour precipitation

df["precipitation_next_hour"] = (
    df.groupby("location")["precipitation"]
    .shift(-1)
)

# Remove rows without target

df = df.dropna(
    subset=["precipitation_next_hour"]
)


# =========================================================
# DATA
# =========================================================

X = df[features]

y = df[
    "precipitation_next_hour"
]


# =========================================================
# TIME-BASED SPLIT
# =========================================================

cutoff = int(
    len(df) * 0.8
)

X_train = X.iloc[:cutoff]
X_test = X.iloc[cutoff:]

y_train = y.iloc[:cutoff]
y_test = y.iloc[cutoff:]


print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# =========================================================
# MODEL
# =========================================================

print("\nTraining precipitation model...")

model = RandomForestRegressor(

    n_estimators=250,

    max_depth=20,

    min_samples_split=5,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTION
# =========================================================

predictions = model.predict(
    X_test
)


# Rainfall cannot be negative

predictions = np.maximum(
    predictions,
    0
)


# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("WEATHERAI PRECIPITATION MODEL")
print("=" * 60)

print(
    f"MAE  : {mae:.4f} mm"
)

print(
    f"RMSE : {rmse:.4f} mm"
)

print(
    f"R²   : {r2:.4f}"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({

    "feature": features,

    "importance":
        model.feature_importances_

})

importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\nTop Features:")

print(
    importance.head(10).to_string(
        index=False
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    MODEL_PATH
)


print("\nModel saved to:")

print(
    MODEL_PATH
)


print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)