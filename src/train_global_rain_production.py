import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "global_weather_clean.csv"
)

OLD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "global_rain_model.pkl"
)

NEW_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "global_rain_model_production.pkl"
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
    "day_cos"
]

TARGET = "rain_next_hour"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("WEATHERAI PRODUCTION RAIN MODEL")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Records: {len(df):,}")


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=required_columns
).copy()


# ============================================================
# PREPARE TARGET
# ============================================================

X = df[FEATURES]

y = (
    df[TARGET] > 0
).astype(int)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining records: {len(X_train):,}")
print(f"Testing records : {len(X_test):,}")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining production model...")

print("Trees      : 60")
print("Max depth  : 12")
print("Min split  : 5")
print("Class      : balanced")
print("Compression: 3")


model = RandomForestClassifier(
    n_estimators=60,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating model...")

predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving production model...")

joblib.dump(
    model,
    NEW_MODEL_PATH,
    compress=3
)


# ============================================================
# FILE SIZE
# ============================================================

new_size_mb = (
    os.path.getsize(NEW_MODEL_PATH)
    / (1024 * 1024)
)


old_size_mb = 0

if os.path.exists(OLD_MODEL_PATH):
    old_size_mb = (
        os.path.getsize(OLD_MODEL_PATH)
        / (1024 * 1024)
    )


if old_size_mb > 0:
    reduction = (
        1 - (new_size_mb / old_size_mb)
    ) * 100
else:
    reduction = 0


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("PRODUCTION RAIN MODEL RESULTS")
print("=" * 70)

print(f"\nAccuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")

print("\nModel:")
print(NEW_MODEL_PATH)

print(f"\nNew model size: {new_size_mb:.2f} MB")

if old_size_mb > 0:
    print(f"Old model size: {old_size_mb:.2f} MB")
    print(f"Size reduction: {reduction:.2f}%")

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)