import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# WEATHER AI - GLOBAL RAIN MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "global_weather_clean.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "global_rain_model.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print()
print("=" * 70)
print("              WEATHER AI")
print("          GLOBAL RAIN MODEL")
print("=" * 70)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print()
print("Loading global dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Total records: {len(df):,}")

# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Create binary rain target
# ------------------------------------------------------------

df["rain_target"] = (
    df["rain_next_hour"] > 0
).astype(int)

X = df[features]
y = df["rain_target"]

# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

print()
print("Rain class distribution:")

print(
    y.value_counts()
    .rename({
        0: "No Rain",
        1: "Rain"
    })
)

print()
print("Rain percentages:")

print(
    (y.value_counts(normalize=True) * 100)
    .rename({
        0: "No Rain",
        1: "Rain"
    })
    .round(2)
)

# ------------------------------------------------------------
# Time-based split
# ------------------------------------------------------------

print()
print("Creating time-based training/testing split...")

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Training records: {len(X_train):,}")
print(f"Testing records : {len(X_test):,}")

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

print()
print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

print()
print("Evaluating model...")

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

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 70)
print("GLOBAL RAIN MODEL RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "No Rain",
            "Rain"
        ],
        zero_division=0
    )
)

print()
print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("Top feature importance:")

print(
    importance.head(10).to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)

print()
print("=" * 70)