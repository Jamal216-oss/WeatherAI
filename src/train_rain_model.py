import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "data/singida_weather_clean.csv"
MODEL_FILE = "models/rain_model.pkl"


# ==========================================
# LOAD DATA
# ==========================================

print("==========================================")
print("       RAIN PREDICTION MODEL")
print("==========================================")
print()

print("Loading cleaned dataset...")

data = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Records:", len(data))
print()


# ==========================================
# CREATE BINARY RAIN TARGET
# ==========================================

# 0 = No rain
# 1 = Rain

data["rain_target"] = (
    data["rain_next_hour"] > 0
).astype(int)


# ==========================================
# FEATURES
# ==========================================

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


X = data[features]

y = data["rain_target"]


print("Number of features:", len(features))
print("Features:")
for feature in features:
    print("-", feature)

print()


# ==========================================
# TIME-BASED TRAIN/TEST SPLIT
# ==========================================

print("Splitting dataset...")

# IMPORTANT:
# Weather data is time-dependent.
# We train on older data and test on newer data.

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print()


# ==========================================
# TRAIN MODEL
# ==========================================

print("Training Random Forest model...")
print("Please wait...")
print()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


print("Model training completed.")
print()


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("Testing model...")

predictions = model.predict(X_test)


# ==========================================
# EVALUATION
# ==========================================

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


print()
print("==========================================")
print("          MODEL PERFORMANCE")
print("==========================================")
print()

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print()


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("Confusion Matrix:")
print()

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print()


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

import joblib

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


print()
print("==========================================")
print("       MODEL SAVED SUCCESSFULLY")
print("==========================================")
print()

print("Model saved at:")
print(
    os.path.abspath(MODEL_FILE)
)

print()
print("Rain prediction model is ready.")