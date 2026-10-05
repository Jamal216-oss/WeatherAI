import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "data/singida_weather_clean.csv"
MODEL_FILE = "models/temperature_model.pkl"


# ==========================================
# LOAD DATA
# ==========================================

print("==========================================")
print("     TEMPERATURE PREDICTION MODEL")
print("==========================================")
print()

print("Loading cleaned dataset...")

data = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Records:", len(data))
print()


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


# Input features
X = data[features]

# Target: temperature during next hour
y = data["temperature_next_hour"]


print("Number of features:", len(features))
print()


# ==========================================
# TIME-BASED TRAIN/TEST SPLIT
# ==========================================

print("Splitting dataset...")

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

print("Training Random Forest temperature model...")
print("Please wait...")
print()

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
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

predictions = model.predict(
    X_test
)


# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print()
print("==========================================")
print("       MODEL PERFORMANCE")
print("==========================================")
print()

print(
    f"MAE : {mae:.4f} °C"
)

print(
    f"RMSE: {rmse:.4f} °C"
)

print(
    f"R²  : {r2:.4f}"
)

print()


# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

print("Sample predictions:")
print()

results = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": predictions[:10]
})

print(results.round(2))

print()


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)


print("==========================================")
print("     MODEL SAVED SUCCESSFULLY")
print("==========================================")
print()

print("Model saved at:")

print(
    os.path.abspath(MODEL_FILE)
)

print()

print("Temperature prediction model is ready.")