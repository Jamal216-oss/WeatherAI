import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "global_weather_precipitation_clean.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "global_precipitation_model_production_v2.pkl"
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

TARGET = "precipitation_next_hour"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("WEATHER AI - PRODUCTION PRECIPITATION MODEL V2")
print("=" * 70)


# ============================================================
# CHECK DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Records: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )


if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=FEATURES + [TARGET]
).reset_index(drop=True)

print(
    f"Records after cleaning: {len(df):,}"
)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nDataset split:")
print(f"Training records: {len(X_train):,}")
print(f"Testing records:  {len(X_test):,}")


# ============================================================
# MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=60,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

print("Evaluating model...")

predictions = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"\nMAE  : {mae:.4f} mm")
print(f"RMSE : {rmse:.4f} mm")
print(f"R²   : {r2:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nTop Feature Importance:")

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.head(10).to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE,
    compress=3
)


# ============================================================
# FILE SIZE
# ============================================================

size_mb = MODEL_FILE.stat().st_size / (
    1024 * 1024
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\nModel saved:")
print(MODEL_FILE)

print(f"\nModel size: {size_mb:.2f} MB")

print("\nTarget:")
print("precipitation_next_hour")

print("\nThis model predicts:")
print("Current weather conditions")
print("        ↓")
print("NEXT-HOUR precipitation")

print("=" * 70)