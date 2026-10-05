import pandas as pd


# Historical weather dataset
file_path = "data/singida_weather_historical.csv"


print("Loading historical weather dataset...")
print()


# Load dataset
weather_data = pd.read_csv(file_path)


# 1. Dataset size
print("===== DATASET SIZE =====")
print("Number of rows:", weather_data.shape[0])
print("Number of columns:", weather_data.shape[1])
print()


# 2. Columns
print("===== COLUMNS =====")
for column in weather_data.columns:
    print("-", column)
print()


# 3. Date range
print("===== DATE RANGE =====")

weather_data["time"] = pd.to_datetime(
    weather_data["time"]
)

print("First date:", weather_data["time"].min())
print("Last date:", weather_data["time"].max())
print()


# 4. First records
print("===== FIRST 5 RECORDS =====")
print(weather_data.head())
print()


# 5. Last records
print("===== LAST 5 RECORDS =====")
print(weather_data.tail())
print()


# 6. Data types
print("===== DATA TYPES =====")
print(weather_data.dtypes)
print()


# 7. Missing values
print("===== MISSING VALUES =====")

missing_values = weather_data.isnull().sum()

print(missing_values)
print()


# 8. Duplicate records
print("===== DUPLICATE RECORDS =====")

duplicates = weather_data.duplicated().sum()

print("Number of duplicate rows:", duplicates)
print()


# 9. Statistical summary
print("===== STATISTICAL SUMMARY =====")

print(
    weather_data.describe()
)
print()


# 10. Rain statistics
print("===== RAIN STATISTICS =====")

rain_count = (weather_data["rain"] > 0).sum()
no_rain_count = (weather_data["rain"] == 0).sum()

print("Hours with rain:", rain_count)
print("Hours without rain:", no_rain_count)
print()


print("===== DATA INSPECTION COMPLETED =====")