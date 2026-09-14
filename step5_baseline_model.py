import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load data
df = pl.read_csv("data/nyc_weather_2000_2023_rich.csv").sort("time")

# 2. Build Features & Target with STRICT temporal boundaries (no leakage!)
df_featured = df.with_columns([
    # TARGET
    pl.col("temperature_2m_max (°C)").alias("target_today_max_temp"),
    
    # FEATURES (All strictly shifted by at least 1 day!)
    pl.col("temperature_2m_max (°C)").shift(1).alias("feature_temp_lag1"),
    pl.col("temperature_2m_max (°C)").shift(2).alias("feature_temp_lag2"),
    
    # 7-day rolling avg of PAST temperatures (shift 1 first, then rolling mean 7)
    pl.col("temperature_2m_max (°C)").shift(1).rolling_mean(window_size=7).alias("feature_temp_7d_avg"),
    
    # Solar radiation from yesterday
    pl.col("shortwave_radiation_sum (MJ/m²)").shift(1).alias("feature_radiation_lag1"),
    
    # Wind speed from yesterday
    pl.col("wind_speed_10m_max (km/h)").shift(1).alias("feature_wind_lag1"),
])

# 3. Clean initial NaN/Null rows caused by lag/rolling windows
df_clean = df_featured.drop_nulls()

# 4. Strict Time-Ordered Split (Train: 2000-2018 | Test: 2019-2023)
train_df = df_clean.filter(pl.col("time") < "2019-01-01")
test_df = df_clean.filter(pl.col("time") >= "2019-01-01")

feature_cols = [
    "feature_temp_lag1", 
    "feature_temp_lag2", 
    "feature_temp_7d_avg", 
    "feature_radiation_lag1", 
    "feature_wind_lag1"
]
target_col = "target_today_max_temp"

X_train = train_df.select(feature_cols).to_numpy()
y_train = train_df.select(target_col).to_numpy().ravel()

X_test = test_df.select(feature_cols).to_numpy()
y_test = test_df.select(target_col).to_numpy().ravel()

print(f"Train Dataset Size: {X_train.shape[0]} rows (2000 to 2018)")
print(f"Test Dataset Size:  {X_test.shape[0]} rows (2019 to 2023)")

# 5. Scaling: Fit ONLY on Train, Transform both Train & Test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on Train
X_test_scaled = scaler.transform(X_test)        # Transform Test using Train mean & std

# 6. Train simple baseline model (Untuned Linear Regression)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7. Evaluate Baseline Model on UNSEEN Test Set
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- BASELINE MODEL TEST RESULTS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")

print("\n--- SAMPLE PREDICTIONS (First 5 Days of Test Set) ---")
test_sample = test_df.select(["time", target_col]).head(5).with_columns(
    pl.Series("predicted_max_temp", np.round(y_pred[:5], 2))
)
print(test_sample)
