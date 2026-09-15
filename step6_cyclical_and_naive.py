import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load data & sort
df = pl.read_csv("data/nyc_weather_2000_2023_rich.csv").sort("time")

# 2. Convert 'time' to Date type for day-of-year calculations
df = df.with_columns(pl.col("time").str.to_date("%Y-%m-%d"))

# 3. Construct Features & Target
df_featured = df.with_columns([
    # TARGET: Today's Max Temp
    pl.col("temperature_2m_max (°C)").alias("target_today_max_temp"),
    
    # NAIVE PERSISTENCE FEATURE: Yesterday's Max Temp
    pl.col("temperature_2m_max (°C)").shift(1).alias("naive_yesterday_temp"),
    
    # STEP 5 FEATURES
    pl.col("temperature_2m_max (°C)").shift(1).alias("feature_temp_lag1"),
    pl.col("temperature_2m_max (°C)").shift(2).alias("feature_temp_lag2"),
    pl.col("temperature_2m_max (°C)").shift(1).rolling_mean(7).alias("feature_temp_7d_avg"),
    pl.col("shortwave_radiation_sum (MJ/m²)").shift(1).alias("feature_radiation_lag1"),
    pl.col("wind_speed_10m_max (km/h)").shift(1).alias("feature_wind_lag1"),
    
    # STEP 6 CYCLICAL FEATURES (Sine & Cosine of Day of Year)
    (2 * np.pi * pl.col("time").dt.ordinal_day() / 365.25).sin().alias("feature_sin_doy"),
    (2 * np.pi * pl.col("time").dt.ordinal_day() / 365.25).cos().alias("feature_cos_doy"),
])

# 4. Clean initial NaN/Null rows
df_clean = df_featured.drop_nulls()

# 5. Temporal Train/Test Split (Train: 2000-2018 | Test: 2019-2023)
train_df = df_clean.filter(pl.col("time") < pl.date(2019, 1, 1))
test_df = df_clean.filter(pl.col("time") >= pl.date(2019, 1, 1))

# Extract Target arrays
y_train = train_df.select("target_today_max_temp").to_numpy().ravel()
y_test = test_df.select("target_today_max_temp").to_numpy().ravel()

# --- MODEL 1: NAIVE PERSISTENCE MODEL (Tomorrow = Yesterday's Temp) ---
y_pred_naive = test_df.select("naive_yesterday_temp").to_numpy().ravel()
mae_naive = mean_absolute_error(y_test, y_pred_naive)
rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))

# --- MODEL 2: STEP 5 BASELINE (Linear Regression without Cyclical) ---
cols_step5 = ["feature_temp_lag1", "feature_temp_lag2", "feature_temp_7d_avg", "feature_radiation_lag1", "feature_wind_lag1"]
scaler_5 = StandardScaler()
X_tr_5 = scaler_5.fit_transform(train_df.select(cols_step5).to_numpy())
X_te_5 = scaler_5.transform(test_df.select(cols_step5).to_numpy())

model_step5 = LinearRegression()
model_step5.fit(X_tr_5, y_train)
y_pred_step5 = model_step5.predict(X_te_5)
mae_step5 = mean_absolute_error(y_test, y_pred_step5)
rmse_step5 = np.sqrt(mean_squared_error(y_test, y_pred_step5))

# --- MODEL 3: STEP 6 CYCLICAL (Linear Regression with Sin/Cos) ---
cols_step6 = cols_step5 + ["feature_sin_doy", "feature_cos_doy"]
scaler_6 = StandardScaler()
X_tr_6 = scaler_6.fit_transform(train_df.select(cols_step6).to_numpy())
X_te_6 = scaler_6.transform(test_df.select(cols_step6).to_numpy())

model_step6 = LinearRegression()
model_step6.fit(X_tr_6, y_train)
y_pred_step6 = model_step6.predict(X_te_6)
mae_step6 = mean_absolute_error(y_test, y_pred_step6)
rmse_step6 = np.sqrt(mean_squared_error(y_test, y_pred_step6))

# --- DISPLAY COMPARATIVE RESULTS ---
print("===============================================================")
print("         MODEL PERFORMANCE COMPARISON (TEST SET: 2019-2023)    ")
print("===============================================================")
print(f"1. Naive Persistence (Tomorrow = Today): MAE = {mae_naive:.3f} °C | RMSE = {rmse_naive:.3f} °C")
print(f"2. Step 5 Baseline Linear Model:         MAE = {mae_step5:.3f} °C | RMSE = {rmse_step5:.3f} °C")
print(f"3. Step 6 Cyclical Linear Model:         MAE = {mae_step6:.3f} °C | RMSE = {rmse_step6:.3f} °C")
print("---------------------------------------------------------------")
print(f"Lift of Step 5 Model over Naive Persistence:  {(mae_naive - mae_step5):.3f} °C improvement")
print(f"Lift of Cyclical Features over Step 5 Model:  {(mae_step5 - mae_step6):.3f} °C improvement")
print("===============================================================")
