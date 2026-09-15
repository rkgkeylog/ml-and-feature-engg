import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load data
df = pl.read_csv("data/nyc_weather_2000_2023_rich.csv").sort("time")
df = df.with_columns(pl.col("time").str.to_date("%Y-%m-%d"))

# 2. Features & Target
df_featured = df.with_columns([
    pl.col("temperature_2m_max (°C)").alias("target_today_max_temp"),
    pl.col("temperature_2m_max (°C)").shift(1).alias("feature_temp_lag1"),
    pl.col("temperature_2m_max (°C)").shift(2).alias("feature_temp_lag2"),
    pl.col("temperature_2m_max (°C)").shift(1).rolling_mean(7).alias("feature_temp_7d_avg"),
    pl.col("shortwave_radiation_sum (MJ/m²)").shift(1).alias("feature_radiation_lag1"),
    pl.col("wind_speed_10m_max (km/h)").shift(1).alias("feature_wind_lag1"),
    (2 * np.pi * pl.col("time").dt.ordinal_day() / 365.25).sin().alias("feature_sin_doy"),
    (2 * np.pi * pl.col("time").dt.ordinal_day() / 365.25).cos().alias("feature_cos_doy"),
]).drop_nulls()

train_df = df_featured.filter(pl.col("time") < pl.date(2019, 1, 1))

feature_cols = [
    "feature_temp_lag1", 
    "feature_temp_lag2", 
    "feature_temp_7d_avg", 
    "feature_radiation_lag1", 
    "feature_wind_lag1",
    "feature_sin_doy",
    "feature_cos_doy"
]

X_train = train_df.select(feature_cols).to_numpy()
y_train = train_df.select("target_today_max_temp").to_numpy().ravel()

# Scale features so coefficients are directly comparable in magnitude
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Display Intercept and Normalized Coefficients
print("===============================================================")
print(f"MODEL INTERCEPT (Baseline Mean Temp): {model.intercept_:.3f} °C")
print("===============================================================")
print("  FEATURE WEIGHT (COEFFICIENT) INSPECTION (Standardized Features)")
print("===============================================================")

coef_table = pl.DataFrame({
    "Feature Name": feature_cols,
    "Coefficient (Weight)": np.round(model.coef_, 4),
    "Absolute Importance": np.round(np.abs(model.coef_), 4)
}).sort("Absolute Importance", descending=True)

print(coef_table)
