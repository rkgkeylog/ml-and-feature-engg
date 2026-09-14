import polars as pl

# 1. Load data
df = pl.read_csv("data/nyc_weather_2000_2023_rich.csv").sort("time")

# 2. Construct Target, Valid Lag Features, and Leaked Features
df_features = df.select([
    pl.col("time"),
    
    # TARGET we want to predict at 6:00 AM on 'time'
    pl.col("temperature_2m_max (°C)").alias("TARGET_today_max_temp"),
    
    #  VALID FEATURE: 1-Day Lag (Yesterday's Max Temp)
    pl.col("temperature_2m_max (°C)").shift(1).alias("feature_lag1_yesterday_temp"),
    
    #  VALID FEATURE: 2-Day Lag (2 Days Ago Max Temp)
    pl.col("temperature_2m_max (°C)").shift(2).alias("feature_lag2_two_days_ago_temp"),
    
    # ❌ LEAKED FEATURE: Tomorrow's Max Temp (Negative Shift = Future Leakage)
    pl.col("temperature_2m_max (°C)").shift(-1).alias("LEAKED_tomorrow_temp"),
])

print("--- TARGET VS VALID LAGS VS LEAKED FEATURES ---")
print(df_features.head(7))
