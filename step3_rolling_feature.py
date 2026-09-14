import polars as pl

# 1. Load data
df = pl.read_csv("data/nyc_weather_2000_2023_rich.csv")

# 2. Sort by time (mandatory for time-series window operations)
df = df.sort("time")

# 3. Compute 7-day rolling average of max temperature
df_featured = df.with_columns(
    pl.col("temperature_2m_max (°C)")
    .rolling_mean(window_size=7)
    .alias("temp_max_7d_rolling_avg")
)

# 4. Display sample rows showing raw temp vs 7-day rolling average
sample = df_featured.select([
    "time", 
    "temperature_2m_max (°C)", 
    "temp_max_7d_rolling_avg"
]).head(10)

print("--- RAW TEMP vs 7-DAY ROLLING AVERAGE (First 10 Days) ---")
print(sample)
