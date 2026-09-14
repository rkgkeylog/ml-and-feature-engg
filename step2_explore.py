import polars as pl

# 1. Load data
file_path = "data/nyc_weather_2000_2023_rich.csv"
df = pl.read_csv(file_path)

print("--- 1. SCHEMA & TYPES ---")
print(df.schema)

print("\n--- 2. FIRST 5 ROWS ---")
print(df.head(5))

print("\n--- 3. MISSING VALUES (NULL COUNTS) ---")
null_counts = df.null_count()
print(null_counts)

print("\n--- 4. DATE RANGE & GAP CHECK ---")
# Parse 'time' column as Date
df = df.with_columns(pl.col("time").str.to_date("%Y-%m-%d"))

min_date = df["time"].min()
max_date = df["time"].max()
total_rows = len(df)

# Calculate total expected days in range
expected_days = (max_date - min_date).days + 1

print(f"Start Date: {min_date}")
print(f"End Date:   {max_date}")
print(f"Actual Rows (Days recorded):   {total_rows}")
print(f"Expected Days in Date Range:   {expected_days}")
print(f"Missing Days in Time Series:   {expected_days - total_rows}")

print("\n--- 5. DESCRIPTIVE STATISTICS (MIN/MAX/MEAN/OUTLIERS) ---")
print(df.describe())
