import base64
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global matplotlib style for sleek enterprise dark-theme charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# ---------------------------------------------------------
# 1. GENERATE EXPERIMENTAL DATASETS & MATPLOTLIB CHARTS
# ---------------------------------------------------------
np.random.seed(42)
days = np.arange(1, 61)

# --- Plot 1: Rolling Mean (Denoising & Macro Trend) ---
raw_trend = 50 + (days * 0.8) + np.random.randint(-8, 9, size=60)
df1 = pd.DataFrame({'Day': days, 'Raw': raw_trend})
df1['Roll_Mean_3D'] = df1['Raw'].shift(1).rolling(3).mean()
df1['Roll_Mean_7D'] = df1['Raw'].shift(1).rolling(7).mean()

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
ax.plot(df1['Day'], df1['Raw'], color='#94a3b8', linestyle='--', linewidth=1.5, label='Raw Observations (Noisy Sensor/Price)')
ax.plot(df1['Day'], df1['Roll_Mean_3D'], color='#0284c7', linewidth=2.2, label='3-Day Rolling Mean (Responsive Smoothing)')
ax.plot(df1['Day'], df1['Roll_Mean_7D'], color='#ea580c', linewidth=2.6, label='7-Day Rolling Mean (Macro Trendline)')
ax.set_title('1. Rolling Mean: Denoising High-Frequency Jitter & Isolating Trend', fontweight='bold', fontsize=12, pad=10)
ax.set_xlabel('Day Index (t)', fontsize=10)
ax.set_ylabel('Metric Value', fontsize=10)
ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart1.png')
plt.close()

# --- Plot 2: Rolling StdDev (Volatility & Regime Shifts) ---
regime_vals = np.concatenate([100 + np.random.randint(-2, 3, size=30), 100 + np.random.randint(-18, 19, size=30)])
df2 = pd.DataFrame({'Day': days, 'Value': regime_vals})
df2['Roll_Std_7D'] = df2['Value'].shift(1).rolling(7).std(ddof=1)
df2['Roll_Mean_7D'] = df2['Value'].shift(1).rolling(7).mean()

fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=150)
ax1.plot(df2['Day'], df2['Value'], color='#2563eb', alpha=0.75, linewidth=1.5, label='Raw Value (Oscillating around 100)')
ax1.plot(df2['Day'], df2['Roll_Mean_7D'], color='#475569', linestyle='-', linewidth=2, label='Rolling Mean (Baseline ~100)')
ax1.set_ylabel('Raw Value & Rolling Mean', fontsize=10)
ax1.set_xlabel('Day Index (t)', fontsize=10)

ax2 = ax1.twinx()
ax2.plot(df2['Day'], df2['Roll_Std_7D'], color='#dc2626', linewidth=2.5, label='7D Rolling StdDev (Volatility Signal)')
ax2.set_ylabel('Rolling Standard Deviation', color='#dc2626', fontsize=10)
ax2.tick_params(axis='y', labelcolor='#dc2626')

ax1.axvline(31, color='#0f172a', linestyle=':', linewidth=2, label='Regime Shift Transition (Day 31)')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
ax1.set_title('2. Rolling StdDev: Exposing Hidden Instability & Regime Transitions', fontweight='bold', fontsize=12, pad=10)
ax1.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart2.png')
plt.close()

# --- Plot 3: Min / Max Envelopes & Breakouts ---
trend_breakout = np.concatenate([
    100 + np.random.randint(-5, 6, size=33),
    125 + np.random.randint(-3, 4, size=12),
    90 + np.random.randint(-4, 5, size=15)
])
df3 = pd.DataFrame({'Day': days, 'Val': trend_breakout})
df3['Max_10D'] = df3['Val'].shift(1).rolling(10).max()
df3['Min_10D'] = df3['Val'].shift(1).rolling(10).min()

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
ax.plot(df3['Day'], df3['Val'], label='Live Value at Time t', color='#0f172a', linewidth=2)
ax.plot(df3['Day'], df3['Max_10D'], label='10D Rolling Max (Resistance Ceiling)', color='#ea580c', linestyle='--', linewidth=1.8)
ax.plot(df3['Day'], df3['Min_10D'], label='10D Rolling Min (Support Floor)', color='#0284c7', linestyle='--', linewidth=1.8)
ax.fill_between(df3['Day'], df3['Min_10D'], df3['Max_10D'], color='#fed7aa', alpha=0.35, label='Dynamic Operating Corridor')
ax.axvline(34, color='#16a34a', linestyle=':', linewidth=2, label='Bullish Breakout (Day 34: Val > Max)')
ax.axvline(46, color='#dc2626', linestyle=':', linewidth=2, label='Bearish Breakdown (Day 46: Val < Min)')
ax.set_title('3. Dynamic Operating Envelopes: Min/Max Boundaries & Live Breakout Detection', fontweight='bold', fontsize=12, pad=10)
ax.set_xlabel('Day Index (t)', fontsize=10)
ax.set_ylabel('Metric Value', fontsize=10)
ax.set_ylim(75, 140)
ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart3.png')
plt.close()

# --- Plot 4: Rolling Sum (Cumulative Memory & Saturation) ---
vol = np.random.randint(5, 15, size=60)
vol[29:36] = np.random.randint(25, 36, size=7)
df4 = pd.DataFrame({'Day': days, 'Daily_Volume': vol})
df4['Roll_Sum_7D'] = df4['Daily_Volume'].shift(1).rolling(7).sum()

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
ax.bar(df4['Day'], df4['Daily_Volume'], color='#93c5fd', width=0.8, alpha=0.8, label='Daily Volume (Discrete Daily Load)')
ax.plot(df4['Day'], df4['Roll_Sum_7D'], color='#1d4ed8', linewidth=2.5, label='7D Rolling Sum (Accumulated System Memory)')
ax.axhline(120, color='#dc2626', linestyle='--', linewidth=2, label='Capacity / Fraud Exhaustion Threshold (120)')
ax.set_title('4. Cumulative Memory: Rolling Sum Detecting Saturation & Threshold Breaches', fontweight='bold', fontsize=12, pad=10)
ax.set_xlabel('Day Index (t)', fontsize=10)
ax.set_ylabel('Volume / Units', fontsize=10)
ax.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart4.png')
plt.close()

# --- Plot 5: Cyclical Time Features (Sine/Cosine vs Raw Linear Day) ---
doy = np.arange(1, 366)
sin_doy = np.sin(2 * np.pi * doy / 365.0)
cos_doy = np.cos(2 * np.pi * doy / 365.0)

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
ax.plot(doy, sin_doy, color='#0284c7', linewidth=2.2, label='Sin(Day of Year) - East-West Harmonic')
ax.plot(doy, cos_doy, color='#ea580c', linewidth=2.2, label='Cos(Day of Year) - North-South Harmonic')
ax.scatter([1, 365], [sin_doy[0], sin_doy[-1]], color='#dc2626', s=70, zorder=5, label='Dec 31 & Jan 1 Distance = 0.017 (Adjacent!)')
ax.set_title('5. Cyclical Time Features: Sine & Cosine Encoding (Continuous Time Loop)', fontweight='bold', fontsize=12, pad=10)
ax.set_xlabel('Day of Year (1 to 365)', fontsize=10)
ax.set_ylabel('Transformed Coordinate Value [-1, +1]', fontsize=10)
ax.axvline(1, color='#94a3b8', linestyle=':', alpha=0.7)
ax.axvline(365, color='#94a3b8', linestyle=':', alpha=0.7)
ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart5.png')
plt.close()

# --- Plot 6: Domain Interactions (Diurnal Temp Range & Heat Index Gap) ---
t_max = 20 + 10 * np.sin(2 * np.pi * days / 30) + np.random.normal(0, 2, size=60)
t_min = t_max - np.random.uniform(5, 18, size=60)
dtr = t_max - t_min

fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=150)
ax1.plot(days, t_max, color='#ef4444', linewidth=1.8, label='Raw Max Temp (°C)')
ax1.plot(days, t_min, color='#3b82f6', linewidth=1.8, label='Raw Min Temp (°C)')
ax1.set_ylabel('Raw Temperatures (°C)', fontsize=10)
ax1.set_xlabel('Day Index (t)', fontsize=10)

ax2 = ax1.twinx()
ax2.bar(days, dtr, color='#8b5cf6', alpha=0.35, width=0.7, label='Engineered Feature: Diurnal Range (DTR)')
ax2.set_ylabel('Diurnal Temp Range (°C)', color='#7c3aed', fontsize=10)
ax2.tick_params(axis='y', labelcolor='#7c3aed')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
ax1.set_title('6. Domain Interactions: Constructing Physical Feature Columns (DTR)', fontweight='bold', fontsize=12, pad=10)
ax1.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart6.png')
plt.close()

# --- Plot 7: Mathematical Transformations (Log Scaling Skewed Data) ---
skewed_precip = np.random.exponential(scale=2.5, size=200)
skewed_precip[skewed_precip < 1.5] = 0.0  # Many zero rain days
skewed_precip[::15] = np.random.uniform(20, 80, size=len(skewed_precip[::15])) # Torrential rain spikes
log_precip = np.log1p(skewed_precip)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
ax1.hist(skewed_precip, bins=25, color='#f97316', edgecolor='#c2410c', alpha=0.85)
ax1.set_title('Raw Precipitation (Severe Right-Skew)', fontweight='bold', fontsize=10)
ax1.set_xlabel('Precipitation (mm)', fontsize=9)
ax1.set_ylabel('Frequency Count', fontsize=9)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.hist(log_precip, bins=25, color='#10b981', edgecolor='#047857', alpha=0.85)
ax2.set_title('Log Transformed: log1p(Precipitation)', fontweight='bold', fontsize=10)
ax2.set_xlabel('Transformed Scale log(1 + mm)', fontsize=9)
ax2.set_ylabel('Frequency Count', fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.6)

fig.suptitle('7. Mathematical Transformations: Converting Skewed Distributions to Gaussian Curves', fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig('tutorial_chart7.png')
plt.close()

# Helper function to convert images to Base64 and clean up temp files
def to_b64_and_cleanup(path):
    with open(path, 'rb') as f:
        b64_str = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    if os.path.exists(path):
        os.remove(path)
    return b64_str

img1_b64 = to_b64_and_cleanup('tutorial_chart1.png')
img2_b64 = to_b64_and_cleanup('tutorial_chart2.png')
img3_b64 = to_b64_and_cleanup('tutorial_chart3.png')
img4_b64 = to_b64_and_cleanup('tutorial_chart4.png')
img5_b64 = to_b64_and_cleanup('tutorial_chart5.png')
img6_b64 = to_b64_and_cleanup('tutorial_chart6.png')
img7_b64 = to_b64_and_cleanup('tutorial_chart7.png')

# ---------------------------------------------------------
# 2. BUILD COMPREHENSIVE MASTER TUTORIAL HTML
# ---------------------------------------------------------
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Master Architecture & Field Guide to Feature Engineering in Machine Learning</title>
    <style>
        :root {{
            --bg-body: #0b1120;
            --bg-card: #1e293b;
            --bg-inner: #0f172a;
            --border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-body: #cbd5e1;
            --accent-cyan: #38bdf8;
            --accent-green: #4ade80;
            --accent-orange: #fb923c;
            --accent-red: #f87171;
            --accent-purple: #c084fc;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-body);
            color: var(--text-body);
            line-height: 1.7;
            padding: 2.5rem 1rem;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{ border-bottom: 1px solid var(--border); padding-bottom: 2rem; margin-bottom: 2.5rem; }}
        .badge {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent-cyan);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 1rem;
            border: 1px solid rgba(56, 189, 248, 0.25);
        }}
        h1 {{ font-size: 2.4rem; font-weight: 800; color: var(--text-primary); line-height: 1.25; margin-bottom: 1rem; }}
        .lead {{ font-size: 1.15rem; color: var(--text-secondary); }}
        
        section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }}
        h2 {{ font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin-bottom: 1rem; border-left: 4px solid var(--accent-cyan); padding-left: 0.75rem; }}
        h3 {{ font-size: 1.15rem; font-weight: 600; color: var(--accent-cyan); margin: 1.25rem 0 0.5rem 0; }}
        p {{ margin-bottom: 1rem; font-size: 1.02rem; }}
        
        .chart-box {{
            background: #ffffff;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 1.5rem 0;
            text-align: center;
        }}
        .chart-box img {{ max-width: 100%; height: auto; border-radius: 6px; }}
        
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-top: 1rem; }}
        @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
        
        .card-inner {{
            background: var(--bg-inner);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
        }}
        .card-inner strong {{ color: var(--text-primary); }}
        
        .formula-box {{
            background: var(--bg-inner);
            border-left: 3px solid var(--accent-orange);
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            font-family: "Fira Code", monospace;
            color: var(--accent-orange);
            margin: 1rem 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.95rem;
        }}
        th, td {{ padding: 0.85rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: var(--bg-inner); color: var(--text-primary); font-weight: 600; }}
        tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
        
        code {{ background: rgba(56, 189, 248, 0.1); color: var(--accent-cyan); padding: 0.2rem 0.4rem; border-radius: 4px; font-family: monospace; font-size: 0.9rem; }}
        pre {{ background: var(--bg-inner); padding: 1.25rem; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); margin: 1rem 0; }}
        pre code {{ background: none; color: var(--text-body); padding: 0; font-size: 0.9rem; }}
        
        .highlight-leakage {{
            background: rgba(248, 113, 113, 0.1);
            border-left: 4px solid var(--accent-red);
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
        }}
        
        footer {{ text-align: center; padding: 2rem 0; color: var(--text-secondary); font-size: 0.9rem; border-top: 1px solid var(--border); }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <span class="badge">Master Feature Engineering Field Guide</span>
        <h1>The Master Architecture Guide to Feature Engineering in Machine Learning</h1>
        <p class="lead">From Rolling Windows & Cyclical Time Waves to Domain Interaction Ratios and Leakage-Free Scaling Pipelines.</p>
    </header>

    <!-- SECTION 1: ROLLING FEATURES -->
    <section>
        <h2>1. Rolling Windows: Denoising, Volatility, Envelopes & Memory</h2>
        <p>A rolling feature extracts moving statistical signals over a lookback window $W$. To guarantee <strong>zero data leakage</strong>, rolling features must strictly shift by 1 step (<code>shift(1)</code>) so the current row's target is never visible during window aggregation.</p>
        
        <div class="grid-2">
            <div class="card-inner">
                <h3>A. Rolling Mean (Denoising)</h3>
                <p>Filters out high-frequency noise and isolates macro trends.</p>
                <div class="formula-box">Mean = (1 / K) * ∑ x_(t-i)</div>
            </div>
            <div class="card-inner">
                <h3>B. Rolling StdDev (Volatility)</h3>
                <p>Detects market regime shifts, stability, and risk jumps.</p>
                <div class="formula-box">Std = √[ (1 / (K-1)) * ∑ (x - µ)² ]</div>
            </div>
        </div>
        
        <div class="chart-box"><img src="{img1_b64}" alt="Rolling Mean Chart"></div>
        <div class="chart-box"><img src="{img2_b64}" alt="Rolling StdDev Chart"></div>

        <div class="grid-2">
            <div class="card-inner">
                <h3>C. Rolling Min/Max (Dynamic Corridor)</h3>
                <p>Establishes resistance ceilings and support floors to trigger breakout alerts.</p>
            </div>
            <div class="card-inner">
                <h3>D. Rolling Sum (Cumulative Saturation)</h3>
                <p>Tracks system memory (e.g. soil moisture, fraud spend limits, API quotas).</p>
            </div>
        </div>

        <div class="chart-box"><img src="{img3_b64}" alt="Rolling Min/Max Chart"></div>
        <div class="chart-box"><img src="{img4_b64}" alt="Rolling Sum Chart"></div>
    </section>

    <!-- SECTION 2: CYCLICAL TIME FEATURES -->
    <section>
        <h2>2. Cyclical Time Features: Encoding Time as a Continuous Circle</h2>
        <p>Calendar metrics like <code>Day of Year</code> (1 to 365) or <code>Hour of Day</code> (0 to 23) are naturally cyclical. Naive numeric representations create an artificial 364-unit gap between Dec 31st and Jan 1st. We project these variables onto a unit circle using <strong>Sine</strong> and <strong>Cosine</strong> components.</p>
        
        <div class="formula-box">
            sin_time = sin(2 * π * time / Period)<br>
            cos_time = cos(2 * π * time / Period)
        </div>
        
        <div class="chart-box"><img src="{img5_b64}" alt="Cyclical Time Chart"></div>
        
        <div class="card-inner">
            <strong>Why Year is Excluded from Sine/Cosine:</strong>
            <p>Months and hours repeat in loops ($12 \rightarrow 1$). <code>Year</code> is linear and monotonic ($2023 \rightarrow 2024$). Year measures long-term secular drift (e.g., climate change), whereas Sine/Cosine measures annual seasonality.</p>
        </div>
    </section>

    <!-- SECTION 3: DOMAIN INTERACTIONS -->
    <section>
        <h2>3. Domain Interaction Features: Creating Physical Feature Columns</h2>
        <p>Machine learning models perform significantly better when we combine raw variables into high-level physical formulas rather than forcing decision trees to rediscover physics from scratch.</p>
        
        <div class="chart-box"><img src="{img6_b64}" alt="Domain Interactions Chart"></div>

        <table>
            <thead>
                <tr>
                    <th>Engineered Feature Column</th>
                    <th>Formula</th>
                    <th>Physical Signal to ML Model</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Diurnal Temp Range (DTR)</strong></td>
                    <td><code>temp_max - temp_min</code></td>
                    <td>Clear-sky radiational cooling vs overcast cloud cover</td>
                </tr>
                <tr>
                    <td><strong>Apparent Heat Gap</strong></td>
                    <td><code>apparent_temp - raw_temp</code></td>
                    <td>Human heat index & wind-chill humidity distortion</td>
                </tr>
                <tr>
                    <td><strong>Water Balance Ratio</strong></td>
                    <td><code>precipitation / (evapotranspiration + 1e-5)</code></td>
                    <td>Net soil hydration & drought saturation index</td>
                </tr>
            </tbody>
        </table>
    </section>

    <!-- SECTION 4: MATHEMATICAL TRANSFORMS -->
    <section>
        <h2>4. Mathematical Transformations: Fixing Skewed Distributions</h2>
        <p>Variables like rainfall, snowfall, or financial transaction amounts are heavily right-skewed with extreme zero-inflation. Linear models and neural networks fail on these. We apply log or power transformations to compress extreme spikes into bell-shaped Gaussian curves.</p>
        
        <div class="chart-box"><img src="{img7_b64}" alt="Mathematical Transforms Chart"></div>

        <div class="grid-2">
            <div class="card-inner">
                <h3>Log Transform (<code>log1p</code>)</h3>
                <p><code>y = ln(1 + x)</code>: Compresses exponential Spikes while preserving 0s safely.</p>
            </div>
            <div class="card-inner">
                <h3>Yeo-Johnson Power Transform</h3>
                <p>Optimizes power parameter $\\lambda$ to normalize distributions, supporting <strong>negative numbers</strong> (e.g. sub-zero temperatures).</p>
            </div>
        </div>
    </section>

    <!-- SECTION 5: CATEGORICAL ENCODING TAXONOMY -->
    <section>
        <h2>5. Categorical Encoding Taxonomy</h2>
        <p>Translating non-numeric text labels into model-digestible numerical matrices.</p>

        <table>
            <thead>
                <tr>
                    <th>Encoding Technique</th>
                    <th>Mechanism</th>
                    <th>Best Used For</th>
                    <th>Leakage Guardrail</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>One-Hot Encoding</strong></td>
                    <td>Creates binary 0/1 indicator columns per category</td>
                    <td>Low cardinality (&lt; 10 categories: Season, Day)</td>
                    <td>None (Independent of Target)</td>
                </tr>
                <tr>
                    <td><strong>Ordinal Encoding</strong></td>
                    <td>Assigns ordered integers (1, 2, 3)</td>
                    <td>Categories with natural rank (Light, Severe, Extreme)</td>
                    <td>None (Independent of Target)</td>
                </tr>
                <tr>
                    <td><strong>Frequency Encoding</strong></td>
                    <td>Replaces category with its occurrence count</td>
                    <td>High cardinality (Zip codes, User IDs)</td>
                    <td>None (Independent of Target)</td>
                </tr>
                <tr>
                    <td><strong>Target Encoding</strong></td>
                    <td>Replaces category with mean of Target ($y$)</td>
                    <td>Complex high-dimensional tabular data</td>
                    <td><strong style="color: var(--accent-red);">MUST use K-Fold Out-of-Fold on Train set only!</strong></td>
                </tr>
            </tbody>
        </table>
    </section>

    <!-- SECTION 6: PRODUCTION PIPELINE CODE -->
    <section>
        <h2>6. Master Production Feature Engineering Pipeline (Polars & Pandas)</h2>
        <p>A complete, leak-free feature engineering pipeline implementing Lags, Rolling Windows, Cyclical Waves, and Physical Interactions.</p>

        <pre><code>import polars as pl
import numpy as np

def build_master_feature_pipeline(df: pl.DataFrame) -> pl.DataFrame:
    \"\"\"
    Production-grade, leak-free feature pipeline in Polars.
    Guarantees strict temporal boundaries (shift(1)) and zero future leakage.
    \"\"\"
    return (
        df.sort("time")
        .with_columns([
            # 1. TARGET
            pl.col("temperature_2m_max (°C)").alias("target_today_max_temp"),

            # 2. LAG FEATURES (Strictly shift 1+)
            pl.col("temperature_2m_max (°C)").shift(1).alias("temp_lag1"),
            pl.col("temperature_2m_max (°C)").shift(2).alias("temp_lag2"),

            # 3. ROLLING FEATURES (Shifted first!)
            pl.col("temperature_2m_max (°C)").shift(1).rolling_mean(7).alias("temp_roll_mean_7"),
            pl.col("temperature_2m_max (°C)").shift(1).rolling_std(14).alias("temp_roll_std_14"),
            pl.col("temperature_2m_max (°C)").shift(1).rolling_max(30).alias("temp_roll_max_30"),

            # 4. CYCLICAL TIME FEATURES (Sine & Cosine)
            (2 * np.pi * pl.col("time").dt.ordinal_day() / 365.0).sin().alias("sin_day_of_year"),
            (2 * np.pi * pl.col("time").dt.ordinal_day() / 365.0).cos().alias("cos_day_of_year"),

            # 5. DOMAIN PHYSICAL INTERACTIONS
            (pl.col("temperature_2m_max (°C)") - pl.col("temperature_2m_min (°C)")).alias("dtr_range"),
            (pl.col("apparent_temperature_max (°C)") - pl.col("temperature_2m_max (°C)")).alias("heat_index_gap"),

            # 6. SKEWNESS CORRECTION (Log transform)
            (pl.col("precipitation_sum (mm)") + 1.0).log().alias("log_precipitation")
        ])
        .drop_nulls()
    )
</code></pre>
    </section>

    <footer>
        <p>The Master Architecture & Field Guide to Feature Engineering &bull; Production ML Documentation &bull; Google Deepmind Pair Programming Artifact</p>
    </footer>
</div>

</body>
</html>
"""

# Save HTML to root index.html, resources/, and manual-exp/ for GitHub Pages & local access
targets = [
    "index.html",
    "resources/index.html",
    "resources/rolling_features_tutorial_guide.html",
    "manual-exp/rolling_features_tutorial_guide.html"
]

for target in targets:
    with open(target, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated {target} ({os.path.getsize(target)} bytes)")