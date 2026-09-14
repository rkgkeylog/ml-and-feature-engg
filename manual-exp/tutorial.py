import base64
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Generate Synthetic Experimental Datasets
np.random.seed(42)
days = np.arange(1, 61)

# Plot 1: Mean (Denoising & Directional Macro Trend)
raw_trend = 50 + (days * 0.8) + np.random.randint(-8, 9, size=60)
df1 = pd.DataFrame({'Day': days, 'Raw': raw_trend})
df1['Roll_Mean_3D'] = df1['Raw'].shift(1).rolling(3).mean()
df1['Roll_Mean_7D'] = df1['Raw'].shift(1).rolling(7).mean()

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
ax.plot(
    df1['Day'],
    df1['Raw'],
    color='#94a3b8',
    linestyle='--',
    linewidth=1.5,
    label='Raw Observations (Noisy Sensor/Price)',
)
ax.plot(
    df1['Day'],
    df1['Roll_Mean_3D'],
    color='#0284c7',
    linewidth=2.2,
    label='3-Day Rolling Mean (Responsive Smoothing)',
)
ax.plot(
    df1['Day'],
    df1['Roll_Mean_7D'],
    color='#ea580c',
    linewidth=2.6,
    label='7-Day Rolling Mean (Macro Trendline)',
)
ax.set_title(
    '1. Rolling Mean: Denoising High-Frequency Jitter & Isolating Trend',
    fontweight='bold',
    fontsize=12,
    pad=10,
)
ax.set_xlabel('Day Index (t)', fontsize=10)
ax.set_ylabel('Metric Value', fontsize=10)
ax.legend(
    loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1'
)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart1.png')
plt.close()

# Plot 2: StdDev (Regime Shifts & Volatility Risk)
regime_vals = np.concatenate([
    100 + np.random.randint(-2, 3, size=30),
    100 + np.random.randint(-18, 19, size=30),
])
df2 = pd.DataFrame({'Day': days, 'Value': regime_vals})
df2['Roll_Std_7D'] = df2['Value'].shift(1).rolling(7).std(ddof=1)
df2['Roll_Mean_7D'] = df2['Value'].shift(1).rolling(7).mean()

fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=150)
ax1.plot(
    df2['Day'],
    df2['Value'],
    color='#2563eb',
    alpha=0.75,
    linewidth=1.5,
    label='Raw Value (Oscillating around 100)',
)
ax1.plot(
    df2['Day'],
    df2['Roll_Mean_7D'],
    color='#475569',
    linestyle='-',
    linewidth=2,
    label='Rolling Mean (Baseline ~100)',
)
ax1.set_ylabel('Raw Value & Rolling Mean', fontsize=10)
ax1.set_xlabel('Day Index (t)', fontsize=10)

ax2 = ax1.twinx()
ax2.plot(
    df2['Day'],
    df2['Roll_Std_7D'],
    color='#dc2626',
    linewidth=2.5,
    label='7D Rolling StdDev (Volatility Signal)',
)
ax2.set_ylabel('Rolling Standard Deviation', color='#dc2626', fontsize=10)
ax2.tick_params(axis='y', labelcolor='#dc2626')

ax1.axvline(
    31,
    color='#0f172a',
    linestyle=':',
    linewidth=2,
    label='Regime Shift Transition (Day 31)',
)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='upper left',
    frameon=True,
    facecolor='#ffffff',
    edgecolor='#cbd5e1',
)
ax1.set_title(
    '2. Rolling StdDev: Exposing Hidden Instability & Regime Transitions',
    fontweight='bold',
    fontsize=12,
    pad=10,
)
ax1.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart2.png')
plt.close()

# Plot 3: Min / Max Envelopes & Breakouts
trend_breakout = np.concatenate([
    100 + np.random.randint(-5, 6, size=33),
    125 + np.random.randint(-3, 4, size=12),
    90 + np.random.randint(-4, 5, size=15),
])
df3 = pd.DataFrame({'Day': days, 'Val': trend_breakout})
df3['Max_10D'] = df3['Val'].shift(1).rolling(10).max()
df3['Min_10D'] = df3['Val'].shift(1).rolling(10).min()

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
ax.plot(
    df3['Day'],
    df3['Val'],
    label='Live Value at Time t',
    color='#0f172a',
    linewidth=2,
)
ax.plot(
    df3['Day'],
    df3['Max_10D'],
    label='10D Rolling Max (Resistance Ceiling)',
    color='#ea580c',
    linestyle='--',
    linewidth=1.8,
)
ax.plot(
    df3['Day'],
    df3['Min_10D'],
    label='10D Rolling Min (Support Floor)',
    color='#0284c7',
    linestyle='--',
    linewidth=1.8,
)
ax.fill_between(
    df3['Day'],
    df3['Min_10D'],
    df3['Max_10D'],
    color='#fed7aa',
    alpha=0.35,
    label='Dynamic Operating Corridor',
)
ax.axvline(
    34,
    color='#16a34a',
    linestyle=':',
    linewidth=2,
    label='Bullish Breakout (Day 34: Val > Max)',
)
ax.axvline(
    46,
    color='#dc2626',
    linestyle=':',
    linewidth=2,
    label='Bearish Breakdown (Day 46: Val < Min)',
)
ax.set_title(
    '3. Dynamic Operating Envelopes: Min/Max Boundaries & Live Breakout'
    ' Detection',
    fontweight='bold',
    fontsize=12,
    pad=10,
)
ax.set_xlabel('Day Index (t)', fontsize=10)
ax.set_ylabel('Metric Value', fontsize=10)
ax.set_ylim(75, 140)
ax.legend(
    loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1'
)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart3.png')
plt.close()

# Plot 4: Cumulative Memory & Saturation (Rolling Sum)
vol = np.random.randint(5, 15, size=60)
vol[29:36] = np.random.randint(25, 36, size=7)
df4 = pd.DataFrame({'Day': days, 'Daily_Volume': vol})
df4['Roll_Sum_7D'] = df4['Daily_Volume'].shift(1).rolling(7).sum()

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
ax.bar(
    df4['Day'],
    df4['Daily_Volume'],
    color='#93c5fd',
    width=0.8,
    alpha=0.8,
    label='Daily Volume (Discrete Daily Load)',
)
ax.plot(
    df4['Day'],
    df4['Roll_Sum_7D'],
    color='#1d4ed8',
    linewidth=2.5,
    label='7D Rolling Sum (Accumulated System Memory)',
)
ax.axhline(
    120,
    color='#dc2626',
    linestyle='--',
    linewidth=2,
    label='Capacity / Fraud Exhaustion Threshold (120)',
)
ax.set_title(
    '4. Cumulative Memory: Rolling Sum Detecting Saturation & Threshold'
    ' Breaches',
    fontweight='bold',
    fontsize=12,
    pad=10,
)
ax.set_xlabel('Day Index (t)', fontsize=10)
ax.set_ylabel('Volume / Units', fontsize=10)
ax.legend(
    loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1'
)
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig.savefig('tutorial_chart4.png')
plt.close()


def to_b64(path):
  with open(path, 'rb') as f:
    return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"


img1_b64 = to_b64('tutorial_chart1.png')
img2_b64 = to_b64('tutorial_chart2.png')
img3_b64 = to_b64('tutorial_chart3.png')
img4_b64 = to_b64('tutorial_chart4.png')

# 2. Build the Full Tutorial HTML
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mastering Rolling Features in Machine Learning: A Visual & Mathematical Tutorial</title>
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
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-body);
            color: var(--text-body);
            line-height: 1.7;
            padding: 2.5rem 1rem;
        }}
        .container {{ max-width: 960px; margin: 0 auto; }}
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
        h1 {{ font-size: 2.3rem; font-weight: 800; color: var(--text-primary); line-height: 1.25; margin-bottom: 1rem; }}
        .lead {{ font-size: 1.15rem; color: var(--text-secondary); }}
        h2 {{ font-size: 1.6rem; color: var(--text-primary); margin: 2.5rem 0 1rem 0; }}
        h3 {{ font-size: 1.2rem; color: var(--accent-cyan); margin: 1.5rem 0 0.5rem 0; }}
        p {{ margin-bottom: 1rem; }}
        strong {{ color: var(--text-primary); }}
        .concept-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }}
        .chart-box {{
            background: #ffffff;
            border-radius: 8px;
            padding: 0.6rem;
            margin: 1.5rem 0;
            border: 1px solid var(--border);
        }}
        .chart-box img {{ width: 100%; height: auto; display: block; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin: 1.2rem 0; }}
        @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
        .sub-box {{
            background: var(--bg-inner);
            padding: 1.2rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .sub-box h4 {{ font-size: 0.95rem; text-transform: uppercase; margin-bottom: 0.5rem; }}
        .sub-box.why h4 {{ color: var(--accent-orange); }}
        .sub-box.ml h4 {{ color: var(--accent-green); }}
        .takeaway-box {{
            background: rgba(56, 189, 248, 0.08);
            border-left: 4px solid var(--accent-cyan);
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1.2rem;
        }}
        .leakage-callout {{
            background: rgba(248, 113, 113, 0.08);
            border: 1px solid rgba(248, 113, 113, 0.3);
            border-left: 4px solid var(--accent-red);
            padding: 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 2rem 0;
        }}
        .leakage-callout h3 {{ color: var(--accent-red); margin-top: 0; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            background: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        th, td {{ padding: 0.9rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: rgba(255, 255, 255, 0.04); color: var(--text-primary); font-weight: 600; }}
        code {{
            font-family: monospace;
            background: rgba(255, 255, 255, 0.1);
            padding: 0.2rem 0.45rem;
            border-radius: 4px;
            color: var(--accent-cyan);
        }}
        pre {{
            background: var(--bg-inner);
            border: 1px solid var(--border);
            padding: 1.25rem;
            border-radius: 8px;
            overflow-x: auto;
            color: #f1f5f9;
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 1.2rem 0;
        }}
        pre code {{ background: none; padding: 0; color: inherit; }}
        ul {{ margin-left: 1.5rem; margin-bottom: 1rem; }}
        li {{ margin-bottom: 0.4rem; }}
        footer {{
            border-top: 1px solid var(--border);
            padding-top: 2rem;
            margin-top: 3.5rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-align: center;
        }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <div class="badge">Machine Learning Feature Engineering Guide</div>
        <h1>Mastering Rolling Features in Machine Learning</h1>
        <p class="lead">Why isolated tabular rows fail, and how trailing statistical windows inject critical context, expose regime changes, and eliminate lookahead data leakage.</p>
    </header>

    <section>
        <h2>Why ML Models Suffer from "Amnesia"</h2>
        <p>Standard tabular machine learning algorithms—such as <strong>XGBoost</strong>, <strong>LightGBM</strong>, <strong>CatBoost</strong>, and feedforward neural networks—treat every row in your dataset as an independent, identically distributed (IID) observation. Unlike sequential networks (RNNs/LSTMs), tree-based algorithms have <strong>no native memory</strong>.</p>
        <p>If you feed a model a naked value—like <code>temperature = 22°C</code> or <code>price = $120</code>—the model cannot tell whether $120 represents an aggressive breakout rally or an ongoing catastrophic market crash. <strong>Rolling features</strong> solve this by sliding a trailing window over past observations, supplying localized context: trend, volatility, envelope boundaries, and memory load.</p>
    </section>

    <div class="leakage-callout">
        <h3>The Golden Rule of Predictive ML: Eliminating Temporal Leakage</h3>
        <p>In standard data analysis, a 7-day rolling window on day <em>t</em> includes day <em>t</em> itself: <code>df['val'].rolling(7).mean()</code>.</p>
        <p>In <strong>predictive machine learning</strong>, doing this creates <strong>target / lookahead leakage</strong>. If day <em>t</em>'s target or current value is part of the feature calculation, your model peeks into the future during training, scoring 99% accuracy offline and failing completely in production.</p>
        <p><strong>The Fix:</strong> All rolling features must strictly lag by at least one step: <code>df['val'].shift(1).rolling(window).agg()</code>. Day <em>t</em> features must strictly reflect information known up to <em>t-1</em>.</p>
    </div>

    <!-- CONCEPT 1: MEAN -->
    <div class="concept-card">
        <h2>1. Denoising & Directional Trend (Rolling Mean)</h2>
        <p>The <strong>Rolling Mean</strong> computes the average of the preceding <code>k</code> time steps, acting as a low-pass filter on continuous data streams.</p>

        <div class="grid-2">
            <div class="sub-box why">
                <h4>What It Signifies</h4>
                <p>Real-world measurements are contaminated with high-frequency noise (weather fluctuations, intraday chop, sensor flicker). The rolling average cancels out noise and exposes the true macro trajectory.</p>
            </div>
            <div class="sub-box ml">
                <h4>Why It Is Crucial for ML</h4>
                <p>Decision trees split on exact values. Raw daily chop leads to spurious splits and severe overfitting. The rolling mean provides the model with the underlying structural direction.</p>
            </div>
        </div>

        <div class="chart-box">
            <img src="{img1_b64}" alt="Rolling Mean Plot">
        </div>

        <div class="takeaway-box">
            <strong>Key Insight from the Visual:</strong> While the dashed gray raw series whips up and down constantly, the <strong>7-Day Rolling Mean (orange curve)</strong> isolates the clean upward drift. In quantitative finance, this is the exact principle behind 50 DMA and 200 DMA trend filters.
        </div>

        <h3>Production ML Feature Formulations:</h3>
        <ul>
            <li><code>trend_momentum = (rolling_mean_7 - rolling_mean_30) / (rolling_mean_30 + 1e-6)</code> (Moving average velocity crossover).</li>
            <li><code>mean_divergence = (current_val - rolling_mean_20) / (rolling_mean_20 + 1e-6)</code> (Measures whether today's price is statistically overextended).</li>
        </ul>
    </div>

    <!-- CONCEPT 2: STD DEV -->
    <div class="concept-card">
        <h2>2. Quantifying Regime Shifts & Risk (Rolling Standard Deviation)</h2>
        <p>The <strong>Rolling Standard Deviation</strong> measures data dispersion around the trailing rolling mean over the last <code>k</code> time steps.</p>

        <div class="grid-2">
            <div class="sub-box why">
                <h4>What It Signifies</h4>
                <p>Systems transition between operating states (regimes). A metric can oscillate around 100 in a calm state (low variance) and suddenly shift into wild turmoil (high variance), even if its average stays identical.</p>
            </div>
            <div class="sub-box ml">
                <h4>Why It Is Crucial for ML</h4>
                <p>A delta of +2 units in a calm market carries a completely different probability distribution than +2 units during high volatility. Rolling StdDev provides the model with a direct metric of uncertainty to widen prediction intervals or alter tree split paths.</p>
            </div>
        </div>

        <div class="chart-box">
            <img src="{img2_b64}" alt="Rolling StdDev Plot">
        </div>

        <div class="takeaway-box">
            <strong>Key Insight from the Visual:</strong> The gray baseline mean stays nearly flat near 100 across the entire 60 days. A model monitoring only averages would conclude nothing changed. However, on Day 31, the <strong>red Rolling StdDev curve</strong> surges from ~1.2 to 14.0, directly capturing the regime shift.
        </div>

        <h3>Production ML Feature Formulations:</h3>
        <ul>
            <li><code>volatility_regime = rolling_std_14</code> (Direct volatility and risk indicator).</li>
            <li><code>rolling_z_score = (current_val - rolling_mean_14) / (rolling_std_14 + 1e-6)</code> (Standardized score: how many standard deviations today deviates from baseline).</li>
        </ul>
    </div>

    <!-- CONCEPT 3: ENVELOPES (MIN / MAX) -->
    <div class="concept-card">
        <h2>3. Dynamic Operating Envelopes (Rolling Min / Max)</h2>
        <p><strong>Rolling Min and Max</strong> track the trailing low and high over <code>k</code> periods, establishing dynamic support floors and resistance ceilings.</p>

        <div class="grid-2">
            <div class="sub-box why">
                <h4>What It Signifies</h4>
                <p>Fixed static thresholds fail as systems scale or drift. An operating envelope creates an adaptive corridor that expands and contracts with the system's trailing historical variance.</p>
            </div>
            <div class="sub-box ml">
                <h4>Why It Is Crucial for ML</h4>
                <p>Enables real-time breakout detection. By freezing the boundary at <code>t-1</code>, today's live incoming value can pierce above the ceiling, giving tree models an unambiguous split: <code>current_val - rolling_max >= 0</code>.</p>
            </div>
        </div>

        <div class="chart-box">
            <img src="{img3_b64}" alt="Dynamic Envelopes Plot">
        </div>

        <div class="takeaway-box">
            <strong>Key Insight from the Visual:</strong> During Days 11–33, the metric oscillates safely within the shaded 10-day corridor (95 to 105). On <strong>Day 34</strong>, the price explodes to 127. Because the ceiling was strictly lagged at 105, the signal <code>Value - Ceiling = +22</code> fires an immediate real-time breakout trigger without lookahead bias.
        </div>

        <h3>Production ML Feature Formulations:</h3>
        <ul>
            <li><code>is_breakout = int(current_val > rolling_max_10)</code> (Binary breakout alert).</li>
            <li><code>envelope_position = (current_val - roll_min) / (roll_max - roll_min + 1e-6)</code> (Dynamic Min-Max oscillator normalizing non-stationary series between 0.0 and 1.0).</li>
        </ul>
    </div>

    <!-- CONCEPT 4: SUM -->
    <div class="concept-card">
        <h2>4. Cumulative Saturation & Memory (Rolling Sum)</h2>
        <p>The <strong>Rolling Sum</strong> aggregates the discrete volume or load over a fixed trailing interval of <code>k</code> periods.</p>

        <div class="grid-2">
            <div class="sub-box why">
                <h4>What It Signifies</h4>
                <p>Real-world systems have memory and tipping points. Moderate individual events are harmless on their own, but become catastrophic when clustered consecutively.</p>
            </div>
            <div class="sub-box ml">
                <h4>Why It Is Crucial for ML</h4>
                <p>Single-day classifiers only see today's value (e.g., 30 units of rain or a $300 card charge) and predict low risk. The rolling sum supplies the accumulated burden, enabling the model to catch capacity exhaustion and threshold breaches.</p>
            </div>
        </div>

        <div class="chart-box">
            <img src="{img4_b64}" alt="Rolling Sum Plot">
        </div>

        <div class="takeaway-box">
            <strong>Key Insight from the Visual:</strong> The discrete light-blue bars during Days 30–36 are modest (~25–35 units). However, because they arrive consecutively, the <strong>7-Day Rolling Sum (dark blue curve)</strong> launches past 200, cleanly breaching the 120 alert threshold. Exactly 7 days after the burst ends, the sum drains back down to normal.
        </div>

        <h3>Production ML Feature Formulations:</h3>
        <ul>
            <li><code>saturation_ratio = rolling_sum_7 / system_capacity_limit</code> (Direct buffer exhaustion ratio for queues, memory heaps, or reservoir dams).</li>
            <li><code>velocity_risk = rolling_sum_card_spend_1h</code> (Detects automated script attacks and high-velocity payment fraud).</li>
        </ul>
    </div>

    <!-- SUMMARY TABLE -->
    <section>
        <h2>Master Architectural Cheat Sheet</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature Family</th>
                    <th>Pandas Implementation (Lagged)</th>
                    <th>Signal Supplied to ML Model</th>
                    <th>Real-World Domain Applications</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Rolling Mean</strong></td>
                    <td><code>x.shift(1).rolling(k).mean()</code></td>
                    <td>Denoised macro trendline</td>
                    <td>Financial Moving Averages (50 DMA), Baseline Store Sales</td>
                </tr>
                <tr>
                    <td><strong>Rolling StdDev</strong></td>
                    <td><code>x.shift(1).rolling(k).std()</code></td>
                    <td>System stability, volatility regime</td>
                    <td>Bollinger Bands, Risk modeling, SRE latency jitter</td>
                </tr>
                <tr>
                    <td><strong>Rolling Min / Max</strong></td>
                    <td><code>x.shift(1).rolling(k).max()</code></td>
                    <td>Dynamic operating floor & ceiling</td>
                    <td>Donchian channel breakouts, Server CPU peak throttling</td>
                </tr>
                <tr>
                    <td><strong>Rolling Sum</strong></td>
                    <td><code>x.shift(1).rolling(k).sum()</code></td>
                    <td>Cumulative load & saturation</td>
                    <td>Soil moisture flood alerts, Payment velocity fraud detection</td>
                </tr>
            </tbody>
        </table>
    </section>

    <!-- CODE SNIPPET -->
    <section>
        <h2>Production Python Implementation (Pandas)</h2>
        <pre><code>import pandas as pd
import numpy as np

def generate_rolling_feature_pipeline(df: pd.DataFrame, value_col: str, group_col: str = None) -> pd.DataFrame:
    \"\"\"
    Generates a production-ready, leak-free rolling feature vector for tabular time-series models.
    Supports both single-series and grouped entities (e.g., per-stock or per-user IDs).
    \"\"\"
    df = df.copy()
    
    # 1. Isolate the series (grouped or single)
    target = df.groupby(group_col)[value_col] if group_col else df[value_col]
    
    # 2. CRITICAL: Strictly shift by 1 step to prevent target/lookahead leakage
    lagged = target.shift(1)
    
    # 3. Rolling Mean & Relative Stationarity Ratios
    df['roll_mean_7'] = lagged.rolling(7).mean()
    df['roll_mean_30'] = lagged.rolling(30).mean()
    df['trend_ratio_7'] = df[value_col] / (df['roll_mean_7'] + 1e-6)
    df['momentum_spread'] = (df['roll_mean_7'] - df['roll_mean_30']) / (df['roll_mean_30'] + 1e-6)
    
    # 4. Rolling Standard Deviation & Volatility Regime
    df['roll_std_14'] = lagged.rolling(14).std()
    df['rolling_z_score'] = (df[value_col] - df['roll_mean_7']) / (df['roll_std_14'] + 1e-6)
    
    # 5. Dynamic Operating Envelope (Min/Max & Breakouts)
    df['roll_min_14'] = lagged.rolling(14).min()
    df['roll_max_14'] = lagged.rolling(14).max()
    df['is_breakout'] = (df[value_col] > df['roll_max_14']).astype(int)
    df['envelope_pos'] = (df[value_col] - df['roll_min_14']) / (df['roll_max_14'] - df['roll_min_14'] + 1e-6)
    
    # 6. Cumulative Saturation & Memory
    df['roll_sum_7'] = lagged.rolling(7).sum()
    
    return df
</code></pre>
    </section>

    <footer>
        <p>Mastering Rolling Features in Machine Learning &bull; Standalone Tutorial Guide &bull; Ready for offline use, Git repositories, or GitHub Pages deployment.</p>
    </footer>
</div>

</body>
</html>
"""

output_filename = "rolling_features_tutorial_guide.html"
with open(output_filename, "w", encoding="utf-8") as f:
  f.write(html_content)

print(
    f"Successfully written {output_filename} ({os.path.getsize(output_filename)}"
    " bytes)"
)