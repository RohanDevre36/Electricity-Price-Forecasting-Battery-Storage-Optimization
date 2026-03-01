#!/usr/bin/env python3
"""
=============================================================================
Electricity Price Forecasting & Battery Storage Arbitrage Modeling
ERCOT Market Analysis
=============================================================================

Project: ERCOT Day-Ahead LMP Forecasting + BESS Revenue Optimization
Author: Energy Analytics Portfolio Project
Date: March 2026

This project demonstrates:
1. Realistic ERCOT electricity price data generation (based on public patterns)
2. Time-series cleaning & feature engineering
3. Multiple forecasting models (Linear, Ridge, Random Forest, Gradient Boosting)
4. Battery Energy Storage System (BESS) arbitrage simulation
5. Revenue optimization under price volatility
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = '/home/claude/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("ERCOT ELECTRICITY PRICE FORECASTING & BESS ARBITRAGE OPTIMIZATION")
print("=" * 80)

# =============================================================================
# SECTION 1: REALISTIC ERCOT PRICE DATA GENERATION
# =============================================================================
print("\n[1/5] Generating realistic ERCOT day-ahead price data...")

np.random.seed(42)

# Generate 2 years of hourly data (2024-2025) — mimics real ERCOT patterns
date_range = pd.date_range(start='2024-01-01', end='2025-12-31 23:00', freq='h')
n_hours = len(date_range)

# --- Base price components (reflecting real ERCOT characteristics) ---

# 1. Seasonal component (ERCOT: summer peaks, mild winters)
day_of_year = date_range.dayofyear.values
seasonal = 15 * np.sin(2 * np.pi * (day_of_year - 30) / 365)  # Peak in July/Aug

# 2. Hourly profile (ERCOT: afternoon peaks, overnight lows)
hour = date_range.hour.values
hourly_profile = np.array([
    22, 20, 19, 18, 18, 20, 25, 32, 38, 40, 42, 44,  # 0-11
    46, 50, 55, 58, 56, 52, 48, 42, 36, 32, 28, 24    # 12-23
], dtype=float)
hourly = np.array([hourly_profile[h] for h in hour])

# 3. Weekend effect (lower prices on weekends)
is_weekend = date_range.dayofweek.values >= 5
weekend_adj = np.where(is_weekend, -8, 0)

# 4. Temperature-driven demand spikes (summer heat waves in Texas)
month = date_range.month.values
summer_mask = (month >= 6) & (month <= 9)
# Simulate heat wave events (random multi-day spikes)
heat_wave = np.zeros(n_hours)
for _ in range(12):  # ~12 heat wave events across 2 summers
    start = np.random.randint(0, n_hours - 72)
    if summer_mask[start]:
        duration = np.random.randint(24, 96)
        intensity = np.random.uniform(20, 80)
        heat_wave[start:start+duration] += intensity

# 5. Congestion and scarcity events (ERCOT-specific price spikes)
spike_mask = np.random.random(n_hours) < 0.005  # ~0.5% of hours
price_spikes = np.where(spike_mask, np.random.exponential(200, n_hours), 0)
# More spikes in summer
summer_spike_boost = np.where(summer_mask, np.random.exponential(50, n_hours) * (np.random.random(n_hours) < 0.01), 0)

# 6. Wind generation proxy (ERCOT has massive wind, depresses prices)
wind_generation = 8000 + 4000 * np.sin(2 * np.pi * day_of_year / 365 + np.pi) + \
                  np.random.normal(0, 2000, n_hours)
wind_generation = np.clip(wind_generation, 0, 20000)
wind_price_impact = -0.003 * wind_generation  # Higher wind → lower prices

# 7. Solar generation proxy (growing solar in ERCOT)
solar_factor = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
solar_gen = solar_factor * (3000 + 1500 * np.sin(2 * np.pi * (day_of_year - 80) / 365))
solar_gen = np.clip(solar_gen + np.random.normal(0, 300, n_hours), 0, 8000)
solar_price_impact = -0.002 * solar_gen

# 8. Natural gas price proxy (Henry Hub correlation)
gas_base = 2.5 + 0.8 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
gas_noise = np.cumsum(np.random.normal(0, 0.02, n_hours))
gas_noise = gas_noise - np.linspace(gas_noise[0], gas_noise[-1], n_hours)
gas_price = gas_base + gas_noise
gas_price = np.clip(gas_price, 1.5, 6.0)
gas_impact = 5 * gas_price  # Gas-on-margin pricing

# 9. Load proxy (MW)
base_load = 40000 + 15000 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
hourly_factor = np.array([hourly_profile[h] for h in hour]) / 45.0
load_hourly = hourly_factor * base_load
load = load_hourly + np.random.normal(0, 2000, n_hours)
load = np.clip(load, 20000, 80000)

# 10. Random noise
noise = np.random.normal(0, 5, n_hours)

# Combine all components
lmp = (hourly + seasonal + weekend_adj + heat_wave + price_spikes +
       summer_spike_boost + wind_price_impact + solar_price_impact +
       gas_impact + noise)

# Ensure realistic bounds (ERCOT can go negative but rarely below -$20)
lmp = np.clip(lmp, -20, 5000)

# Occasionally allow very high ERCOT spikes (system-wide offer cap is $5000)
extreme_idx = np.random.random(n_hours) < 0.001
lmp[extreme_idx & summer_mask] = np.random.uniform(500, 3000, n_hours)[extreme_idx & summer_mask]

# Build DataFrame
df = pd.DataFrame({
    'timestamp': date_range,
    'lmp_da': lmp,  # Day-Ahead LMP ($/MWh)
    'load_mw': load,
    'wind_gen_mw': wind_generation,
    'solar_gen_mw': solar_gen,
    'gas_price_mmbtu': gas_price,
    'is_weekend': is_weekend.astype(int),
})
df.set_index('timestamp', inplace=True)

print(f"   Generated {len(df):,} hourly records ({df.index.min().date()} to {df.index.max().date()})")
print(f"   LMP range: ${df['lmp_da'].min():.2f} to ${df['lmp_da'].max():.2f}/MWh")
print(f"   Mean LMP: ${df['lmp_da'].mean():.2f}/MWh")
print(f"   Median LMP: ${df['lmp_da'].median():.2f}/MWh")

# =============================================================================
# SECTION 2: DATA CLEANING & EXPLORATORY ANALYSIS
# =============================================================================
print("\n[2/5] Time-series cleaning & exploratory analysis...")

# Inject some realistic data quality issues, then clean them
df_raw = df.copy()
# Add some missing values
missing_idx = np.random.choice(df.index, size=50, replace=False)
df_raw.loc[missing_idx, 'lmp_da'] = np.nan

# Clean: forward-fill then back-fill for small gaps
df_clean = df_raw.copy()
df_clean['lmp_da'] = df_clean['lmp_da'].ffill(limit=3).bfill(limit=3)
remaining_na = df_clean['lmp_da'].isna().sum()
if remaining_na > 0:
    df_clean['lmp_da'] = df_clean['lmp_da'].interpolate(method='linear')

print(f"   Cleaned {50 - remaining_na} missing values via forward/back-fill")

# Cap extreme outliers at 99.5th percentile for modeling (keep originals for reference)
cap_value = df_clean['lmp_da'].quantile(0.995)
df_clean['lmp_da_capped'] = df_clean['lmp_da'].clip(upper=cap_value)
n_capped = (df_clean['lmp_da'] > cap_value).sum()
print(f"   Capped {n_capped} extreme prices above ${cap_value:.2f}/MWh")

# --- Summary Statistics ---
stats = df_clean['lmp_da'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
print(f"\n   Summary Statistics (LMP $/MWh):")
for idx, val in stats.items():
    print(f"     {idx:>8s}: ${val:>10.2f}")

# --- Generate EDA Plots ---

# Plot 1: Price Duration Curve
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sorted_prices = np.sort(df_clean['lmp_da'].values)[::-1]
pct = np.linspace(0, 100, len(sorted_prices))
axes[0, 0].plot(pct, sorted_prices, color='#e74c3c', linewidth=1.5)
axes[0, 0].set_xlabel('Percentage of Hours (%)', fontsize=11)
axes[0, 0].set_ylabel('LMP ($/MWh)', fontsize=11)
axes[0, 0].set_title('Price Duration Curve — ERCOT Day-Ahead', fontsize=13, fontweight='bold')
axes[0, 0].axhline(y=df_clean['lmp_da'].mean(), color='gray', linestyle='--', alpha=0.7, label=f'Mean: ${df_clean["lmp_da"].mean():.1f}')
axes[0, 0].legend()
axes[0, 0].set_ylim(-30, 300)

# Plot 2: Hourly Price Profile by Season
df_clean['hour'] = df_clean.index.hour
df_clean['month'] = df_clean.index.month
df_clean['season'] = df_clean['month'].map(lambda m: 'Winter' if m in [12, 1, 2] else
                                            'Spring' if m in [3, 4, 5] else
                                            'Summer' if m in [6, 7, 8] else 'Fall')
season_hourly = df_clean.groupby(['season', 'hour'])['lmp_da_capped'].mean().unstack(level=0)
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    if season in season_hourly.columns:
        axes[0, 1].plot(season_hourly.index, season_hourly[season], linewidth=2.5, label=season)
axes[0, 1].set_xlabel('Hour of Day', fontsize=11)
axes[0, 1].set_ylabel('Avg LMP ($/MWh)', fontsize=11)
axes[0, 1].set_title('Hourly Price Profile by Season', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].set_xticks(range(0, 24, 3))

# Plot 3: Monthly Distribution Boxplot
monthly_data = [df_clean.loc[df_clean['month'] == m, 'lmp_da_capped'].values for m in range(1, 13)]
bp = axes[1, 0].boxplot(monthly_data, labels=['Jan','Feb','Mar','Apr','May','Jun',
                                                'Jul','Aug','Sep','Oct','Nov','Dec'],
                         patch_artist=True, showfliers=False)
colors_months = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 12))
for patch, color in zip(bp['boxes'], colors_months):
    patch.set_facecolor(color)
axes[1, 0].set_xlabel('Month', fontsize=11)
axes[1, 0].set_ylabel('LMP ($/MWh)', fontsize=11)
axes[1, 0].set_title('Monthly Price Distribution', fontsize=13, fontweight='bold')

# Plot 4: Price vs Load Scatter
scatter = axes[1, 1].scatter(df_clean['load_mw'] / 1000, df_clean['lmp_da_capped'],
                              c=df_clean['hour'], cmap='viridis', alpha=0.1, s=2)
axes[1, 1].set_xlabel('System Load (GW)', fontsize=11)
axes[1, 1].set_ylabel('LMP ($/MWh)', fontsize=11)
axes[1, 1].set_title('Price vs. Load (colored by hour)', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=axes[1, 1], label='Hour of Day')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 01_eda_analysis.png")

# =============================================================================
# SECTION 3: FEATURE ENGINEERING
# =============================================================================
print("\n[3/5] Feature engineering for forecasting models...")

df_feat = df_clean.copy()

# Temporal features
df_feat['hour'] = df_feat.index.hour
df_feat['day_of_week'] = df_feat.index.dayofweek
df_feat['month'] = df_feat.index.month
df_feat['day_of_year'] = df_feat.index.dayofyear
df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
df_feat['is_peak'] = ((df_feat['hour'] >= 7) & (df_feat['hour'] <= 22)).astype(int)
df_feat['is_super_peak'] = ((df_feat['hour'] >= 14) & (df_feat['hour'] <= 19)).astype(int)

# Cyclical encoding
df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)

# Lag features (using capped prices)
for lag in [1, 2, 3, 24, 48, 168]:  # 1h, 2h, 3h, 1d, 2d, 1w
    df_feat[f'lmp_lag_{lag}h'] = df_feat['lmp_da_capped'].shift(lag)

# Rolling statistics
for window in [24, 48, 168]:
    df_feat[f'lmp_roll_mean_{window}h'] = df_feat['lmp_da_capped'].rolling(window).mean()
    df_feat[f'lmp_roll_std_{window}h'] = df_feat['lmp_da_capped'].rolling(window).std()
    df_feat[f'lmp_roll_max_{window}h'] = df_feat['lmp_da_capped'].rolling(window).max()

# Same-hour-yesterday price
df_feat['lmp_same_hour_yesterday'] = df_feat['lmp_da_capped'].shift(24)
df_feat['lmp_same_hour_last_week'] = df_feat['lmp_da_capped'].shift(168)

# Load and renewables features
df_feat['net_load'] = df_feat['load_mw'] - df_feat['wind_gen_mw'] - df_feat['solar_gen_mw']
df_feat['renewable_penetration'] = (df_feat['wind_gen_mw'] + df_feat['solar_gen_mw']) / df_feat['load_mw']
df_feat['load_wind_ratio'] = df_feat['load_mw'] / (df_feat['wind_gen_mw'] + 1)

# Drop NaN rows from lagging
df_feat = df_feat.dropna()
print(f"   Created {len([c for c in df_feat.columns if c not in df_clean.columns])} features")
print(f"   Training data: {len(df_feat):,} hourly records after dropping NaN")

# =============================================================================
# SECTION 4: FORECASTING MODELS
# =============================================================================
print("\n[4/5] Building and evaluating forecasting models...")

# Define features and target
feature_cols = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak', 'is_super_peak',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'load_mw', 'wind_gen_mw', 'solar_gen_mw', 'gas_price_mmbtu',
    'net_load', 'renewable_penetration', 'load_wind_ratio',
    'lmp_lag_1h', 'lmp_lag_2h', 'lmp_lag_3h', 'lmp_lag_24h', 'lmp_lag_48h', 'lmp_lag_168h',
    'lmp_roll_mean_24h', 'lmp_roll_std_24h', 'lmp_roll_max_24h',
    'lmp_roll_mean_48h', 'lmp_roll_std_48h',
    'lmp_roll_mean_168h', 'lmp_roll_std_168h',
    'lmp_same_hour_yesterday', 'lmp_same_hour_last_week',
]

target_col = 'lmp_da_capped'

# Time-based train/test split (last 3 months = test)
split_date = '2025-10-01'
train = df_feat[df_feat.index < split_date]
test = df_feat[df_feat.index >= split_date]

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

print(f"   Train: {len(train):,} samples ({train.index.min().date()} to {train.index.max().date()})")
print(f"   Test:  {len(test):,} samples ({test.index.min().date()} to {test.index.max().date()})")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=10,
                                            random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                                     subsample=0.8, random_state=42),
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    if 'Linear' in name or 'Ridge' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}
    predictions[name] = y_pred
    print(f"     MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f} | MAPE: {mape:.1f}%")

# Results DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('MAE')
print(f"\n   Model Comparison (sorted by MAE):")
print(results_df.to_string())

# Best model
best_model_name = results_df.index[0]
best_pred = predictions[best_model_name]
print(f"\n   ✓ Best model: {best_model_name}")

# Feature importance (from best tree model)
if 'Gradient Boosting' in models:
    gb_model = models['Gradient Boosting']
    feat_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)

# --- Forecasting Plots ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Actual vs Predicted (2-week window)
plot_start = '2025-11-01'
plot_end = '2025-11-14'
mask = (test.index >= plot_start) & (test.index <= plot_end)
axes[0, 0].plot(test.index[mask], y_test[mask], label='Actual', color='#2c3e50', linewidth=1.5)
axes[0, 0].plot(test.index[mask], predictions[best_model_name][mask],
                label=f'{best_model_name}', color='#e74c3c', linewidth=1.5, alpha=0.8)
axes[0, 0].set_title(f'Actual vs Predicted LMP — {best_model_name}', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('LMP ($/MWh)', fontsize=11)
axes[0, 0].legend()
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=30)

# Plot 2: Model Comparison Bar Chart
model_names = results_df.index.tolist()
maes = results_df['MAE'].values
colors_bar = ['#27ae60' if n == best_model_name else '#3498db' for n in model_names]
axes[0, 1].barh(model_names, maes, color=colors_bar)
axes[0, 1].set_xlabel('MAE ($/MWh)', fontsize=11)
axes[0, 1].set_title('Model Comparison — MAE', fontsize=13, fontweight='bold')
for i, v in enumerate(maes):
    axes[0, 1].text(v + 0.2, i, f'${v:.2f}', va='center', fontsize=10)

# Plot 3: Feature Importance (top 15)
top_feat = feat_importance.head(15)
axes[1, 0].barh(top_feat['feature'], top_feat['importance'], color='#8e44ad')
axes[1, 0].set_xlabel('Importance', fontsize=11)
axes[1, 0].set_title('Top 15 Feature Importances (Gradient Boosting)', fontsize=13, fontweight='bold')
axes[1, 0].invert_yaxis()

# Plot 4: Residual Distribution
residuals = y_test.values - best_pred
axes[1, 1].hist(residuals, bins=80, color='#16a085', edgecolor='white', alpha=0.8)
axes[1, 1].axvline(x=0, color='red', linestyle='--')
axes[1, 1].set_xlabel('Residual ($/MWh)', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title(f'Residual Distribution — {best_model_name}', fontsize=13, fontweight='bold')
axes[1, 1].text(0.95, 0.95, f'Mean: ${np.mean(residuals):.2f}\nStd: ${np.std(residuals):.2f}',
                transform=axes[1, 1].transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_forecasting_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 02_forecasting_results.png")

# =============================================================================
# SECTION 5: BESS ARBITRAGE SIMULATION
# =============================================================================
print("\n[5/5] Running BESS arbitrage simulation...")

# --- BESS Configuration ---
bess_config = {
    'capacity_mwh': 100,       # 100 MWh battery
    'power_mw': 25,            # 25 MW charge/discharge rate (4-hour duration)
    'efficiency_rt': 0.87,     # 87% round-trip efficiency
    'efficiency_charge': np.sqrt(0.87),
    'efficiency_discharge': np.sqrt(0.87),
    'min_soc': 0.10,           # 10% minimum state of charge
    'max_soc': 0.90,           # 90% maximum state of charge
    'degradation_cost': 2.0,   # $/MWh degradation cost per cycle
    'cycles_per_year_limit': 365,  # 1 cycle/day target
}

print(f"   BESS Configuration:")
print(f"     Capacity: {bess_config['capacity_mwh']} MWh")
print(f"     Power: {bess_config['power_mw']} MW ({bess_config['capacity_mwh'] // bess_config['power_mw']}h duration)")
print(f"     Round-trip efficiency: {bess_config['efficiency_rt']*100:.0f}%")

# --- Strategy 1: Perfect Foresight Arbitrage (Upper Bound) ---
def simulate_bess_perfect(prices, config):
    """Simulate BESS with perfect price foresight — buy low, sell high within each day."""
    n = len(prices)
    soc = np.zeros(n + 1)  # State of charge (MWh)
    soc[0] = config['capacity_mwh'] * 0.5  # Start at 50%
    revenue = np.zeros(n)
    action = np.full(n, '', dtype='U10')
    charge_mw = np.zeros(n)
    discharge_mw = np.zeros(n)

    capacity = config['capacity_mwh']
    power = config['power_mw']
    eff_c = config['efficiency_charge']
    eff_d = config['efficiency_discharge']
    min_e = config['min_soc'] * capacity
    max_e = config['max_soc'] * capacity
    deg_cost = config['degradation_cost']

    # Process each day: rank hours by price, charge in cheapest, discharge in most expensive
    for day_start in range(0, n, 24):
        day_end = min(day_start + 24, n)
        day_prices = prices[day_start:day_end]
        if len(day_prices) < 24:
            continue

        # Rank hours
        sorted_hours = np.argsort(day_prices)
        charge_hours = set(sorted_hours[:6])     # Cheapest 6 hours → charge
        discharge_hours = set(sorted_hours[-6:])  # Most expensive 6 hours → discharge

        for h in range(day_start, day_end):
            h_in_day = h - day_start
            price = prices[h]

            if h_in_day in charge_hours and soc[h] < max_e:
                energy_in = min(power, (max_e - soc[h]) / eff_c)
                soc[h + 1] = soc[h] + energy_in * eff_c
                revenue[h] = -energy_in * price  # Pay to charge
                charge_mw[h] = energy_in
                action[h] = 'charge'

            elif h_in_day in discharge_hours and soc[h] > min_e:
                energy_out = min(power, (soc[h] - min_e))
                delivered = energy_out * eff_d
                soc[h + 1] = soc[h] - energy_out
                revenue[h] = delivered * price - energy_out * deg_cost  # Earn from discharge
                discharge_mw[h] = delivered
                action[h] = 'discharge'
            else:
                soc[h + 1] = soc[h]
                action[h] = 'idle'

    return pd.DataFrame({
        'price': prices,
        'soc_mwh': soc[:-1],
        'revenue': revenue,
        'action': action,
        'charge_mw': charge_mw,
        'discharge_mw': discharge_mw,
    })

# --- Strategy 2: Simple Threshold Arbitrage ---
def simulate_bess_threshold(prices, config, charge_threshold=30, discharge_threshold=60):
    """Charge below threshold, discharge above threshold."""
    n = len(prices)
    soc = np.zeros(n + 1)
    soc[0] = config['capacity_mwh'] * 0.5
    revenue = np.zeros(n)
    action = np.full(n, '', dtype='U10')

    capacity = config['capacity_mwh']
    power = config['power_mw']
    eff_c = config['efficiency_charge']
    eff_d = config['efficiency_discharge']
    min_e = config['min_soc'] * capacity
    max_e = config['max_soc'] * capacity
    deg_cost = config['degradation_cost']

    for h in range(n):
        price = prices[h]
        if price < charge_threshold and soc[h] < max_e:
            energy_in = min(power, (max_e - soc[h]) / eff_c)
            soc[h + 1] = soc[h] + energy_in * eff_c
            revenue[h] = -energy_in * price
            action[h] = 'charge'
        elif price > discharge_threshold and soc[h] > min_e:
            energy_out = min(power, (soc[h] - min_e))
            delivered = energy_out * eff_d
            soc[h + 1] = soc[h] - energy_out
            revenue[h] = delivered * price - energy_out * deg_cost
            action[h] = 'discharge'
        else:
            soc[h + 1] = soc[h]
            action[h] = 'idle'

    return pd.DataFrame({
        'price': prices,
        'soc_mwh': soc[:-1],
        'revenue': revenue,
        'action': action,
    })

# --- Strategy 3: Forecast-Based Arbitrage ---
def simulate_bess_forecast(actual_prices, predicted_prices, config):
    """Use ML forecast to decide when to charge/discharge."""
    n = len(actual_prices)
    soc = np.zeros(n + 1)
    soc[0] = config['capacity_mwh'] * 0.5
    revenue = np.zeros(n)
    action = np.full(n, '', dtype='U10')

    capacity = config['capacity_mwh']
    power = config['power_mw']
    eff_c = config['efficiency_charge']
    eff_d = config['efficiency_discharge']
    min_e = config['min_soc'] * capacity
    max_e = config['max_soc'] * capacity
    deg_cost = config['degradation_cost']

    for day_start in range(0, n, 24):
        day_end = min(day_start + 24, n)
        day_forecast = predicted_prices[day_start:day_end]
        if len(day_forecast) < 24:
            continue

        sorted_hours = np.argsort(day_forecast)
        charge_hours = set(sorted_hours[:6])
        discharge_hours = set(sorted_hours[-6:])

        for h in range(day_start, day_end):
            h_in_day = h - day_start
            price = actual_prices[h]  # Settle at actual price

            if h_in_day in charge_hours and soc[h] < max_e:
                energy_in = min(power, (max_e - soc[h]) / eff_c)
                soc[h + 1] = soc[h] + energy_in * eff_c
                revenue[h] = -energy_in * price
                action[h] = 'charge'
            elif h_in_day in discharge_hours and soc[h] > min_e:
                energy_out = min(power, (soc[h] - min_e))
                delivered = energy_out * eff_d
                soc[h + 1] = soc[h] - energy_out
                revenue[h] = delivered * price - energy_out * deg_cost
                action[h] = 'discharge'
            else:
                soc[h + 1] = soc[h]
                action[h] = 'idle'

    return pd.DataFrame({
        'actual_price': actual_prices,
        'forecast_price': predicted_prices,
        'soc_mwh': soc[:-1],
        'revenue': revenue,
        'action': action,
    })

# Run simulations on test period
test_prices = y_test.values
test_forecast = best_pred

result_perfect = simulate_bess_perfect(test_prices, bess_config)
result_threshold = simulate_bess_threshold(test_prices, bess_config)
result_forecast = simulate_bess_forecast(test_prices, test_forecast, bess_config)

# --- Revenue Summary ---
strategies = {
    'Perfect Foresight': result_perfect,
    'Threshold (30/60)': result_threshold,
    f'ML Forecast ({best_model_name})': result_forecast,
}

print(f"\n   {'Strategy':<35s} {'Revenue':>12s} {'$/kW-yr':>10s} {'Cycles':>8s} {'Capture %':>10s}")
print("   " + "-" * 78)

revenue_summary = {}
for name, res in strategies.items():
    total_rev = res['revenue'].sum()
    # Annualize (test is ~3 months)
    test_days = len(test) / 24
    annual_rev = total_rev * (365 / test_days)
    rev_per_kw_yr = annual_rev / (bess_config['power_mw'] * 1000)  # $/kW-year
    total_discharge = res.get('discharge_mw', (res['action'] == 'discharge').sum() * bess_config['power_mw']).sum() if 'discharge_mw' in res.columns else (res['action'] == 'discharge').sum() * bess_config['power_mw']
    cycles = total_discharge / bess_config['capacity_mwh'] if hasattr(total_discharge, '__float__') else 0

    # Capture rate vs perfect
    perfect_rev = result_perfect['revenue'].sum()
    capture = (total_rev / perfect_rev * 100) if perfect_rev > 0 else 0

    revenue_summary[name] = {
        'total_revenue': total_rev,
        'annualized_revenue': annual_rev,
        'rev_per_kw_yr': rev_per_kw_yr,
        'cycles': cycles,
        'capture_rate': capture,
    }
    print(f"   {name:<35s} ${total_rev:>11,.0f} ${rev_per_kw_yr:>9,.0f} {cycles:>7.0f} {capture:>9.1f}%")

# --- BESS Plots ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: 1-week BESS operation detail
week_start = test.index[0]
week_end = week_start + timedelta(days=7)
week_mask = (test.index >= week_start) & (test.index < week_end)

ax1 = axes[0, 0]
ax1_twin = ax1.twinx()
ax1.plot(test.index[week_mask], test_prices[:week_mask.sum()], color='#2c3e50', linewidth=1.5, label='LMP')
ax1_twin.plot(test.index[week_mask], result_forecast['soc_mwh'].values[:week_mask.sum()],
              color='#27ae60', linewidth=2, linestyle='--', label='SOC')
ax1.set_ylabel('LMP ($/MWh)', fontsize=11, color='#2c3e50')
ax1_twin.set_ylabel('SOC (MWh)', fontsize=11, color='#27ae60')
ax1.set_title('BESS Operation Detail — 1 Week (Forecast Strategy)', fontsize=13, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Plot 2: Cumulative Revenue Comparison
ax2 = axes[0, 1]
for name, res in strategies.items():
    cum_rev = np.cumsum(res['revenue'].values)
    ax2.plot(test.index[:len(cum_rev)], cum_rev, linewidth=2, label=name)
ax2.set_ylabel('Cumulative Revenue ($)', fontsize=11)
ax2.set_title('Cumulative BESS Revenue by Strategy', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 3: Daily Revenue Distribution
ax3 = axes[1, 0]
for name, res in strategies.items():
    daily_rev = res['revenue'].values.reshape(-1, 24).sum(axis=1) if len(res) % 24 == 0 else \
                pd.Series(res['revenue'].values, index=test.index[:len(res)]).resample('D').sum().values
    ax3.hist(daily_rev, bins=40, alpha=0.5, label=name, edgecolor='white')
ax3.set_xlabel('Daily Revenue ($)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Daily Revenue Distribution', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)

# Plot 4: Strategy Revenue Comparison Bar
ax4 = axes[1, 1]
strat_names = list(revenue_summary.keys())
strat_annual = [revenue_summary[s]['annualized_revenue'] for s in strat_names]
colors_strat = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax4.bar(range(len(strat_names)), strat_annual, color=colors_strat[:len(strat_names)])
ax4.set_xticks(range(len(strat_names)))
ax4.set_xticklabels([s[:20] + '...' if len(s) > 20 else s for s in strat_names], fontsize=9)
ax4.set_ylabel('Annualized Revenue ($)', fontsize=11)
ax4.set_title('Annualized BESS Revenue by Strategy', fontsize=13, fontweight='bold')
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
for bar, val in zip(bars, strat_annual):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'${val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_bess_arbitrage.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 03_bess_arbitrage.png")

# =============================================================================
# ADDITIONAL ANALYSIS: Sensitivity & Risk
# =============================================================================
print("\n[BONUS] Running sensitivity analysis...")

# Threshold sensitivity
thresholds = [(15, 45), (20, 50), (25, 55), (30, 60), (35, 65), (40, 70), (25, 80)]
threshold_results = []
for ct, dt in thresholds:
    res = simulate_bess_threshold(test_prices, bess_config, charge_threshold=ct, discharge_threshold=dt)
    rev = res['revenue'].sum()
    test_days = len(test) / 24
    annual = rev * (365 / test_days)
    threshold_results.append({'charge_thresh': ct, 'discharge_thresh': dt,
                               'total_rev': rev, 'annual_rev': annual})
    print(f"   Threshold ({ct}/{dt}): Annual Rev = ${annual:,.0f}")

threshold_df = pd.DataFrame(threshold_results)

# Price volatility impact
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sensitivity plot
ax = axes[0]
labels = [f"{r['charge_thresh']}/{r['discharge_thresh']}" for _, r in threshold_df.iterrows()]
ax.bar(labels, threshold_df['annual_rev'], color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(threshold_df))))
ax.set_xlabel('Charge/Discharge Threshold ($/MWh)', fontsize=11)
ax.set_ylabel('Annualized Revenue ($)', fontsize=11)
ax.set_title('Threshold Sensitivity Analysis', fontsize=13, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Monthly revenue profile
monthly_rev = pd.Series(result_forecast['revenue'].values, index=test.index[:len(result_forecast)]).resample('ME').sum()
axes[1].bar(monthly_rev.index.strftime('%b %Y'), monthly_rev.values,
            color=['#e74c3c' if v < 0 else '#27ae60' for v in monthly_rev.values])
axes[1].set_ylabel('Monthly Revenue ($)', fontsize=11)
axes[1].set_title('Monthly BESS Revenue (Forecast Strategy)', fontsize=13, fontweight='bold')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: 04_sensitivity_analysis.png")

# =============================================================================
# EXPORT RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

# Save model comparison
results_df.to_csv(f'{OUTPUT_DIR}/model_comparison.csv')

# Save BESS revenue summary
rev_df = pd.DataFrame(revenue_summary).T
rev_df.to_csv(f'{OUTPUT_DIR}/bess_revenue_summary.csv')

# Save sample data
df_clean.head(1000).to_csv(f'{OUTPUT_DIR}/sample_ercot_data.csv')

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    FORECASTING MODEL RESULTS                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Best Model: {best_model_name:<54s} ║
║  MAE:  ${results[best_model_name]['MAE']:<56.2f}  ║
║  RMSE: ${results[best_model_name]['RMSE']:<56.2f}  ║
║  R²:   {results[best_model_name]['R²']:<57.4f}  ║
║  MAPE: {results[best_model_name]['MAPE']:<56.1f}% ║
╠══════════════════════════════════════════════════════════════════════╣
║                    BESS ARBITRAGE RESULTS (100MWh/25MW)              ║
╠══════════════════════════════════════════════════════════════════════╣""")

for name, summary in revenue_summary.items():
    print(f"║  {name:<34s} Annual Rev: ${summary['annualized_revenue']:>12,.0f}  ║")

print(f"""╠══════════════════════════════════════════════════════════════════════╣
║  ML Forecast captures {revenue_summary[f'ML Forecast ({best_model_name})']['capture_rate']:.1f}% of perfect foresight revenue            ║
╚══════════════════════════════════════════════════════════════════════╝

Output files saved to: {OUTPUT_DIR}/
  - 01_eda_analysis.png
  - 02_forecasting_results.png
  - 03_bess_arbitrage.png
  - 04_sensitivity_analysis.png
  - model_comparison.csv
  - bess_revenue_summary.csv
  - sample_ercot_data.csv
""")

print("Project complete! ✓")
