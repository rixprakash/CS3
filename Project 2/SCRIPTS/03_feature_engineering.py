"""
03_feature_engineering.py

This script creates features for SPY price prediction based on historical
SPY and VIX data. It takes the cleaned data as input and generates derived
features that will be used for model training.

Features created include:
- Lagged values (previous n days) for both SPY and VIX
- Moving averages and standard deviations over various time windows
- Relative strength indicators
- Rate of change metrics
- Volatility measures
- Day of week, month indicators (calendar effects)

Author: JARHeads Team
Date: March 2025
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Create necessary directories if they don't exist
os.makedirs('DATA/processed', exist_ok=True)
os.makedirs('OUTPUT/figures', exist_ok=True)

# Load the cleaned data
print("Loading cleaned data...")
try:
    df = pd.read_csv('DATA/processed/clean_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Check for NaN values in the original data
print("\nChecking for NaN values in cleaned data:")
print(df.isnull().sum())

# Set date as index for time series operations
df = df.sort_values('Date').set_index('Date')

print("\nCreating features...")

# 1. Create lagged features
print("Creating lagged features...")
# Lag SPY prices
for lag in [1, 2, 3, 5, 10, 21]:  # 1, 2, 3 days, 1 week, 2 weeks, 1 month
    df[f'SPY_lag_{lag}'] = df['Price_SPY'].shift(lag)
    df[f'VIX_lag_{lag}'] = df['Price_VIX'].shift(lag)

# 2. Create moving averages and standard deviations
print("Creating moving averages and volatility measures...")
for window in [5, 10, 21, 63]:  # 1 week, 2 weeks, 1 month, 3 months
    # Moving averages
    df[f'SPY_MA_{window}'] = df['Price_SPY'].rolling(window=window).mean()
    df[f'VIX_MA_{window}'] = df['Price_VIX'].rolling(window=window).mean()
    
    # Standard deviations (volatility)
    df[f'SPY_STD_{window}'] = df['Price_SPY'].rolling(window=window).std()
    df[f'VIX_STD_{window}'] = df['Price_VIX'].rolling(window=window).std()

# Calculate MACD after all moving averages have been created
df['SPY_MACD'] = df['SPY_MA_5'] - df['SPY_MA_21']
df['VIX_MACD'] = df['VIX_MA_5'] - df['VIX_MA_21']

# 3. Price momentum (rate of change)
print("Creating momentum indicators...")
for period in [1, 5, 10, 21]:
    df[f'SPY_ROC_{period}'] = df['Price_SPY'].pct_change(periods=period)
    df[f'VIX_ROC_{period}'] = df['Price_VIX'].pct_change(periods=period)

# 4. Relative indicators
print("Creating relative indicators...")
# Relative strength
df['SPY_RS_1m'] = df['Price_SPY'] / df['SPY_MA_21']
df['VIX_RS_1m'] = df['Price_VIX'] / df['VIX_MA_21']

# VIX/SPY ratio
df['VIX_SPY_ratio'] = df['Price_VIX'] / df['Price_SPY']

# 5. Calendar effects
print("Adding calendar features...")
# Reset index to access Date
df = df.reset_index()
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

# One-hot encode day of week and month for better modeling
for day in range(5):  # 0-4 (weekdays only in trading data)
    df[f'DOW_{day}'] = (df['DayOfWeek'] == day).astype(int)

for month in range(1, 13):
    df[f'Month_{month}'] = (df['Month'] == month).astype(int)

# 6. Target variable creation
print("Creating target variables...")
# Next day return (primary target)
df['target_next_day_return'] = df['Price_SPY'].shift(-1) / df['Price_SPY'] - 1

# 5-day forward return
df['target_5d_return'] = df['Price_SPY'].shift(-5) / df['Price_SPY'] - 1

# Binary direction indicators
df['target_next_day_up'] = (df['target_next_day_return'] > 0).astype(int)
df['target_5d_up'] = (df['target_5d_return'] > 0).astype(int)

# 7. Create interaction terms
print("Creating interaction features...")
# VIX change × SPY change
df['VIX_SPY_interaction'] = df['Change %_VIX'] * df['Change %_SPY']

# 8. Check NaN values BEFORE cleaning
print("Checking for NaN values before cleaning:")
nan_counts = df.isnull().sum()
print(nan_counts[nan_counts > 0])  # Only show columns with NaNs

# 9. Handle NaN values more carefully
print("Handling NaN values...")
# Keep track of the original number of rows
original_rows = len(df)

# For lagged and rolling window features, NaNs at the beginning are expected
# We'll trim the beginning of the dataset rather than dropping all rows with NaNs
# Find the maximum window size used
max_window = 63  # Maximum rolling window size used

# Trim the dataset to remove the beginning NaN values from lag and rolling windows
df_trimmed = df.iloc[max_window:].copy()
print(f"Trimmed {max_window} rows from the beginning to remove NaNs from lag/rolling features")

# For any remaining NaNs in the middle, fill with appropriate method
# Forward fill for any remaining NaNs
df_trimmed = df_trimmed.fillna(method='ffill')

# Check if there are still any NaNs
remaining_nans = df_trimmed.isnull().sum()
if remaining_nans.sum() > 0:
    print("Remaining NaN values after forward fill:")
    print(remaining_nans[remaining_nans > 0])
    
    # For any stubborn NaNs, use column means for numerical columns
    for col in df_trimmed.columns:
        if df_trimmed[col].isnull().sum() > 0 and df_trimmed[col].dtype in ['float64', 'int64']:
            col_mean = df_trimmed[col].mean()
            df_trimmed[col] = df_trimmed[col].fillna(col_mean)
            print(f"Filled remaining NaNs in {col} with mean: {col_mean}")

# Remove NaNs from target variables (which are at the end of the dataset)
df_final = df_trimmed.dropna(subset=['target_next_day_return', 'target_5d_return'])
print(f"Rows: original: {original_rows}, after trimming: {len(df_trimmed)}, after target NaN removal: {len(df_final)}")

# 10. Feature correlation analysis
print("\nAnalyzing feature correlations...")
# Correlation with target
target_corr = df_final.corr()['target_next_day_return'].sort_values(ascending=False)
print("\nTop features correlated with next day return:")
print(target_corr.head(10))
print("\nBottom features correlated with next day return:")
print(target_corr.tail(10))

# Create correlation heatmap for top features
plt.figure(figsize=(12, 10))
top_features = list(target_corr.head(15).index) + list(target_corr.tail(5).index)
correlation_matrix = df_final[top_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation of Top Features with Target Variable')
plt.tight_layout()
plt.savefig('OUTPUT/figures/feature_correlation_heatmap.png')

# 11. Save the feature-engineered dataset
print("Saving feature-engineered dataset...")
df_final.to_csv('DATA/processed/feature_data.csv', index=False)
print(f"Feature engineering complete! Created {df_final.shape[1] - len(df.columns)} new features.")
print(f"Feature data saved to DATA/processed/feature_data.csv with {df_final.shape[0]} rows and {df_final.shape[1]} columns.")

# 12. Create a feature summary
feature_summary = pd.DataFrame({
    'Feature': df_final.columns,
    'Type': df_final.dtypes,
    'Missing': df_final.isnull().sum(),
    'Unique': [df_final[col].nunique() for col in df_final.columns],
    'Target_Correlation': [df_final[col].corr(df_final['target_next_day_return']) 
                          if df_final[col].dtype in ['float64', 'int64'] else None 
                          for col in df_final.columns]
})

print("\nFeature summary (first 10 rows):")
print(feature_summary.head(10))
feature_summary.to_csv('DATA/processed/feature_summary.csv', index=False)
print("Feature summary saved to DATA/processed/feature_summary.csv")
