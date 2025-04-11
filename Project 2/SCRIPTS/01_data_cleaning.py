"""
01_data_cleaning.py

This script processes the raw merged data file containing SPY and VIX historical data,
performs necessary cleaning operations, and outputs a cleaned dataset.

Operations performed:
- Converting date strings to datetime format
- Converting percentage changes to proper numeric format
- Standardizing volume values
- Handling missing values
- Ensuring proper data types
- Identifying and handling outliers

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

# Function to convert volume string to numeric
def convert_volume(vol_str):
    """Convert volume strings like '1.5M' to numeric values."""
    if isinstance(vol_str, str):
        if 'K' in vol_str:
            return float(vol_str.replace('K', '')) * 1000
        elif 'M' in vol_str:
            return float(vol_str.replace('M', '')) * 1000000
        elif 'B' in vol_str:
            return float(vol_str.replace('B', '')) * 1000000000
        else:
            return float(vol_str.replace(',', ''))
    return vol_str

# Function to convert percentage strings to float
def convert_percentage(pct_str):
    """Convert percentage strings like '2.5%' to float values (0.025)."""
    if isinstance(pct_str, str):
        return float(pct_str.replace('%', '')) / 100
    return pct_str

# Load the raw data
print("Loading raw data...")
try:
    df = pd.read_csv('DATA/merged_data.csv')
    print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Display initial data info
print("\nInitial data information:")
print(df.info())
print("\nSample data:")
print(df.head())

# Check for missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Convert date to datetime
print("\nConverting date column to datetime...")
df['Date'] = pd.to_datetime(df['Date'])

# Convert percentage changes to numeric
print("\nConverting percentage changes to numeric values...")
df['Change %_VIX'] = df['Change %_VIX'].apply(convert_percentage)
df['Change %_SPY'] = df['Change %_SPY'].apply(convert_percentage)

# Convert volume columns to numeric
print("\nStandardizing volume values...")
df['Vol._VIX'] = df['Vol._VIX'].apply(convert_volume)
df['Vol._SPY'] = df['Vol._SPY'].apply(convert_volume)

# Handle missing values
print("\nHandling missing values...")
# For time series data, forward fill is often appropriate
df = df.sort_values('Date').set_index('Date')
df = df.fillna(method='ffill')
# Alternative approach: interpolate
# df = df.interpolate(method='time')
df = df.reset_index()

# Check for outliers in price data
print("\nChecking for outliers...")
# Using IQR method for demonstration
def identify_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return series[(series < lower_bound) | (series > upper_bound)]

for col in ['Price_VIX', 'Price_SPY']:
    outliers = identify_outliers(df[col])
    if len(outliers) > 0:
        print(f"Found {len(outliers)} outliers in {col}:")
        print(outliers)
        
        # Create visualization of outliers
        plt.figure(figsize=(10, 6))
        plt.boxplot(df[col])
        plt.title(f'Boxplot of {col} showing outliers')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.savefig(f'OUTPUT/figures/{col}_outliers_boxplot.png')
        
        # For this project, we'll keep outliers as they may represent important market events
        print(f"Note: Outliers in {col} are kept as they may represent significant market events.")

# Save the cleaned data
print("\nSaving cleaned data...")
df.to_csv('DATA/processed/clean_data.csv', index=False)
print("Cleaning complete! Cleaned data saved to DATA/processed/clean_data.csv")

# Display summary statistics of cleaned data
print("\nSummary statistics of cleaned data:")
print(df.describe())

# Create a simple visualization to verify the data
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price_SPY'], label='SPY')
plt.plot(df['Date'], df['Price_VIX'] * 10, label='VIX (scaled x10)') # Scaling for visibility
plt.title('SPY and VIX Prices After Cleaning')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig('OUTPUT/figures/cleaned_data_verification.png')
print("\nGenerated verification plot at OUTPUT/figures/cleaned_data_verification.png")
