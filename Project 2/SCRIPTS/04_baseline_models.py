"""
04_baseline_models.py

This script trains baseline models for SPY price prediction using historical
SPY and VIX data. It focuses on simple linear regression models using only
SPY historical data to establish performance benchmarks.

Models trained:
- Linear regression with SPY only
- Linear regression with SPY lagged features
- Linear regression with SPY and VIX
- ARIMA time series model

Author: JARHeads Team
Date: March 2025
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories if they don't exist
os.makedirs('OUTPUT/model_results', exist_ok=True)
os.makedirs('OUTPUT/figures', exist_ok=True)

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and return model performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy (up/down prediction)
    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    directional_accuracy = (direction_true == direction_pred).mean()
    
    results = {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Directional_Accuracy': directional_accuracy
    }
    
    return results

# Load the feature-engineered data
print("Loading feature-engineered data...")
try:
    df = pd.read_csv('DATA/processed/feature_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Define the target variable
target = 'target_next_day_return'

# Prepare data for modeling - create train/test split
print("\nPreparing data for modeling...")
# Use 80% of data for training, 20% for testing
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"Training data size: {len(train_df)}, Test data size: {len(test_df)}")

# Store all model results for comparison
all_results = []

# 1. Baseline Model 1: Linear Regression with SPY only
print("\n1. Training Linear Regression with SPY only...")
# Use previous day's SPY price only
X_train_spy = train_df[['SPY_lag_1']].values
X_test_spy = test_df[['SPY_lag_1']].values
y_train = train_df[target].values
y_test = test_df[target].values

lr_spy = LinearRegression()
lr_spy.fit(X_train_spy, y_train)
y_pred_spy = lr_spy.predict(X_test_spy)

# Evaluate model
results_spy = evaluate_model(y_test, y_pred_spy, "LR_SPY_Only")
all_results.append(results_spy)
print(f"Model: LR_SPY_Only, RMSE: {results_spy['RMSE']:.6f}, Directional Accuracy: {results_spy['Directional_Accuracy']:.4f}")

# Save model
with open('OUTPUT/model_results/lr_spy_only.pkl', 'wb') as f:
    pickle.dump(lr_spy, f)

# 2. Baseline Model 2: Linear Regression with SPY lagged features
print("\n2. Training Linear Regression with SPY lagged features...")
spy_lag_cols = [col for col in train_df.columns if col.startswith('SPY_lag_')]
X_train_spy_lag = train_df[spy_lag_cols].values
X_test_spy_lag = test_df[spy_lag_cols].values

lr_spy_lag = LinearRegression()
lr_spy_lag.fit(X_train_spy_lag, y_train)
y_pred_spy_lag = lr_spy_lag.predict(X_test_spy_lag)

# Evaluate model
results_spy_lag = evaluate_model(y_test, y_pred_spy_lag, "LR_SPY_Lagged")
all_results.append(results_spy_lag)
print(f"Model: LR_SPY_Lagged, RMSE: {results_spy_lag['RMSE']:.6f}, Directional Accuracy: {results_spy_lag['Directional_Accuracy']:.4f}")

# Save model
with open('OUTPUT/model_results/lr_spy_lagged.pkl', 'wb') as f:
    pickle.dump(lr_spy_lag, f)

# 3. Baseline Model 3: Linear Regression with SPY and VIX
print("\n3. Training Linear Regression with SPY and VIX...")
# Use lagged features for both SPY and VIX
spy_vix_cols = [col for col in train_df.columns if col.startswith('SPY_lag_') or col.startswith('VIX_lag_')]
X_train_spy_vix = train_df[spy_vix_cols].values
X_test_spy_vix = test_df[spy_vix_cols].values

lr_spy_vix = LinearRegression()
lr_spy_vix.fit(X_train_spy_vix, y_train)
y_pred_spy_vix = lr_spy_vix.predict(X_test_spy_vix)

# Evaluate model
results_spy_vix = evaluate_model(y_test, y_pred_spy_vix, "LR_SPY_VIX")
all_results.append(results_spy_vix)
print(f"Model: LR_SPY_VIX, RMSE: {results_spy_vix['RMSE']:.6f}, Directional Accuracy: {results_spy_vix['Directional_Accuracy']:.4f}")

# Save model
with open('OUTPUT/model_results/lr_spy_vix.pkl', 'wb') as f:
    pickle.dump(lr_spy_vix, f)

# 4. ARIMA Time Series Model
print("\n4. Training ARIMA model...")
# Prepare the data for ARIMA - needs the actual prices
try:
    # Set Date as index
    train_ts = train_df.set_index('Date')['Price_SPY']
    test_ts = test_df.set_index('Date')['Price_SPY']
    
    # Fit ARIMA model
    # Start with a simple ARIMA(1,1,1) model
    model = sm.tsa.ARIMA(train_ts, order=(1, 1, 1))
    arima_model = model.fit()
    
    # Make predictions
    arima_preds = arima_model.forecast(steps=len(test_ts))
    
    # Convert price predictions to returns
    arima_returns = arima_preds.pct_change().dropna()
    test_returns = test_ts.pct_change().dropna()
    
    # Align the predictions with actual returns
    common_index = arima_returns.index.intersection(test_returns.index)
    if len(common_index) > 0:
        arima_returns = arima_returns.loc[common_index]
        test_returns = test_returns.loc[common_index]
        
        # Evaluate model
        results_arima = evaluate_model(test_returns.values, arima_returns.values, "ARIMA")
        all_results.append(results_arima)
        print(f"Model: ARIMA, RMSE: {results_arima['RMSE']:.6f}, Directional Accuracy: {results_arima['Directional_Accuracy']:.4f}")
        
        # Save model summary
        with open('OUTPUT/model_results/arima_summary.txt', 'w') as f:
            f.write(str(arima_model.summary()))
    else:
        print("Could not evaluate ARIMA model due to index alignment issues.")

except Exception as e:
    print(f"Error during ARIMA modeling: {e}")
    print("Proceeding with other models.")

# Create comparison visualization
print("\nCreating baseline model comparison visualization...")
# Convert results to DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.set_index('Model')

# Save results to CSV
results_df.to_csv('OUTPUT/model_results/baseline_metrics.csv')

# Bar plot of RMSE
plt.figure(figsize=(10, 6))
results_df['RMSE'].plot(kind='bar', color='skyblue')
plt.title('RMSE Comparison of Baseline Models')
plt.ylabel('RMSE (lower is better)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('OUTPUT/figures/baseline_rmse_comparison.png')

# Bar plot of Directional Accuracy
plt.figure(figsize=(10, 6))
results_df['Directional_Accuracy'].plot(kind='bar', color='lightgreen')
plt.title('Directional Accuracy Comparison of Baseline Models')
plt.ylabel('Accuracy (higher is better)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('OUTPUT/figures/baseline_accuracy_comparison.png')

# Plot predictions vs actual for the best model
print("\nCreating prediction visualization for best model...")
# Determine best model by RMSE
best_model_name = results_df['RMSE'].idxmin()
best_model_preds = y_pred_spy_vix  # Default to SPY_VIX model

if best_model_name == "LR_SPY_Only":
    best_model_preds = y_pred_spy
elif best_model_name == "LR_SPY_Lagged":
    best_model_preds = y_pred_spy_lag
elif best_model_name == "ARIMA" and 'results_arima' in locals():
    best_model_preds = arima_returns.values
    
# Create actual vs predicted plot
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'].iloc[:len(y_test)], y_test, label='Actual Returns', color='blue')
plt.plot(test_df['Date'].iloc[:len(best_model_preds)], best_model_preds, label=f'Predicted ({best_model_name})', color='red')
plt.title(f'Actual vs Predicted Returns - {best_model_name}')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('OUTPUT/figures/best_baseline_predictions.png')

print("\nBaseline modeling complete!")
print(f"Results saved to OUTPUT/model_results/baseline_metrics.csv")
print(f"Visualizations saved in OUTPUT/figures/")