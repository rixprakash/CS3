"""
05_advanced_models.py

This script trains advanced machine learning models for SPY price prediction using
both SPY and VIX historical data with engineered features. It focuses on tree-based
algorithms which can capture non-linear relationships in the data.

Models trained:
- Random Forest Regressor
- XGBoost Regressor
- Ensemble model (combining multiple predictions)

Author: JARHeads Team
Date: March 2025
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories if they don't exist
os.makedirs('OUTPUT/model_results', exist_ok=True)
os.makedirs('OUTPUT/figures', exist_ok=True)

# Function to evaluate model performance (same as in baseline script for consistency)
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
# Using same split method as baseline for consistency
print("\nPreparing data for modeling...")
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"Training data size: {len(train_df)}, Test data size: {len(test_df)}")

# Define features to use
# Exclude date, target variables, and any redundant columns
exclude_columns = ['Date', 'target_next_day_return', 'target_5d_return', 
                  'target_next_day_up', 'target_5d_up']

# For numeric features only
feature_columns = [col for col in train_df.columns 
                  if col not in exclude_columns and 
                  train_df[col].dtype in ['float64', 'int64']]

print(f"Using {len(feature_columns)} features for modeling")

# Prepare the training and test sets
X_train = train_df[feature_columns].values
X_test = test_df[feature_columns].values
y_train = train_df[target].values
y_test = test_df[target].values

# Store all model results for comparison
all_results = []

# Try loading baseline results if available
try:
    baseline_results = pd.read_csv('OUTPUT/model_results/baseline_metrics.csv')
    print("\nLoaded baseline model results for comparison.")
except:
    print("\nNo baseline results found. Creating new results file.")
    baseline_results = pd.DataFrame()

# 1. Random Forest Model
print("\n1. Training Random Forest model...")
# Start with a simpler model for speed
rf_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}

rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate model
results_rf = evaluate_model(y_test, y_pred_rf, "RandomForest")
all_results.append(results_rf)
print(f"Model: RandomForest, RMSE: {results_rf['RMSE']:.6f}, Directional Accuracy: {results_rf['Directional_Accuracy']:.4f}")

# Save model
with open('OUTPUT/model_results/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Features by Importance (Random Forest)')
plt.tight_layout()
plt.savefig('OUTPUT/figures/rf_feature_importance.png')
print("Feature importance plot saved to OUTPUT/figures/rf_feature_importance.png")

# 2. XGBoost Model
print("\n2. Training XGBoost model...")
xgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate model
results_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")
all_results.append(results_xgb)
print(f"Model: XGBoost, RMSE: {results_xgb['RMSE']:.6f}, Directional Accuracy: {results_xgb['Directional_Accuracy']:.4f}")

# Save model
with open('OUTPUT/model_results/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Feature importance for XGBoost
xgb_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': xgb_model.feature_importances_
})
xgb_importance = xgb_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_importance.head(20))
plt.title('Top 20 Features by Importance (XGBoost)')
plt.tight_layout()
plt.savefig('OUTPUT/figures/xgb_feature_importance.png')
print("Feature importance plot saved to OUTPUT/figures/xgb_feature_importance.png")

# 3. Optional: Hyperparameter Tuning with Grid Search
# Note: This is computationally intensive and commented out by default
"""
print("\n3. Hyperparameter tuning for XGBoost...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Use time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
best_xgb = grid_search.best_estimator_

# Evaluate tuned model
y_pred_best_xgb = best_xgb.predict(X_test)
results_best_xgb = evaluate_model(y_test, y_pred_best_xgb, "XGBoost_Tuned")
all_results.append(results_best_xgb)
print(f"Model: XGBoost_Tuned, RMSE: {results_best_xgb['RMSE']:.6f}, Directional Accuracy: {results_best_xgb['Directional_Accuracy']:.4f}")

# Save best model
with open('OUTPUT/model_results/xgboost_tuned_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)
"""

# 4. Ensemble Model - Simple Average
print("\n4. Creating ensemble model (average of RF and XGB predictions)...")
# Simple averaging of predictions
y_pred_ensemble = (y_pred_rf + y_pred_xgb) / 2

# Evaluate ensemble
results_ensemble = evaluate_model(y_test, y_pred_ensemble, "Ensemble_Avg")
all_results.append(results_ensemble)
print(f"Model: Ensemble_Avg, RMSE: {results_ensemble['RMSE']:.6f}, Directional Accuracy: {results_ensemble['Directional_Accuracy']:.4f}")

# 5. Compare all advanced models
# Convert results to DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.set_index('Model')

# Save results to CSV
results_df.to_csv('OUTPUT/model_results/advanced_metrics.csv')

# Combine with baseline results if available
if not baseline_results.empty:
    baseline_results = baseline_results.set_index('Model')
    combined_results = pd.concat([baseline_results, results_df])
    combined_results.to_csv('OUTPUT/model_results/all_model_metrics.csv')
    print("\nCombined metrics with baseline models saved to OUTPUT/model_results/all_model_metrics.csv")

# Create comparison visualization for all advanced models
print("\nCreating advanced model comparison visualizations...")

# Bar plot of RMSE
plt.figure(figsize=(10, 6))
results_df['RMSE'].plot(kind='bar', color='skyblue')
plt.title('RMSE Comparison of Advanced Models')
plt.ylabel('RMSE (lower is better)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('OUTPUT/figures/advanced_rmse_comparison.png')

# Bar plot of Directional Accuracy
plt.figure(figsize=(10, 6))
results_df['Directional_Accuracy'].plot(kind='bar', color='lightgreen')
plt.title('Directional Accuracy Comparison of Advanced Models')
plt.ylabel('Accuracy (higher is better)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('OUTPUT/figures/advanced_accuracy_comparison.png')

# Plot predictions vs actual for the best model
print("\nCreating prediction visualization for best advanced model...")
# Determine best model by RMSE
best_model_name = results_df['RMSE'].idxmin()
best_model_preds = y_pred_ensemble  # Default to ensemble model

if best_model_name == "RandomForest":
    best_model_preds = y_pred_rf
elif best_model_name == "XGBoost":
    best_model_preds = y_pred_xgb

# Create actual vs predicted plot
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'], y_test, label='Actual Returns', color='blue', alpha=0.7)
plt.plot(test_df['Date'], best_model_preds, label=f'Predicted ({best_model_name})', color='red', alpha=0.7)
plt.title(f'Actual vs Predicted Returns - {best_model_name}')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('OUTPUT/figures/best_advanced_predictions.png')

# And a scatterplot of actual vs predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test, best_model_preds, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # diagonal line
plt.title(f'Actual vs Predicted Scatter - {best_model_name}')
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('OUTPUT/figures/best_advanced_scatter.png')

print("\nAdvanced modeling complete!")
print(f"Results saved to OUTPUT/model_results/advanced_metrics.csv")
print(f"Visualizations saved in OUTPUT/figures/")

# Create a results summary for our hypothesis
print("\nAnalyzing results to test hypothesis...")
try:
    # If we have the combined results
    if 'combined_results' in locals():
        results = combined_results
    else:
        # Try to load all metrics
        results = pd.read_csv('OUTPUT/model_results/all_model_metrics.csv')
        results = results.set_index('Model')
    
    # Identify best model with SPY only
    spy_only_models = [m for m in results.index if 'SPY_Only' in m or 'SPY_Lagged' in m]
    if spy_only_models:
        best_spy_only = spy_only_models[results.loc[spy_only_models, 'RMSE'].argmin()]
        best_spy_rmse = results.loc[best_spy_only, 'RMSE']
        
        # Find best overall model
        best_overall = results['RMSE'].idxmin()
        best_overall_rmse = results.loc[best_overall, 'RMSE']
        
        # Calculate improvement
        improvement = (best_spy_rmse - best_overall_rmse) / best_spy_rmse * 100
        
        print(f"\nHypothesis Testing Results:")
        print(f"Best model using SPY data only: {best_spy_only}, RMSE: {best_spy_rmse:.6f}")
        print(f"Best overall model: {best_overall}, RMSE: {best_overall_rmse:.6f}")
        print(f"Improvement by incorporating VIX data: {improvement:.2f}%")
        
        # Save this info
        with open('OUTPUT/model_results/hypothesis_results.txt', 'w') as f:
            f.write(f"Hypothesis Testing Results:\n")
            f.write(f"Best model using SPY data only: {best_spy_only}, RMSE: {best_spy_rmse:.6f}\n")
            f.write(f"Best overall model: {best_overall}, RMSE: {best_overall_rmse:.6f}\n")
            f.write(f"Improvement by incorporating VIX data: {improvement:.2f}%\n")
        
        print("Hypothesis testing results saved to OUTPUT/model_results/hypothesis_results.txt")
except Exception as e:
    print(f"Could not complete hypothesis testing: {e}")
    print("Complete the baseline models first or run evaluation script for full comparison.")
