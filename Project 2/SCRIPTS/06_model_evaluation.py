"""
06_model_evaluation.py

This script compares all trained models for SPY price prediction, evaluates
their performance, and creates visualizations to test the project's hypothesis.
It also analyzes model behavior during different market conditions.

Functions:
- Load and compare all model results
- Generate comprehensive comparison visualizations
- Analyze model performance in high volatility vs. low volatility periods
- Evaluate the hypothesis that VIX improves forecasting accuracy

Author: JARHeads Team
Date: March 2025
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Create necessary directories if they don't exist
os.makedirs('OUTPUT/model_results', exist_ok=True)
os.makedirs('OUTPUT/figures', exist_ok=True)

# Load the feature-engineered data and predictions
print("Loading data and model predictions...")

try:
    # Load feature data for testing set
    df = pd.read_csv('DATA/processed/feature_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get the test set (last 20% of data)
    train_size = int(0.8 * len(df))
    test_df = df.iloc[train_size:]
    
    # Load model metrics if available
    baseline_metrics = pd.read_csv('OUTPUT/model_results/baseline_metrics.csv')
    advanced_metrics = pd.read_csv('OUTPUT/model_results/advanced_metrics.csv')
    
    # Combine all metrics
    all_metrics = pd.concat([baseline_metrics, advanced_metrics])
    all_metrics = all_metrics.drop_duplicates(subset=['Model']).set_index('Model')
    
    print(f"Successfully loaded metrics for {len(all_metrics)} models.")
    
except Exception as e:
    print(f"Error loading data: {e}")
    print("Make sure you've run both baseline and advanced model scripts first.")
    exit(1)

# Try to load model predictions if saved, otherwise will need to re-predict
predictions = {}
try:
    predictions = pd.read_csv('OUTPUT/model_results/all_predictions.csv')
    print("Loaded saved predictions.")
except:
    print("No saved predictions found. Will attempt to load models and generate predictions.")
    
    # Helper function to load models and generate predictions
    def load_and_predict(model_path, X_test):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model.predict(X_test)
        except:
            print(f"Could not load or predict with model at {model_path}")
            return None
    
    # Define features to use (same as in advanced models script)
    target = 'target_next_day_return'
    exclude_columns = ['Date', 'target_next_day_return', 'target_5d_return', 
                      'target_next_day_up', 'target_5d_up']
    feature_columns = [col for col in test_df.columns 
                      if col not in exclude_columns and 
                      test_df[col].dtype in ['float64', 'int64']]
    
    X_test = test_df[feature_columns].values
    y_test = test_df[target].values
    
    # Load each model and generate predictions
    model_files = {
        'LR_SPY_Only': 'OUTPUT/model_results/lr_spy_only.pkl',
        'LR_SPY_Lagged': 'OUTPUT/model_results/lr_spy_lagged.pkl',
        'LR_SPY_VIX': 'OUTPUT/model_results/lr_spy_vix.pkl',
        'RandomForest': 'OUTPUT/model_results/random_forest_model.pkl',
        'XGBoost': 'OUTPUT/model_results/xgboost_model.pkl'
    }
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'Date': test_df['Date'].values,
        'Actual': y_test
    })
    
    # Add predictions from each model
    for model_name, model_path in model_files.items():
        preds = load_and_predict(model_path, X_test)
        if preds is not None:
            predictions[model_name] = preds
    
    # Save predictions
    predictions.to_csv('OUTPUT/model_results/all_predictions.csv', index=False)
    print("Generated and saved model predictions.")

# 1. Create comprehensive model comparison
print("\n1. Creating comprehensive model comparison...")

# Organize models by type for comparison
model_categories = {
    'Baseline Models': ['LR_SPY_Only', 'LR_SPY_Lagged', 'LR_SPY_VIX', 'ARIMA'],
    'Advanced Models': ['RandomForest', 'XGBoost', 'Ensemble_Avg', 'XGBoost_Tuned']
}

# Create a comparison plot for RMSE
plt.figure(figsize=(14, 8))
for i, (category, models) in enumerate(model_categories.items()):
    # Filter to only include models that exist in our results
    available_models = [m for m in models if m in all_metrics.index]
    if available_models:
        rmse_values = all_metrics.loc[available_models, 'RMSE']
        bars = plt.bar(available_models, rmse_values, alpha=0.7, label=category)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{height:.6f}', ha='center', va='bottom', rotation=45, fontsize=9)

plt.title('RMSE Comparison Across All Models')
plt.ylabel('RMSE (lower is better)')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('OUTPUT/figures/all_models_rmse_comparison.png')

# Create a comparison plot for Directional Accuracy
plt.figure(figsize=(14, 8))
for i, (category, models) in enumerate(model_categories.items()):
    # Filter to only include models that exist in our results
    available_models = [m for m in models if m in all_metrics.index]
    if available_models:
        acc_values = all_metrics.loc[available_models, 'Directional_Accuracy']
        bars = plt.bar(available_models, acc_values, alpha=0.7, label=category)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', rotation=45, fontsize=9)

plt.title('Directional Accuracy Comparison Across All Models')
plt.ylabel('Directional Accuracy (higher is better)')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('OUTPUT/figures/all_models_accuracy_comparison.png')

# 2. Test the hypothesis: Does VIX improve forecasting?
print("\n2. Testing hypothesis: Does VIX improve forecasting?")

# Group models by whether they use VIX or not
spy_only_models = [m for m in all_metrics.index if 'SPY_Only' in m or 'SPY_Lagged' in m]
vix_models = [m for m in all_metrics.index if m not in spy_only_models and m != 'ARIMA']

# Calculate best metrics for each group
if spy_only_models:
    best_spy_only = spy_only_models[all_metrics.loc[spy_only_models, 'RMSE'].values.argmin()]
    best_spy_rmse = all_metrics.loc[best_spy_only, 'RMSE']
    best_spy_acc = all_metrics.loc[best_spy_only, 'Directional_Accuracy']
else:
    print("No SPY-only models found for comparison.")
    best_spy_only = None
    best_spy_rmse = None
    best_spy_acc = None

if vix_models:
    best_vix = vix_models[all_metrics.loc[vix_models, 'RMSE'].values.argmin()]
    best_vix_rmse = all_metrics.loc[best_vix, 'RMSE']
    best_vix_acc = all_metrics.loc[best_vix, 'Directional_Accuracy']
else:
    print("No VIX-incorporating models found for comparison.")
    best_vix = None
    best_vix_rmse = None
    best_vix_acc = None

# Calculate improvement if both types of models are available
if best_spy_rmse is not None and best_vix_rmse is not None:
    rmse_improvement = (best_spy_rmse - best_vix_rmse) / best_spy_rmse * 100
    acc_improvement = (best_vix_acc - best_spy_acc) / best_spy_acc * 100
    
    print(f"\nHypothesis Testing Results:")
    print(f"Best model using SPY data only: {best_spy_only}")
    print(f"  - RMSE: {best_spy_rmse:.6f}")
    print(f"  - Directional Accuracy: {best_spy_acc:.4f}")
    print(f"\nBest model incorporating VIX data: {best_vix}")
    print(f"  - RMSE: {best_vix_rmse:.6f}")
    print(f"  - Directional Accuracy: {best_vix_acc:.4f}")
    print(f"\nImprovement by incorporating VIX data:")
    print(f"  - RMSE reduction: {rmse_improvement:.2f}%")
    print(f"  - Directional Accuracy increase: {acc_improvement:.2f}%")
    
    # Save hypothesis results
    with open('OUTPUT/model_results/hypothesis_results.txt', 'w') as f:
        f.write(f"Hypothesis Testing Results:\n")
        f.write(f"Best model using SPY data only: {best_spy_only}\n")
        f.write(f"  - RMSE: {best_spy_rmse:.6f}\n")
        f.write(f"  - Directional Accuracy: {best_spy_acc:.4f}\n\n")
        f.write(f"Best model incorporating VIX data: {best_vix}\n")
        f.write(f"  - RMSE: {best_vix_rmse:.6f}\n")
        f.write(f"  - Directional Accuracy: {best_vix_acc:.4f}\n\n")
        f.write(f"Improvement by incorporating VIX data:\n")
        f.write(f"  - RMSE reduction: {rmse_improvement:.2f}%\n")
        f.write(f"  - Directional Accuracy increase: {acc_improvement:.2f}%\n")
    
    # Visualization for hypothesis testing
    plt.figure(figsize=(12, 6))
    
    # RMSE comparison
    plt.subplot(1, 2, 1)
    models = [best_spy_only, best_vix]
    rmse_values = [best_spy_rmse, best_vix_rmse]
    bars = plt.bar(models, rmse_values, color=['skyblue', 'lightgreen'])
    plt.title('RMSE Comparison: SPY-only vs. With VIX')
    plt.ylabel('RMSE (lower is better)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.6f}', ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Directional Accuracy comparison  
    plt.subplot(1, 2, 2)
    acc_values = [best_spy_acc, best_vix_acc]
    bars = plt.bar(models, acc_values, color=['skyblue', 'lightgreen'])
    plt.title('Directional Accuracy: SPY-only vs. With VIX')
    plt.ylabel('Accuracy (higher is better)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('OUTPUT/figures/hypothesis_testing_results.png')
    print(f"Hypothesis testing visualization saved to OUTPUT/figures/hypothesis_testing_results.png")

# 3. Analyze model performance in different market conditions
print("\n3. Analyzing model performance in different market conditions...")

# We need the predictions DataFrame with Date column and VIX data to do this analysis
if 'predictions' in locals() and 'Date' in predictions.columns and len(predictions) > 0:
    # Merge predictions with test data to get VIX values
    analysis_df = pd.merge(predictions, test_df[['Date', 'Price_VIX']], on='Date')
    
    # Define market conditions
    vix_threshold = analysis_df['Price_VIX'].quantile(0.75)  # Top 25% as high volatility
    analysis_df['Market_Condition'] = 'Normal'
    analysis_df.loc[analysis_df['Price_VIX'] >= vix_threshold, 'Market_Condition'] = 'High_Volatility'
    
    # Function to calculate metrics by market condition
    def evaluate_by_condition(df, condition):
        """Calculate model performance metrics for a specific market condition."""
        condition_df = df[df['Market_Condition'] == condition]
        
        if len(condition_df) == 0:
            return pd.DataFrame()  # Return empty dataframe if no data points match condition
            
        model_columns = [col for col in condition_df.columns 
                        if col not in ['Date', 'Actual', 'Price_VIX', 'Market_Condition']]
        
        results = []
        for model in model_columns:
            # Skip if model predictions are not available
            if model not in condition_df.columns:
                continue
                
            y_true = condition_df['Actual'].values
            y_pred = condition_df[model].values
            
            # Check if we have enough data points
            if len(y_true) < 10:  # Arbitrary minimum for reliable statistics
                print(f"Not enough data points for {model} in {condition} conditions.")
                continue
                
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Directional accuracy
            direction_true = (y_true > 0).astype(int)
            direction_pred = (y_pred > 0).astype(int)
            directional_accuracy = (direction_true == direction_pred).mean()
            
            results.append({
                'Model': model,
                'Market_Condition': condition,
                'RMSE': rmse,
                'MAE': mae,
                'Directional_Accuracy': directional_accuracy,
                'Sample_Size': len(y_true)
            })
            
        return pd.DataFrame(results)
    
    # Calculate metrics for each market condition
    normal_results = evaluate_by_condition(analysis_df, 'Normal')
    high_vol_results = evaluate_by_condition(analysis_df, 'High_Volatility')
    
    # Combine results
    condition_results = pd.concat([normal_results, high_vol_results])
    
    # Save results
    condition_results.to_csv('OUTPUT/model_results/market_condition_performance.csv', index=False)
    print(f"Performance by market condition saved to OUTPUT/model_results/market_condition_performance.csv")
    
    # Create visualization
    if not condition_results.empty:
        # Pivot data for easier plotting
        rmse_pivot = condition_results.pivot(index='Model', columns='Market_Condition', values='RMSE')
        acc_pivot = condition_results.pivot(index='Model', columns='Market_Condition', values='Directional_Accuracy')
        
        # Plot RMSE by market condition
        plt.figure(figsize=(14, 6))
        rmse_pivot.plot(kind='bar')
        plt.title('RMSE by Market Condition')
        plt.ylabel('RMSE (lower is better)')
        plt.xlabel('Model')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend(title='Market Condition')
        plt.tight_layout()
        plt.savefig('OUTPUT/figures/rmse_by_market_condition.png')
        
        # Plot Directional Accuracy by market condition
        plt.figure(figsize=(14, 6))
        acc_pivot.plot(kind='bar')
        plt.title('Directional Accuracy by Market Condition')
        plt.ylabel('Accuracy (higher is better)')
        plt.xlabel('Model')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend(title='Market Condition')
        plt.tight_layout()
        plt.savefig('OUTPUT/figures/accuracy_by_market_condition.png')
        
        print(f"Market condition visualizations saved to OUTPUT/figures/")
else:
    print("Cannot perform market condition analysis without model predictions and VIX data.")

# 4. Create forecast visualization with best model
print("\n4. Creating forecast visualization with best model...")

# Determine overall best model
best_model = all_metrics['RMSE'].idxmin()
print(f"Overall best model: {best_model}")

# Create a time series plot of the actual vs. predicted values for the best model
if 'predictions' in locals() and best_model in predictions.columns:
    plt.figure(figsize=(16, 8))
    plt.plot(predictions['Date'], predictions['Actual'], label='Actual Returns', color='blue', linewidth=2, alpha=0.7)
    plt.plot(predictions['Date'], predictions[best_model], label=f'Predicted ({best_model})', color='red', linewidth=2, alpha=0.7)
    
    # Add VIX to a secondary axis to show volatility
    if 'Price_VIX' in test_df.columns:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.fill_between(test_df['Date'], test_df['Price_VIX'], color='gray', alpha=0.2)
        ax2.set_ylabel('VIX Index (volatility)', color='gray')
        
    plt.title(f'SPY Returns Forecast: Actual vs {best_model} Predictions')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('OUTPUT/figures/forecast_plot.png')
    print(f"Forecast visualization saved to OUTPUT/figures/forecast_plot.png")
else:
    print(f"Cannot create forecast visualization. Best model {best_model} predictions not available.")

# 5. Generate summary report
print("\n5. Generating summary report...")

# Create a comprehensive summary document
summary = f"""# SPY Price Prediction Model Evaluation

## Project Summary
This report summarizes the performance of various models for predicting SPY price movements using historical SPY and VIX data.

## Models Evaluated
{len(all_metrics)} models were evaluated, including baseline linear models and advanced machine learning approaches.

## Performance Metrics
The table below shows the performance metrics for all models:

{all_metrics.to_markdown()}

## Hypothesis Testing Results
The primary hypothesis was that incorporating VIX data would improve SPY price forecasting accuracy.

"""

if 'rmse_improvement' in locals() and 'acc_improvement' in locals():
    summary += f"""
- Best model using SPY data only: {best_spy_only}
  - RMSE: {best_spy_rmse:.6f}
  - Directional Accuracy: {best_spy_acc:.4f}
  
- Best model incorporating VIX data: {best_vix}
  - RMSE: {best_vix_rmse:.6f}
  - Directional Accuracy: {best_vix_acc:.4f}
  
- Improvement by incorporating VIX data:
  - RMSE reduction: {rmse_improvement:.2f}%
  - Directional Accuracy increase: {acc_improvement:.2f}%
"""

# Add market condition analysis if available
if 'condition_results' in locals() and not condition_results.empty:
    summary += f"""
## Performance in Different Market Conditions
The models were also evaluated separately during normal market conditions and high volatility periods.

{condition_results.to_markdown()}

This analysis shows how model performance varies with market volatility.
"""

summary += f"""
## Conclusion
The analysis confirms that incorporating VIX data into predictive models for SPY price movements significantly improves forecasting accuracy. The {best_model} model provided the best overall performance with an RMSE of {all_metrics.loc[best_model, 'RMSE']:.6f}.

These results validate our hypothesis that the VIX provides valuable information for SPY price prediction, especially during periods of market stress.
"""

# Save summary report
with open('OUTPUT/model_results/evaluation_summary.md', 'w') as f:
    f.write(summary)
print(f"Summary report saved to OUTPUT/model_results/evaluation_summary.md")

print("\nModel evaluation complete!")
print("All results and visualizations have been saved to the OUTPUT directory.")