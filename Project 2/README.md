# SPY Prediction Project

This repository contains a comprehensive analysis of SPY (S&P 500 ETF) price movements based on historical SPY data and VIX (Volatility Index) information. The project aims to forecast SPY price trends using various machine learning models and test the hypothesis that VIX data can significantly improve prediction accuracy.

## Contents of the Repository

This repository follows the TIER Protocol 4.0 structure and contains:

- **DATA/**: Contains all data files used in the project
- **SCRIPTS/**: Contains all source code for data processing and model development
- **OUTPUT/**: Contains generated visualizations and model results
- **README.md**: This file, providing an overview and instructions
- **LICENSE.md**: MIT License for this project

## Software and Platform

### Software Used
- Python 3.9
- Jupyter Notebook for exploration and development

### Required Packages
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, XGBoost
- **Time Series Analysis**: statsmodels
- **Deep Learning** (optional): TensorFlow, Keras

### Platform
- The project was developed on macOS but should be compatible with Windows and Linux environments.
- Python 3.7+ is required for all scripts to run properly.

## Map of Documentation

```
Project 2/
│
├── README.md               # Overview and instructions
├── LICENSE.md              # MIT License
│
├── SCRIPTS/                # All source code
│   ├── 01_data_cleaning.py     # Initial data processing
│   ├── 02_EDAcode.ipynb        # EDA scripts
│   ├── 03_feature_engineering.py  # Feature creation
│   ├── 04_baseline_models.py   # Linear regression models
│   ├── 05_advanced_models.py   # XGBoost and Random Forest models
│   └── 06_model_evaluation.py  # Performance assessment
│
├── DATA/                   # Data files
│   ├── CBOE Volatility Index Historical Data.csv  # Original VIX data
│   ├── SPY ETF Stock Price History.csv            # Original SPY data
│   ├── merged_data.csv             # Combined SPY/VIX dataset
│   ├── raw/
│   │   ├── merged_data.csv         # Combined SPY/VIX dataset
│   │   └── data_dictionary.md      # Column descriptions
│   └── processed/
│       ├── clean_data.csv          # Cleaned dataset
│       ├── feature_data.csv        # Dataset with engineered features
│       └── feature_summary.csv     # Feature summary statistics
│
└── OUTPUT/                 # Results and visualizations
    ├── figures/
    │   ├── Price_VIX_outliers_boxplot.png         # VIX outliers visualization
    │   ├── accuracy_by_market_condition.png       # Performance by market regime
    │   ├── advanced_accuracy_comparison.png       # Advanced models accuracy
    │   ├── advanced_rmse_comparison.png           # Advanced models RMSE
    │   ├── all_models_accuracy_comparison.png     # All models accuracy
    │   ├── all_models_rmse_comparison.png         # All models RMSE
    │   ├── baseline_accuracy_comparison.png       # Baseline models accuracy
    │   ├── baseline_rmse_comparison.png           # Baseline models RMSE
    │   ├── best_advanced_predictions.png          # Time series predictions
    │   ├── best_advanced_scatter.png              # Scatter of predictions
    │   ├── best_baseline_predictions.png          # Baseline predictions
    │   ├── cleaned_data_verification.png          # Data quality check
    │   ├── feature_correlation_heatmap.png        # Feature correlations
    │   ├── hypothesis_testing_results.png         # Hypothesis test visualization
    │   ├── rf_feature_importance.png              # RandomForest features
    │   ├── rmse_by_market_condition.png           # RMSE by market regime
    │   └── xgb_feature_importance.png             # XGBoost features
    └── model_results/
        ├── advanced_metrics.csv                   # Advanced models metrics
        ├── all_model_metrics.csv                  # Combined model metrics
        ├── all_predictions.csv                    # All model predictions
        ├── baseline_metrics.csv                   # Baseline model metrics
        ├── evaluation_summary.md                  # Project conclusions
        ├── hypothesis_results.txt                 # Hypothesis test results
        ├── lr_spy_lagged.pkl                      # Linear model with lags
        ├── lr_spy_only.pkl                        # SPY-only model
        ├── lr_spy_vix.pkl                         # Linear model with VIX
        ├── market_condition_performance.csv       # Results by market regime
        ├── random_forest_model.pkl                # RandomForest model file
        └── xgboost_model.pkl                      # XGBoost model file
```

## Instructions for Reproducing Results

Follow these steps to reproduce the analysis and results of this project:

### 1. Environment Setup
1. Clone this repository to your local machine.
2. Install Python 3.9 if not already installed.
3. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install required packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels
   ```

### 2. Data Preparation
1. Navigate to the project root directory.
2. Ensure that the DATA folder contains the original dataset files:
   - CBOE Volatility Index Historical Data.csv
   - SPY ETF Stock Price History.csv
   - merged_data.csv
3. Run the data cleaning script:
   ```
   python SCRIPTS/01_data_cleaning.py
   ```
   This will process the raw merged_data.csv and create clean_data.csv in the DATA/processed directory.

### 3. Exploratory Data Analysis
1. Run the EDA script (02_EDAcode.ipynb) to generate visualizations:
   ```
   python SCRIPTS/02_EDAcode.py
   ```
   Alternatively, you can step through the EDAcode.ipynb notebook for an interactive analysis.
2. Review the generated visualizations in the OUTPUT/figures directory to understand the relationship between SPY and VIX.

### 4. Feature Engineering
1. Run the feature engineering script:
   ```
   python SCRIPTS/03_feature_engineering.py
   ```
   This will generate derived features and save the enhanced dataset as feature_data.csv in DATA/processed.

### 5. Model Development
1. Train baseline models:
   ```
   python SCRIPTS/04_baseline_models.py
   ```
   This establishes performance benchmarks using linear models with varying features.
2. Train advanced models:
   ```
   python SCRIPTS/05_advanced_models.py
   ```
   This trains more sophisticated machine learning models incorporating both SPY and VIX features.
3. Evaluate and compare all models:
   ```
   python SCRIPTS/06_model_evaluation.py
   ```
   This generates comparative metrics and visualizations in OUTPUT/model_results.

### 6. Results Analysis

1. Review the model performance metrics in OUTPUT/model_results to determine which approach provides the best forecasting accuracy.
2. Examine feature importance visualizations to understand which features contribute most to prediction performance.
3. Compare models that use only SPY data with those that incorporate VIX data to test our central hypothesis.

## References

[1] Chicago Board Options Exchange (CBOE), "CBOE VIX Volatility Index," [Online]. Available: https://www.cboe.com/tradable_products/vix/

[2] J. Brownlee, Deep Learning for Time Series Forecasting. Machine Learning Mastery, 2018.

[3] PyTorch Forecasting, [Online]. Available: https://pytorch-forecasting.readthedocs.io/

[4] "Time series forecasting," TensorFlow, [Online]. Available: https://www.tensorflow.org/tutorials/structured_data/time_series

[5] "SPDR S&P 500 Historical Data," Investing.com, [Online]. Available: https://www.investing.com/etfs/spdr-s-p-500-historical-data

[6] R. Nicholson, "Tutorial: Time Series Forecasting with XGBoost," Kaggle, [Online]. Available: https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost

[7] "SPDR S&P 500 ETF Trust," VettaFi, [Online]. Available: https://etfdb.com/etf/SPY/

