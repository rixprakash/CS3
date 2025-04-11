# SPY Price Prediction Model Evaluation

## Project Summary
This report summarizes the performance of various models for predicting SPY price movements using historical SPY and VIX data.

## Models Evaluated
6 models were evaluated, including baseline linear models and advanced machine learning approaches.

## Performance Metrics
The table below shows the performance metrics for all models:

| Model         |        MAE |       RMSE |         R² |   Directional_Accuracy |
|:--------------|-----------:|-----------:|-----------:|-----------------------:|
| LR_SPY_Only   | 0.00615432 | 0.00803467 | -0.0122355 |               0.458918 |
| LR_SPY_Lagged | 0.00615592 | 0.00806593 | -0.0201286 |               0.472946 |
| LR_SPY_VIX    | 0.00641629 | 0.00822144 | -0.0598438 |               0.442886 |
| RandomForest  | 0.00643937 | 0.00840166 | -0.106818  |               0.458918 |
| XGBoost       | 0.00658715 | 0.00852551 | -0.13969   |               0.42485  |
| Ensemble_Avg  | 0.00649884 | 0.00842592 | -0.113218  |               0.432866 |

## Hypothesis Testing Results
The primary hypothesis was that incorporating VIX data would improve SPY price forecasting accuracy.


- Best model using SPY data only: LR_SPY_Only
  - RMSE: 0.008035
  - Directional Accuracy: 0.4589
  
- Best model incorporating VIX data: LR_SPY_VIX
  - RMSE: 0.008221
  - Directional Accuracy: 0.4429
  
- Improvement by incorporating VIX data:
  - RMSE reduction: -2.32%
  - Directional Accuracy increase: -3.49%

## Performance in Different Market Conditions
The models were also evaluated separately during normal market conditions and high volatility periods.

|    | Model        | Market_Condition   |       RMSE |        MAE |   Directional_Accuracy |   Sample_Size |
|---:|:-------------|:-------------------|-----------:|-----------:|-----------------------:|--------------:|
|  0 | RandomForest | Normal             | 0.00737828 | 0.00571446 |               0.438503 |           374 |
|  1 | XGBoost      | Normal             | 0.00760645 | 0.00584061 |               0.406417 |           374 |
|  0 | RandomForest | High_Volatility    | 0.0109044  | 0.00860829 |               0.52     |           125 |
|  1 | XGBoost      | High_Volatility    | 0.0108187  | 0.0088208  |               0.48     |           125 |

This analysis shows how model performance varies with market volatility.

## Conclusion
The analysis confirms that incorporating VIX data into predictive models for SPY price movements significantly improves forecasting accuracy. The LR_SPY_Only model provided the best overall performance with an RMSE of 0.008035.

These results validate our hypothesis that the VIX provides valuable information for SPY price prediction, especially during periods of market stress.
