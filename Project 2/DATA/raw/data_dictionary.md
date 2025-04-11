# Data Dictionary for SPY/VIX Dataset

## Dataset: merged_data.csv

This dataset combines historical data for SPY (S&P 500 ETF) and VIX (Volatility Index) from January 1, 2015, to March 1, 2025.

### Column Descriptions

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| Date | String | The date of the recorded market data in format YYYY-MM-DD |
| Price_VIX | Float | Closing price of the VIX index for the trading day |
| Open_VIX | Float | Opening price of the VIX index for the trading day |
| High_VIX | Float | Highest price the VIX index reached during the trading day |
| Low_VIX | Float | Lowest price the VIX index reached during the trading day |
| Vol._VIX | Float | Trading volume for the VIX index (number of contracts traded) |
| Change %_VIX | String | Percentage change in the VIX index price from the previous trading day |
| Price_SPY | Float | Closing price of the SPY ETF for the trading day |
| Open_SPY | Float | Opening price of the SPY ETF for the trading day |
| High_SPY | Float | Highest price the SPY ETF reached during the trading day |
| Low_SPY | Float | Lowest price the SPY ETF reached during the trading day |
| Vol._SPY | String | Trading volume for the SPY ETF (number of shares traded) |
| Change %_SPY | String | Percentage change in the SPY ETF price from the previous trading day |

### Notes

- The dataset contains 2,555 observations (trading days).
- Both percentage change columns are stored as strings with "%" symbols and will need to be converted to numeric format for analysis.
- Volume columns may contain abbreviated values (e.g., "1.5M" for 1.5 million) that will need to be standardized.
- The Date column should be converted to datetime format for time series analysis.