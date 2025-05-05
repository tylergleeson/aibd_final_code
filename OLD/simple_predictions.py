import pandas as pd
import numpy as np
import joblib
from business_optimized_model import create_features

# Load data
df = pd.read_excel('final_total_clean.xlsx')
df['Date'] = pd.to_datetime(df['Date'])

# Filter for SKU 1 and 2023
df_2023 = df[(df['SKU'] == 1) & (df['Date'].dt.year == 2023)].copy()

# Aggregate quantities by date first
daily_quantities = df_2023.groupby('Date')['Quantity'].sum()
unique_dates = daily_quantities.index

# Set date as index for predictions
df_2023_indexed = df_2023.set_index('Date')
df_2023_indexed = df_2023_indexed[~df_2023_indexed.index.duplicated(keep='first')]

# Load model and make predictions
model = joblib.load('business_optimized_model.joblib')
features = create_features(df_2023_indexed)

# Keep only the features the model expects
expected_features = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'windspeedmean', 'Fuel_Price', 
                    'Avg_Price', 'day_of_week', 'month', 'is_weekend', 'day_of_year', 'week_of_year', 
                    'quarter', 'is_holiday', 'days_to_holiday', 'days_after_holiday', 'is_peak_season', 
                    'seasonal_cycle', 'seasonal_cycle_2', 'temp_weekend', 'temp_holiday', 'temp_peak_season', 
                    'temp_seasonal', 'lag_1', 'peak_lag_1', 'offpeak_lag_1', 'holiday_lag_1', 'lag_2', 
                    'peak_lag_2', 'offpeak_lag_2', 'holiday_lag_2', 'lag_3', 'peak_lag_3', 'offpeak_lag_3', 
                    'holiday_lag_3', 'lag_7', 'peak_lag_7', 'offpeak_lag_7', 'holiday_lag_7', 'lag_14', 
                    'peak_lag_14', 'offpeak_lag_14', 'holiday_lag_14', 'lag_21', 'peak_lag_21', 
                    'offpeak_lag_21', 'holiday_lag_21', 'rmean_7', 'rstd_7', 'peak_rmean_7', 
                    'offpeak_rmean_7', 'rmean_14', 'rstd_14', 'peak_rmean_14', 'offpeak_rmean_14', 
                    'rmean_28', 'rstd_28', 'peak_rmean_28', 'offpeak_rmean_28', 'seasonal_7', 'trend_7', 
                    'residual_7', 'seasonal_14', 'trend_14', 'residual_14', 'seasonal_30', 'trend_30', 
                    'residual_30', 'trend_slope', 'trend_acceleration', 'holiday_effect_short', 
                    'holiday_effect_long', 'post_holiday_effect', 'time_since_start', 
                    'time_since_start_squared']

features = features[expected_features]

# Make predictions
predictions = model.predict(features)

# Create result dataframe with both actual and predicted
result = pd.DataFrame({
    'Date': unique_dates,
    'Actual_Orders': daily_quantities.values,
    'Predicted_Orders': predictions
})

# Sort by date
result = result.sort_values('Date')

# Save to CSV
result.to_csv('2023_orders.csv', index=False)
print("Created 2023_orders.csv with actual and predicted orders") 