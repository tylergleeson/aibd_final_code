import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from business_optimized_model import create_features
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
print("Loading data...")
df = pd.read_excel('final_total_clean sku1.xlsx')
df['Date'] = pd.to_datetime(df['Date'])

# Print date range
print("\nData date range:")
print(f"Earliest date: {df['Date'].min()}")
print(f"Latest date: {df['Date'].max()}")

# Filter for 2023 and aggregate by date
print("\nFiltering and aggregating 2023 data...")
df_2023 = df[df['Date'].dt.year == 2023].copy()
print(f"Number of raw 2023 records before aggregation: {len(df_2023)}")

df_2023 = df_2023.groupby('Date').agg({
    'Quantity': 'sum',
    'tempmax': 'first',
    'tempmin': 'first',
    'temp': 'first',
    'humidity': 'first',
    'precip': 'first',
    'windspeedmean': 'first',
    'Fuel_Price': 'first',
    'Avg_Price': 'first'
}).reset_index()

print(f"Number of unique dates in 2023 after aggregation: {len(df_2023)}")
print("\nFirst few dates in 2023:")
print(df_2023[['Date', 'Quantity']].head())
print("\nLast few dates in 2023:")
print(df_2023[['Date', 'Quantity']].tail())

# Set date as index for predictions
df_2023_indexed = df_2023.set_index('Date')

# Load both models
print("\nLoading models and making predictions...")
model_new = joblib.load('business_optimized_model_new.joblib')
model_old = joblib.load('business_optimized_model.joblib')

# Print model information
print("\nModel Information:")
print("Old model feature names:", model_old.feature_name_)
print("New model feature names:", model_new.feature_name_)

# Create features
features = create_features(df_2023_indexed)

# Keep only the expected features
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

# Get predictions from both models
predictions_new = model_new.predict(features)
predictions_old = model_old.predict(features)

# Create result dataframe
result = pd.DataFrame({
    'Date': df_2023.Date,
    'Actual_Orders': df_2023.Quantity,
    'Predicted_New': predictions_new,
    'Predicted_Old': predictions_old
})

# Calculate overall metrics
print("\nOverall Performance Metrics:")
print("\nNew Model:")
print(f"MAE: {mean_absolute_error(result.Actual_Orders, result.Predicted_New):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(result.Actual_Orders, result.Predicted_New)):.2f}")
print(f"Mean Actual Orders: {result.Actual_Orders.mean():.2f}")
print(f"Mean Predicted Orders: {result.Predicted_New.mean():.2f}")

print("\nOld Model:")
print(f"MAE: {mean_absolute_error(result.Actual_Orders, result.Predicted_Old):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(result.Actual_Orders, result.Predicted_Old)):.2f}")
print(f"Mean Actual Orders: {result.Actual_Orders.mean():.2f}")
print(f"Mean Predicted Orders: {result.Predicted_Old.mean():.2f}")

# Calculate metrics by season
print("\nPeak Season Performance (May-September):")
peak_mask = (result.Date.dt.month >= 5) & (result.Date.dt.month <= 9)
peak_data = result[peak_mask]
print("\nNew Model Peak Season:")
print(f"MAE: {mean_absolute_error(peak_data.Actual_Orders, peak_data.Predicted_New):.2f}")
print(f"Mean Actual Orders: {peak_data.Actual_Orders.mean():.2f}")
print(f"Mean Predicted Orders: {peak_data.Predicted_New.mean():.2f}")

print("\nOld Model Peak Season:")
print(f"MAE: {mean_absolute_error(peak_data.Actual_Orders, peak_data.Predicted_Old):.2f}")
print(f"Mean Actual Orders: {peak_data.Actual_Orders.mean():.2f}")
print(f"Mean Predicted Orders: {peak_data.Predicted_Old.mean():.2f}")

# Create visualization
plt.figure(figsize=(15, 8))

# Plot actual values
plt.plot(result.Date, result.Actual_Orders, 
         label='Actual Orders', color='#2ecc71', linewidth=2)

# Plot new model predictions
plt.plot(result.Date, result.Predicted_New, 
         label='New Model Predictions', color='#e74c3c', linewidth=2, alpha=0.8)

# Plot old model predictions
plt.plot(result.Date, result.Predicted_Old, 
         label='Old Model Predictions', color='#3498db', linewidth=2, alpha=0.8)

# Add shaded area for peak season (May-September)
peak_season_mask = (result.Date.dt.month >= 5) & (result.Date.dt.month <= 9)
plt.fill_between(result.Date, plt.ylim()[0], plt.ylim()[1],
                where=peak_season_mask, color='gray', alpha=0.1, label='Peak Season')

# Customize plot
plt.title('2023 Pool Guy Chlorine: Actual vs Predicted Orders', fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Order Quantity', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('2023_predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Save predictions to CSV
result.to_csv('2023_predictions_comparison.csv', index=False)
print("\nResults saved to:")
print("- 2023_predictions_comparison.png")
print("- 2023_predictions_comparison.csv")

# Print some example predictions
print("\nSample predictions (first 5 days):")
print(result[['Date', 'Actual_Orders', 'Predicted_New', 'Predicted_Old']].head().to_string()) 