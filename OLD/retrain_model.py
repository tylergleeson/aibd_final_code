import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import joblib
from business_optimized_model import create_features, weighted_quantile_loss

# Load data
print("Loading data...")
df = pd.read_excel('final_total_clean sku1.xlsx')
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate quantities by date first
print("Aggregating quantities by date...")
df = df.groupby('Date').agg({
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

# Set date as index
df = df.set_index('Date')

# Create features
print("Creating features...")
features = create_features(df)

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

# Prepare target
target = df['Quantity']

# Split data using TimeSeriesSplit
print("Splitting data...")
tscv = TimeSeriesSplit(n_splits=5)
train_idx, val_idx = list(tscv.split(features))[-1]

X_train = features.iloc[train_idx]
y_train = target.iloc[train_idx]
X_val = features.iloc[val_idx]
y_val = target.iloc[val_idx]

# Define model parameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'custom',
    'num_leaves': 16,
    'max_depth': 6,
    'learning_rate': 0.03,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'lambda_l1': 0.3,
    'lambda_l2': 0.2,
    'min_data_in_leaf': 25,
    'min_gain_to_split': 0.2,
    'verbose': -1
}

# Create LightGBM datasets
print("Creating LightGBM datasets...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train model
print("Training model...")
callbacks = [lgb.early_stopping(50)]
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    num_boost_round=1000,
    callbacks=callbacks,
    feval=lambda preds, train_data: [('custom_metric', weighted_quantile_loss(train_data.get_label(), preds), False)]
)

# Save model
print("Saving model...")
joblib.dump(model, 'business_optimized_model_new.joblib')

print("Model training complete!")

# Print feature importance
importance = pd.DataFrame({
    'feature': expected_features,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(importance.head(10))

# Print some statistics about the aggregation
print("\nData statistics:")
print(f"Number of unique dates: {len(df)}")
print("\nSample of daily quantities:")
print(df['Quantity'].head()) 