import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import joblib
from datetime import datetime
import importlib.util
import sys
from typing import Tuple, List
from statsmodels.tsa.seasonal import seasonal_decompose
import holidays

# Import loss function
spec = importlib.util.spec_from_file_location("loss_function", "loss function.py")
loss_module = importlib.util.module_from_spec(spec)
sys.modules["loss_function"] = loss_module
spec.loader.exec_module(loss_module)

# Constants
ID_COL = 'SKU'
DATE_COL = 'Date'
TARGET = 'Quantity'
POOL_GUY_CHLORINE_SKU = 1.0
PRODUCT_NAME = "Pool Guy Chlorine"

# External features
EXOG_FEATURES = [
    'tempmax', 'tempmin', 'temp', 'humidity',
    'precip', 'windspeedmean', 'Fuel_Price',
    'Avg_Price'
]

def is_holiday(date):
    """Check if date is a US holiday."""
    us_holidays = holidays.US()
    return date in us_holidays

def is_peak_season(month):
    """Define peak pool season (May through September)."""
    return 5 <= month <= 9

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced time series features with seasonal components."""
    df = df.copy()
    
    # Basic time features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['quarter'] = df.index.quarter
    
    # Enhanced holiday features
    df['is_holiday'] = df.index.map(is_holiday).astype(int)
    df['days_to_holiday'] = 0
    df['days_after_holiday'] = 0
    df['holiday_type'] = 'None'
    
    # Calculate days to/from next/previous holiday for each date
    us_holidays = holidays.US(years=[2021, 2022, 2023])
    for date in df.index:
        # Find next holiday
        next_holidays = [h for h in us_holidays.keys() if pd.Timestamp(h) > date]
        if next_holidays:
            next_holiday = min(next_holidays)
            df.loc[date, 'days_to_holiday'] = (pd.Timestamp(next_holiday) - date).days
            df.loc[date, 'holiday_type'] = us_holidays.get(next_holiday, 'None')
        
        # Find previous holiday
        prev_holidays = [h for h in us_holidays.keys() if pd.Timestamp(h) < date]
        if prev_holidays:
            prev_holiday = max(prev_holidays)
            df.loc[date, 'days_after_holiday'] = (date - pd.Timestamp(prev_holiday)).days
    
    # Enhanced seasonal features
    df['is_peak_season'] = df['month'].map(is_peak_season).astype(int)
    df['seasonal_cycle'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['seasonal_cycle_2'] = np.sin(4 * np.pi * df['day_of_year'] / 365)
    
    # Temperature interaction features with seasonal consideration
    df['temp_weekend'] = df['temp'] * df['is_weekend']
    df['temp_holiday'] = df['temp'] * df['is_holiday']
    df['temp_peak_season'] = df['temp'] * df['is_peak_season']
    df['temp_seasonal'] = df['temp'] * df['seasonal_cycle']
    
    # Enhanced lag features with exponential decay
    for lag in [1, 2, 3, 7, 14, 21]:
        df[f'lag_{lag}'] = df[TARGET].shift(lag)
        # Create season-specific lags with decay
        decay_factor = np.exp(-lag/7)  # Weekly decay
        df[f'peak_lag_{lag}'] = df[f'lag_{lag}'] * df['is_peak_season'] * decay_factor
        df[f'offpeak_lag_{lag}'] = df[f'lag_{lag}'] * (1 - df['is_peak_season']) * decay_factor
        # Create holiday-specific lags with decay
        df[f'holiday_lag_{lag}'] = df[f'lag_{lag}'] * df['is_holiday'] * decay_factor
    
    # Enhanced rolling windows with seasonal consideration
    for window in [7, 14, 28]:
        # Basic rolling stats with exponential weights
        weights = np.exp(-np.arange(window)/window)
        weights = weights / weights.sum()
        
        df[f'rmean_{window}'] = df[TARGET].shift(1).rolling(window).apply(
            lambda x: np.sum(x * weights) if len(x) == window else np.nan
        )
        df[f'rstd_{window}'] = df[TARGET].shift(1).rolling(window).apply(
            lambda x: np.sqrt(np.sum(weights * (x - np.sum(x * weights))**2)) if len(x) == window else np.nan
        )
        
        # Season-specific rolling means with weights
        peak_mask = df['is_peak_season'] == 1
        df[f'peak_rmean_{window}'] = df[TARGET].where(peak_mask).shift(1).rolling(window).apply(
            lambda x: np.sum(x * weights) if len(x) == window else np.nan
        )
        df[f'offpeak_rmean_{window}'] = df[TARGET].where(~peak_mask).shift(1).rolling(window).apply(
            lambda x: np.sum(x * weights) if len(x) == window else np.nan
        )
    
    # Multiple seasonal decomposition periods with trend smoothing
    for period in [7, 14, 30]:
        decomposition = seasonal_decompose(
            df[TARGET].fillna(0),
            period=period,
            extrapolate_trend=True
        )
        df[f'seasonal_{period}'] = decomposition.seasonal
        # Apply exponential smoothing to trend
        df[f'trend_{period}'] = decomposition.trend.ewm(span=period).mean()
        df[f'residual_{period}'] = decomposition.resid
    
    # Enhanced trend features with smoothing
    df['trend_slope'] = df['trend_30'].diff().ewm(span=7).mean()
    df['trend_acceleration'] = df['trend_slope'].diff().ewm(span=7).mean()
    
    # Holiday proximity effects with enhanced decay rates
    df['holiday_effect_short'] = np.exp(-df['days_to_holiday'] / 3)  # Short-term effect
    df['holiday_effect_long'] = np.exp(-df['days_to_holiday'] / 14)  # Long-term effect
    df['post_holiday_effect'] = np.exp(-df['days_after_holiday'] / 7)  # Post-holiday effect
    
    # Add time-based regularization features
    df['time_since_start'] = (df.index - df.index.min()).days
    df['time_since_start_squared'] = df['time_since_start'] ** 2
    
    # Fill any remaining NaN values with 0
    df = df.fillna(0)
    
    return df

def weighted_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Loss function with company-specified costs:
    - Underprediction cost: 11.54 (lost sales, customer dissatisfaction)
    - Overprediction cost: 1.78 (holding costs, wastage)
    """
    total_cost = 0
    for true, pred in zip(y_true, y_pred):
        if pred > true:  # Over-prediction
            total_cost += 1.78 * (pred - true)  # Company-specified holding cost
        else:  # Under-prediction
            total_cost += 11.54 * (true - pred)  # Company-specified shortage cost
    return total_cost / len(y_true)

def calculate_safety_stock(errors: np.ndarray, temp: float, is_peak: bool, is_holiday: bool) -> float:
    """
    Calculate consolidated safety stock based on multiple factors.
    Uses 80th percentile instead of 87th and includes bounds.
    """
    # Base safety stock at 80th percentile
    base_safety = np.percentile(errors, 80)
    
    # Temperature adjustment (reduced impact)
    temp_adj = 0.0
    if temp >= 85:
        temp_adj = 1.0  # Reduced from 2.0
    elif temp >= 75:
        temp_adj = 0.5  # Reduced from 1.0
    elif temp <= 60:
        temp_adj = -0.5  # Reduced from -1.0
    
    # Season and holiday adjustments (reduced)
    season_adj = 1.0 if is_peak else 0.5
    holiday_adj = 0.5 if is_holiday else 0.0
    
    # Combine adjustments with bounds
    total_adj = base_safety * season_adj + temp_adj + holiday_adj
    
    # Apply upper bound based on historical patterns
    max_adjustment = np.percentile(errors, 95)  # Cap at 95th percentile
    return min(total_adj, max_adjustment)

def train_model():
    print("Loading and preparing data...")
    df = pd.read_excel('final_total_clean.xlsx', parse_dates=[DATE_COL])
    
    # Filter for SKU 1
    df = df[df[ID_COL] == 1]
    
    # Create complete date range
    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Create DataFrame with all dates
    daily_df = pd.DataFrame(index=all_dates)
    daily_df.index.name = DATE_COL
    
    # Aggregate quantities
    quantities_by_date = df.groupby(DATE_COL)[TARGET].sum()
    daily_df = daily_df.join(quantities_by_date)
    daily_df[TARGET] = daily_df[TARGET].fillna(0)
    
    # Add external features
    for col in EXOG_FEATURES:
        daily_values = df.groupby(DATE_COL)[col].first()
        daily_df[col] = daily_values
        daily_df[col] = daily_df[col].ffill().bfill()
    
    # Create enhanced features
    daily_features = create_features(daily_df)
    
    # Split into train and validation (use 2023 as validation)
    train_df = daily_features[daily_features.index.year < 2023].copy()
    valid_df = daily_features[daily_features.index.year == 2023].copy()
    
    # Calculate historical upper bounds for predictions
    historical_max = train_df[TARGET].max()
    rolling_max = train_df[TARGET].rolling(window=30).max().max()
    
    # Prepare feature columns
    feature_cols = [col for col in train_df.columns 
                   if col != TARGET and col != DATE_COL and pd.api.types.is_numeric_dtype(train_df[col])]
    
    # Parameters optimized for prediction accuracy with regularization
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 16,
        'max_depth': 6,
        'learning_rate': 0.03,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'n_estimators': 1000,
        'lambda_l1': 0.3,
        'lambda_l2': 0.2,
        'min_data_in_leaf': 25,
        'min_gain_to_split': 0.2,
        'min_child_weight': 0.01
    }
    
    print("Training model...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train_df[feature_cols],
        train_df[TARGET],
        eval_set=[(valid_df[feature_cols], valid_df[TARGET])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # Make base predictions
    train_preds = model.predict(train_df[feature_cols])
    valid_preds = model.predict(valid_df[feature_cols])
    
    # Calculate errors for safety stock
    train_errors = train_df[TARGET].values - train_preds
    
    # Apply consolidated safety stock and bounds
    final_preds = []
    for i, pred in enumerate(valid_preds):
        # Get current conditions
        temp = valid_df['temp'].iloc[i]
        is_peak = valid_df['is_peak_season'].iloc[i] == 1
        is_holiday = valid_df['is_holiday'].iloc[i] == 1
        
        # Calculate safety stock
        safety_stock = calculate_safety_stock(train_errors, temp, is_peak, is_holiday)
        
        # Apply safety stock and bounds
        final_pred = pred + safety_stock
        final_pred = min(final_pred, rolling_max * 1.2)  # Allow 20% above historical rolling max
        final_pred = max(final_pred, 0)  # Ensure non-negative
        
        final_preds.append(final_pred)
    
    final_preds = np.array(final_preds)
    
    # Calculate metrics
    print("\nValidation Metrics (2023):")
    print(f"Mean Actual Orders: {valid_df[TARGET].mean():.2f}")
    print(f"Mean Predicted Orders: {final_preds.mean():.2f}")
    print(f"Median Actual Orders: {valid_df[TARGET].median():.2f}")
    print(f"Median Predicted Orders: {np.median(final_preds):.2f}")
    print(f"Max Actual Orders: {valid_df[TARGET].max():.2f}")
    print(f"Max Predicted Orders: {final_preds.max():.2f}")
    
    # Calculate percentage of overpredictions
    over_pred_pct = (final_preds > valid_df[TARGET].values).mean() * 100
    print(f"\nPercentage of Overpredictions: {over_pred_pct:.1f}%")
    
    # Save model
    print("\nSaving model...")
    joblib.dump(model, 'business_optimized_model.joblib')
    
    return model

if __name__ == "__main__":
    train_model() 