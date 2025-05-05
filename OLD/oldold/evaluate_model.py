import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime
import holidays
from typing import Tuple, Dict
import os
from business_optimized_model import create_features  # Import the feature creation function

# Constants
ID_COL = 'SKU'
DATE_COL = 'Date'
TARGET = 'Quantity'
POOL_GUY_CHLORINE_SKU = 1.0

def is_holiday(date):
    """Check if date is a US holiday."""
    us_holidays = holidays.US()
    return date in us_holidays

def is_peak_season(month):
    """Define peak pool season (May through September)."""
    return 5 <= month <= 9

def get_temperature_band(temp: float) -> str:
    """Categorize temperature into bands."""
    if temp <= 60:
        return "Cold (≤60°F)"
    elif temp <= 74:
        return "Normal (61-74°F)"
    elif temp <= 84:
        return "Hot (75-84°F)"
    else:
        return "Very Hot (≥85°F)"

def calculate_business_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate business loss based on company-specified costs."""
    total_cost = 0
    for true, pred in zip(y_true, y_pred):
        if pred > true:  # Over-prediction
            total_cost += 1.78 * (pred - true)  # Holding cost
        else:  # Under-prediction
            total_cost += 11.54 * (true - pred)  # Shortage cost
    return total_cost / len(y_true)

def evaluate_by_temperature_band(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Evaluate model performance across temperature bands."""
    results = {}
    for band in ["Cold (≤60°F)", "Normal (61-74°F)", "Hot (75-84°F)", "Very Hot (≥85°F)"]:
        mask = df['temp_band'] == band
        if mask.any():
            results[band] = {
                'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                'business_loss': calculate_business_loss(y_true[mask], y_pred[mask]),
                'over_predictions': (y_pred[mask] > y_true[mask]).mean() * 100,
                'under_predictions': (y_pred[mask] < y_true[mask]).mean() * 100,
                'avg_safety_stock': np.mean(y_pred[mask] - y_true[mask])
            }
    return results

def evaluate_by_season(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Evaluate model performance by season."""
    results = {}
    for season in ['Peak', 'Off-Peak']:
        mask = df['is_peak_season'] == (1 if season == 'Peak' else 0)
        if mask.any():
            results[season] = {
                'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                'business_loss': calculate_business_loss(y_true[mask], y_pred[mask])
            }
    return results

def evaluate_by_holiday(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Evaluate model performance around holidays."""
    results = {}
    for period in ['Pre-Holiday', 'Holiday', 'Post-Holiday', 'Normal']:
        if period == 'Holiday':
            mask = df['is_holiday'] == 1
        elif period == 'Pre-Holiday':
            mask = (df['days_to_holiday'] <= 3) & (df['days_to_holiday'] > 0)
        elif period == 'Post-Holiday':
            mask = (df['days_after_holiday'] <= 3) & (df['days_after_holiday'] > 0)
        else:
            mask = (df['is_holiday'] == 0) & (df['days_to_holiday'] > 3) & (df['days_after_holiday'] > 3)
        
        if mask.any():
            results[period] = {
                'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                'rmse': np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                'business_loss': calculate_business_loss(y_true[mask], y_pred[mask])
            }
    return results

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot the distribution of prediction errors."""
    plt.figure(figsize=(10, 6))
    errors = y_pred - y_true
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.close()

def plot_time_series_comparison(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot actual vs predicted values over time."""
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, y_true, label='Actual', alpha=0.7)
    plt.plot(df.index, y_pred, label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Orders Over Time')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot actual vs predicted scatter plot with perfect prediction line."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Actual vs Predicted Orders')
    plt.xlabel('Actual Orders')
    plt.ylabel('Predicted Orders')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Create output directory for visualizations
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Load data and model
    print("Loading data and model...")
    df = pd.read_excel('final_total_clean.xlsx', parse_dates=[DATE_COL])
    df = df[df[ID_COL] == POOL_GUY_CHLORINE_SKU]
    
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
    for col in ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'windspeedmean', 'Fuel_Price', 'Avg_Price']:
        daily_values = df.groupby(DATE_COL)[col].first()
        daily_df[col] = daily_values
        daily_df[col] = daily_df[col].ffill().bfill()
    
    # Create all features using the same function as training
    print("Creating features...")
    daily_features = create_features(daily_df)
    
    # Load the trained model
    model = joblib.load('business_optimized_model.joblib')
    
    # Prepare features for prediction
    feature_cols = [col for col in daily_features.columns 
                   if col != TARGET and col != DATE_COL and pd.api.types.is_numeric_dtype(daily_features[col])]
    
    # Get predictions
    y_pred = model.predict(daily_features[feature_cols])
    y_true = daily_features[TARGET].values
    
    # Add temperature band for evaluation
    daily_features['temp_band'] = daily_features['temp'].apply(get_temperature_band)
    
    # Calculate overall metrics
    print("\nOverall Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"Root Mean Square Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"Business Loss: ${calculate_business_loss(y_true, y_pred):.2f}")
    
    # Evaluate by temperature band
    print("\nTemperature Band Performance:")
    temp_results = evaluate_by_temperature_band(daily_features, y_true, y_pred)
    for band, metrics in temp_results.items():
        print(f"\n{band}:")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  Business Loss: ${metrics['business_loss']:.2f}")
        print(f"  Over-predictions: {metrics['over_predictions']:.1f}%")
        print(f"  Under-predictions: {metrics['under_predictions']:.1f}%")
        print(f"  Avg Safety Stock: {metrics['avg_safety_stock']:.2f}")
    
    # Evaluate by season
    print("\nSeasonal Performance:")
    season_results = evaluate_by_season(daily_features, y_true, y_pred)
    for season, metrics in season_results.items():
        print(f"\n{season} Season:")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  Business Loss: ${metrics['business_loss']:.2f}")
    
    # Evaluate by holiday
    print("\nHoliday Performance:")
    holiday_results = evaluate_by_holiday(daily_features, y_true, y_pred)
    for period, metrics in holiday_results.items():
        print(f"\n{period}:")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  Business Loss: ${metrics['business_loss']:.2f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_error_distribution(y_true, y_pred, 'evaluation_results/prediction_error_distribution_2023.png')
    plot_time_series_comparison(daily_features, y_true, y_pred, 'evaluation_results/time_series_comparison_2023.png')
    plot_actual_vs_predicted(y_true, y_pred, 'evaluation_results/actual_vs_predicted_2023.png')
    
    print("\nEvaluation complete! Results and visualizations saved in 'evaluation_results' directory.")

if __name__ == "__main__":
    main() 