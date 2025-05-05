import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import importlib.util
import sys

# Import loss function from file with space in name
spec = importlib.util.spec_from_file_location("loss_function", "loss function.py")
loss_module = importlib.util.module_from_spec(spec)
sys.modules["loss_function"] = loss_module
spec.loader.exec_module(loss_module)

# Constants
ID_COL = 'SKU'
DATE_COL = 'Date'
TARGET = 'orders'
POOL_GUY_CHLORINE_SKU = 1.0

# Match the exact features from the training model
EXOG_FEATURES = [
    'tempmax', 'tempmin', 'temp', 'humidity',
    'precip', 'windspeedmean', 'Fuel_Price',
    'Avg_Price'
]

def create_features(df: pd.DataFrame):
    """Create time series features matching the original training."""
    df = df.copy()
    
    # Lag features
    for lag in [1, 7, 14]:
        df[f'lag_{lag}'] = df['Quantity'].shift(lag)
    
    # Rolling window stats
    for window in [7, 14]:
        df[f'rmean_{window}'] = df['Quantity'].shift(1).rolling(window).mean()
        df[f'rstd_{window}'] = df['Quantity'].shift(1).rolling(window).std()
    
    return df

def generate_2024_dates():
    """Generate dates for 2024 predictions."""
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return dates

def main():
    print("Loading historical data...")
    df = pd.read_excel('final_total_clean.xlsx', parse_dates=[DATE_COL])
    
    # Filter for Pool Guy Chlorine
    df = df[df[ID_COL] == POOL_GUY_CHLORINE_SKU]
    
    # Create complete date range including all days
    min_date = df[DATE_COL].min()
    max_date = datetime(2023, 12, 31)  # Use end of 2023 as cutoff
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Create DataFrame with all dates and aggregate quantities
    daily_df = pd.DataFrame(index=all_dates)
    daily_df.index.name = DATE_COL
    quantities_by_date = df.groupby(DATE_COL)['Quantity'].sum()
    daily_df = daily_df.join(quantities_by_date)
    daily_df['Quantity'] = daily_df['Quantity'].fillna(0)
    
    # Add weather and price features (using last known values from historical data)
    for col in EXOG_FEATURES:
        daily_values = df.groupby(DATE_COL)[col].first()
        daily_df[col] = daily_values
        # Forward fill and then backward fill to handle missing values
        daily_df[col] = daily_df[col].ffill().bfill()
    
    # Create features for historical data
    daily_features = create_features(daily_df)
    
    # Generate 2024 dates
    dates_2024 = generate_2024_dates()
    
    # Create DataFrame for 2024 with initial values from end of 2023
    df_2024 = pd.DataFrame(index=dates_2024)
    df_2024.index.name = DATE_COL
    
    # Add weather and price features for 2024 (using last known values)
    for col in EXOG_FEATURES:
        df_2024[col] = daily_df[col].iloc[-1]
    
    # Initialize 2024 quantities with seasonal patterns from 2023
    last_year_pattern = daily_df.loc['2023'].groupby(lambda x: x.strftime('%m-%d'))['Quantity'].mean()
    df_2024['Quantity'] = df_2024.index.strftime('%m-%d').map(last_year_pattern)
    
    # Combine historical and 2024 data for proper feature creation
    df_combined = pd.concat([daily_df, df_2024])
    df_combined = df_combined.sort_index()
    
    # Create features for combined data
    df_features = create_features(df_combined)
    
    # Get 2024 features
    df_2024_features = df_features.loc[dates_2024]
    
    print("Loading the trained model...")
    model = joblib.load('best_model.joblib')
    
    # Get exact feature columns from the model
    print("Model feature names:", model.feature_name_)
    feature_cols = model.feature_name_
    
    # Debug: Print feature statistics
    print("\nFeature Statistics:")
    for col in feature_cols:
        if col not in df_2024_features.columns:
            print(f"Missing feature: {col}")
            df_2024_features[col] = 0  # Add missing feature with default value
        print(f"{col}:")
        print(f"  Mean: {df_2024_features[col].mean():.2f}")
        print(f"  Std: {df_2024_features[col].std():.2f}")
        print(f"  Min: {df_2024_features[col].min():.2f}")
        print(f"  Max: {df_2024_features[col].max():.2f}")
    
    print("\nGenerating predictions for 2024...")
    predictions = model.predict(df_2024_features[feature_cols])
    
    # Calculate historical statistics by month
    monthly_stats = daily_df.groupby(daily_df.index.month)['Quantity'].agg([
        'mean', 'std', 'min', 'max'
    ]).round(2)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Create subplot for daily predictions
    plt.subplot(2, 1, 1)
    plt.plot(dates_2024, predictions, label='Predicted Orders', color='blue', alpha=0.7)
    plt.title('Pool Guy Chlorine: Predicted Daily Orders for 2024')
    plt.xlabel('Date')
    plt.ylabel('Number of Orders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Create subplot for monthly statistics
    plt.subplot(2, 1, 2)
    monthly_stats['mean'].plot(kind='bar', yerr=monthly_stats['std'], 
                              capsize=5, color='skyblue', 
                              alpha=0.7, label='Monthly Mean ± Std')
    plt.title('Historical Monthly Order Patterns')
    plt.xlabel('Month')
    plt.ylabel('Number of Orders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Saving visualization...")
    plt.savefig('pool_guy_chlorine_2024_forecast.png')
    plt.close()
    
    # Print summary statistics
    print("\nForecast Summary:")
    print(f"Average predicted orders: {predictions.mean():.2f}")
    print(f"Maximum predicted orders: {predictions.max():.2f}")
    print(f"Minimum predicted orders: {predictions.min():.2f}")
    
    print("\nHistorical Monthly Statistics:")
    print(monthly_stats)
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Date': dates_2024,
        'Predicted_Quantity': predictions
    })
    predictions_df.to_csv('pool_guy_chlorine_2024_predictions.csv', index=False)
    print("\nPredictions saved to 'pool_guy_chlorine_2024_predictions.csv'")

if __name__ == "__main__":
    main() 