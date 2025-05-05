import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib

# Column settings
ID_COL = 'SKU'               # Entity identifier for panel
DATE_COL = 'Date'            # Date column
TARGET = 'Quantity'          # Forecast target

# Cost dictionary
PRODUCT_COSTS = {
    "Case Chlorine": {"over": 1.78, "under": 26.10}
}

def business_loss(preds, actuals, products):
    """Calculate business-aware loss per prediction."""
    total_loss = 0
    n = len(preds)

    for pred, actual, product in zip(preds, actuals, products):
        error_over = max(0, pred - actual)
        error_under = max(0, actual - pred)
        costs = PRODUCT_COSTS.get(product, {"over": 1.0, "under": 1.0})
        total_loss += (costs["over"] * error_over + costs["under"] * error_under)

    return total_loss / n

def load_and_aggregate_weekly(path: str) -> pd.DataFrame:
    """Load data and aggregate to weekly level for Case Chlorine."""
    # Load raw data
    df = pd.read_excel(path, parse_dates=[DATE_COL])
    
    # Filter for Case Chlorine (SKU 102.0)
    df = df[df[ID_COL] == 102.0]
    
    # Set date as index for resampling
    df = df.set_index(DATE_COL)
    
    # Resample to weekly frequency, summing orders
    weekly_df = df.resample('W').agg({
        TARGET: 'sum',
        'tempmax': 'mean',
        'tempmin': 'mean',
        'temp': 'mean',
        'humidity': 'mean',
        'precip': 'sum',
        'windspeedmean': 'mean',
        'Fuel_Price': 'mean',
        'Avg_Price': 'mean'
    }).reset_index()
    
    # Add SKU column back
    weekly_df[ID_COL] = 102.0
    
    return weekly_df

def create_weekly_features(df: pd.DataFrame, lags: list = [1,4,12], rolling_windows: list = [4,12]):
    """Create features for weekly data."""
    df = df.copy()
    
    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df[TARGET].shift(lag)
    
    # Rolling window stats
    for window in rolling_windows:
        df[f'rmean_{window}'] = df[TARGET].shift(1).rolling(window).mean()
        df[f'rstd_{window}'] = df[TARGET].shift(1).rolling(window).std()
    
    # Seasonal features
    df['week_of_year'] = df[DATE_COL].dt.isocalendar().week
    df['month'] = df[DATE_COL].dt.month
    df['quarter'] = df[DATE_COL].dt.quarter
    
    # Drop NaNs from new features
    df = df.dropna()
    
    # Feature list
    FEATURES = [c for c in df.columns if c not in [ID_COL, DATE_COL, TARGET]]
    
    return df, FEATURES

def train_test_split_weekly(df: pd.DataFrame, test_year: int = 2023):
    """Split weekly data based on year boundary."""
    train_mask = df[DATE_COL].dt.year < test_year
    test_mask = df[DATE_COL].dt.year == test_year
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"\nTrain period: {train_df[DATE_COL].min()} to {train_df[DATE_COL].max()}")
    print(f"Test period: {test_df[DATE_COL].min()} to {test_df[DATE_COL].max()}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    return train_df, test_df

def evaluate_weekly_predictions(model, test_df: pd.DataFrame, FEATURES: list):
    """Evaluate weekly predictions and create visualizations."""
    preds = model.predict(test_df[FEATURES])
    y_true = test_df[TARGET]
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    
    # Calculate business loss
    business_cost = business_loss(preds, y_true, ["Case Chlorine"] * len(preds))
    
    print(f"\nWeekly Analysis Results:")
    print(f"MAE: {mae:.3f} orders per week")
    print(f"RMSE: {rmse:.3f} orders per week")
    print(f"Business Loss: ${business_cost:.2f} per prediction")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Weekly Orders Time Series
    plt.subplot(2, 1, 1)
    plt.plot(test_df[DATE_COL], y_true, label='Actual Orders', color='blue', alpha=0.7)
    plt.plot(test_df[DATE_COL], preds, label='Predicted Orders', color='red', alpha=0.7)
    plt.title('Case Chlorine: Weekly Actual vs Predicted Orders')
    plt.xlabel('Date')
    plt.ylabel('Number of Orders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Error Distribution
    plt.subplot(2, 1, 2)
    errors = preds - y_true
    plt.hist(errors, bins=20, alpha=0.7, color='green')
    plt.title('Distribution of Weekly Prediction Errors')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('weekly_case_chlorine_analysis.png')
    plt.close()
    
    return preds

if __name__ == '__main__':
    print("Starting weekly analysis for Case Chlorine...")
    
    # Load and aggregate data
    print("\n1. Loading and aggregating data to weekly level...")
    df = load_and_aggregate_weekly('final_total_clean.xlsx')
    print(f"Total weeks of data: {len(df)}")
    print(f"Average weekly orders: {df[TARGET].mean():.2f}")
    
    # Create features
    print("\n2. Creating features...")
    df_features, FEATURES = create_weekly_features(df)
    print(f"Created {len(FEATURES)} features")
    
    # Split data
    print("\n3. Splitting data...")
    train_df, test_df = train_test_split_weekly(df_features)
    
    # Define model parameters
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'num_leaves': [15, 31],
        'min_child_samples': [20, 50]
    }
    
    # Train model
    print("\n4. Training model...")
    model = lgb.LGBMRegressor(objective='regression')
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(train_df[FEATURES], train_df[TARGET])
    print("Best parameters:", grid_search.best_params_)
    
    # Evaluate predictions
    print("\n5. Evaluating predictions...")
    predictions = evaluate_weekly_predictions(grid_search.best_estimator_, test_df, FEATURES)
    
    # Save model
    print("\n6. Saving model...")
    joblib.dump(grid_search.best_estimator_, 'weekly_case_chlorine_model.joblib')
    print("Model saved as 'weekly_case_chlorine_model.joblib'") 