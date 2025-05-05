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

def load_and_aggregate_three_day(path: str) -> pd.DataFrame:
    """Load data and aggregate to 3-day level for Case Chlorine."""
    # Load raw data
    df = pd.read_excel(path, parse_dates=[DATE_COL])
    
    # Filter for Case Chlorine (SKU 102.0)
    df = df[df[ID_COL] == 102.0]
    
    # Set date as index for resampling
    df = df.set_index(DATE_COL)
    
    # Resample to 3-day frequency, summing orders
    three_day_df = df.resample('3D').agg({
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
    three_day_df[ID_COL] = 102.0
    
    return three_day_df

def create_features(df: pd.DataFrame, lags: list = [1,2,4,8], rolling_windows: list = [3,6,12]):
    """Create features for 3-day data."""
    df = df.copy()
    
    # Lag features (adjusted for 3-day frequency)
    for lag in lags:
        df[f'lag_{lag}'] = df[TARGET].shift(lag)
    
    # Rolling window stats (adjusted for 3-day frequency)
    for window in rolling_windows:
        df[f'rmean_{window}'] = df[TARGET].shift(1).rolling(window).mean()
        df[f'rstd_{window}'] = df[TARGET].shift(1).rolling(window).std()
    
    # Seasonal features
    df['day_of_week'] = df[DATE_COL].dt.dayofweek
    df['week_of_year'] = df[DATE_COL].dt.isocalendar().week
    df['month'] = df[DATE_COL].dt.month
    df['quarter'] = df[DATE_COL].dt.quarter
    
    # Drop NaNs from new features
    df = df.dropna()
    
    # Feature list
    FEATURES = [c for c in df.columns if c not in [ID_COL, DATE_COL, TARGET]]
    
    return df, FEATURES

def train_test_split_time(df: pd.DataFrame, test_year: int = 2023):
    """Split 3-day data based on year boundary."""
    train_mask = df[DATE_COL].dt.year < test_year
    test_mask = df[DATE_COL].dt.year == test_year
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"\nTrain period: {train_df[DATE_COL].min()} to {train_df[DATE_COL].max()}")
    print(f"Test period: {test_df[DATE_COL].min()} to {test_df[DATE_COL].max()}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    return train_df, test_df

def evaluate_predictions(model, test_df: pd.DataFrame, FEATURES: list):
    """Evaluate 3-day predictions and create visualizations."""
    preds = model.predict(test_df[FEATURES])
    y_true = test_df[TARGET]
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    
    # Calculate business loss
    business_cost = business_loss(preds, y_true, ["Case Chlorine"] * len(preds))
    
    print(f"\n3-Day Analysis Results:")
    print(f"MAE: {mae:.3f} orders per 3-day period")
    print(f"RMSE: {rmse:.3f} orders per 3-day period")
    print(f"Business Loss: ${business_cost:.2f} per prediction")
    
    # Create visualizations
    plt.figure(figsize=(15, 15))
    
    # 1. Orders Time Series
    plt.subplot(3, 1, 1)
    plt.plot(test_df[DATE_COL], y_true, label='Actual Orders', color='blue', alpha=0.7)
    plt.plot(test_df[DATE_COL], preds, label='Predicted Orders', color='red', alpha=0.7)
    plt.title('Case Chlorine: 3-Day Actual vs Predicted Orders')
    plt.xlabel('Date')
    plt.ylabel('Number of Orders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Error Distribution
    plt.subplot(3, 1, 2)
    errors = preds - y_true
    plt.hist(errors, bins=20, alpha=0.7, color='green')
    plt.title('Distribution of 3-Day Prediction Errors')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. Feature Importance
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.subplot(3, 1, 3)
    sns.barplot(data=importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Feature Importance')
    
    plt.tight_layout()
    plt.savefig('three_day_case_chlorine_analysis.png')
    plt.close()
    
    return preds, importance

if __name__ == '__main__':
    print("Starting 3-day analysis for Case Chlorine...")
    
    # Load and aggregate data
    print("\n1. Loading and aggregating data to 3-day level...")
    df = load_and_aggregate_three_day('final_total_clean.xlsx')
    print(f"Total 3-day periods: {len(df)}")
    print(f"Average orders per 3-day period: {df[TARGET].mean():.2f}")
    
    # Create features
    print("\n2. Creating features...")
    df_features, FEATURES = create_features(df)
    print(f"Created {len(FEATURES)} features")
    
    # Split data
    print("\n3. Splitting data...")
    train_df, test_df = train_test_split_time(df_features)
    
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
    predictions, feature_importance = evaluate_predictions(grid_search.best_estimator_, test_df, FEATURES)
    
    # Print top features
    print("\nTop 5 most important features:")
    print(feature_importance.head().to_string())
    
    # Save model
    print("\n6. Saving model...")
    joblib.dump(grid_search.best_estimator_, 'three_day_case_chlorine_model.joblib')
    print("Model saved as 'three_day_case_chlorine_model.joblib'") 