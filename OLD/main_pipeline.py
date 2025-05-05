import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import lightgbm as lgb
import joblib
import importlib.util
import sys

# Import loss function from file with space in name
spec = importlib.util.spec_from_file_location("loss_function", "loss function.py")
loss_module = importlib.util.module_from_spec(spec)
sys.modules["loss_function"] = loss_module
spec.loader.exec_module(loss_module)
business_loss = loss_module.business_loss

# Column settings
ID_COL = 'SKU'               # Entity identifier for panel
DATE_COL = 'Date'            # Date column
TARGET = 'orders'            # Forecast target

# SKU to product name mapping
SKU_TO_PRODUCT = {
    1.0: "Pool Guy Chlorine",
    102.0: "Case Chlorine",
    283.0: "Pool Guy Acid",
    332.0: "Case Acid",
    487.0: "Salt",
    486.0: "7-in-1 Test Strips",
    2314.0: "Renew"
}

# Cost dictionary based on your finalized table
PRODUCT_COSTS = {
    "Pool Guy Chlorine": {"over": 1.78, "under": 11.54},
    "Case Chlorine": {"over": 1.78, "under": 26.10},
    "Pool Guy Acid": {"over": 1.87, "under": 10.73},
    "Case Acid": {"over": 1.87, "under": 26.73},
    "Salt": {"over": 2.27, "under": 17.43},
    "7-in-1 Test Strips": {"over": 0.96, "under": 27.50},
    "Renew": {"over": 1.22, "under": 29.44},
    "Cal Hypo Shock": {"over": 1.11, "under": 12.26},
    "Phosphate Remover": {"over": 1.30, "under": 33.04},
    "Algaecide": {"over": 1.27, "under": 30.57}
}

# Exogenous features to carry through from raw data
EXOG_FEATURES = [
    'tempmax', 'tempmin', 'temp', 'humidity',
    'precip', 'windspeedmean', 'Fuel_Price',
    'Avg_Price'  # Removing 'conditions' and 'description' as they are text
]

def analyze_sparsity(df: pd.DataFrame, sku: str = None):
    """
    Analyze sparsity of order data for a specific SKU or all SKUs
    """
    if sku:
        df = df[df[ID_COL] == sku]
    
    # Calculate sparsity metrics
    total_days = len(df)
    zero_days = (df[TARGET] == 0).sum()
    sparsity = zero_days / total_days
    
    print(f"\nSparsity Analysis for SKU {str(sku) if sku else 'all'}:")
    print(f"Total days: {total_days}")
    print(f"Days with zero orders: {zero_days}")
    print(f"Sparsity ratio: {sparsity:.2%}")
    
    # Plot distribution of orders and save to file
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=TARGET, bins=30)
    plt.title(f"Distribution of Daily Orders {'for SKU ' + str(sku) if sku else ''}")
    plt.savefig(f'sparsity_histogram_{sku if sku else "all"}.png')
    plt.close()
    
    return sparsity

def analyze_data_gaps(df: pd.DataFrame, sku: float = None):
    """
    Analyze gaps in the time series data
    """
    if sku:
        df = df[df[ID_COL] == sku]
    
    # Convert to datetime if not already
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    # Sort by date
    df = df.sort_values(DATE_COL)
    
    # Calculate the difference between consecutive dates
    date_diffs = df[DATE_COL].diff()
    
    # Find gaps larger than 1 day
    gaps = df[date_diffs > pd.Timedelta(days=1)]
    
    if len(gaps) > 0:
        print(f"\nFound {len(gaps)} gaps in data for SKU {sku if sku else 'all'}:")
        for idx, row in gaps.iterrows():
            gap_start = df.loc[idx-1, DATE_COL] if idx > 0 else df[DATE_COL].min()
            gap_end = row[DATE_COL]
            gap_days = (gap_end - gap_start).days
            print(f"Gap from {gap_start.date()} to {gap_end.date()} ({gap_days} days)")
    else:
        print(f"\nNo gaps found in data for SKU {sku if sku else 'all'}")
    
    # Print overall date range
    print(f"Data range: {df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")

def get_top_products(df: pd.DataFrame) -> list:
    """
    Analyze specific list of products
    """
    # Predefined list of SKUs we want to analyze (removed 4943)
    target_skus = [1.0, 102.0, 283.0, 332.0, 487.0, 486.0, 2314.0]
    
    # Calculate statistics for these specific SKUs
    product_stats = df[df[ID_COL].isin(target_skus)].groupby(ID_COL).agg({
        TARGET: ['sum', 'mean', 'count'],
        'Avg_Price': 'mean'
    }).round(2)
    
    # Flatten column names
    product_stats.columns = ['total_orders', 'avg_daily_orders', 'days_with_orders', 'avg_price']
    
    # Sort by SKU number to maintain the order
    product_stats = product_stats.reindex(target_skus)
    
    # Save to CSV
    product_stats.to_csv('selected_products.csv')
    
    print("\nAnalysis of selected products:")
    for idx, (sku, row) in enumerate(product_stats.iterrows(), 1):
        print(f"{idx}. SKU: {sku}")
        print(f"   Total Orders: {row['total_orders']:.0f}")
        print(f"   Avg Daily Orders: {row['avg_daily_orders']:.2f}")
        print(f"   Days with Orders: {row['days_with_orders']}")
        print(f"   Average Price: ${row['avg_price']:.2f}")
        # Analyze gaps for this SKU
        analyze_data_gaps(df, sku)
        print()
    
    return target_skus

def load_and_analyze_data(path: str, n_products: int = 10) -> pd.DataFrame:
    """
    Load data and select top N products for analysis
    """
    print("\nLoading and analyzing data...")
    # Load raw transactional data
    df = pd.read_excel(path, parse_dates=[DATE_COL])
    
    # Aggregate to daily level
    aggs = { 'Transaction_ID': 'nunique' }
    for col in EXOG_FEATURES:
        aggs[col] = 'first'
    
    panel = (
        df.groupby([ID_COL, DATE_COL])
          .agg(aggs)
          .reset_index()
          .rename(columns={'Transaction_ID': TARGET})
    )
    panel = panel.sort_values([ID_COL, DATE_COL])
    
    # Get top N products
    top_products = get_top_products(panel)
    
    # Filter for top products
    panel_filtered = panel[panel[ID_COL].isin(top_products)]
    
    # Analyze sparsity for each top product
    print("\nAnalyzing sparsity for top products:")
    for sku in top_products:
        analyze_sparsity(panel_filtered, sku)
    
    return panel_filtered

def business_loss(preds, actuals, products):
    """
    Calculate business-aware loss per prediction.
    """
    total_loss = 0
    n = len(preds)

    for pred, actual, product in zip(preds, actuals, products):
        error_over = max(0, pred - actual)
        error_under = max(0, actual - pred)
        costs = PRODUCT_COSTS.get(product, {"over": 1.0, "under": 1.0})  # fallback to neutral
        total_loss += (costs["over"] * error_over + costs["under"] * error_under)

    return total_loss / n

def custom_cost_score(y_true, y_pred, X=None):
    """
    Custom cost-based scoring function using business loss
    """
    # Get SKUs from X if available, otherwise assume order matches SKU_TO_PRODUCT
    if X is not None and ID_COL in X.columns:
        skus = X[ID_COL].values
    else:
        skus = list(SKU_TO_PRODUCT.keys())[:len(y_true)]
    
    # Map SKUs to product names
    products = [SKU_TO_PRODUCT.get(sku, "Pool Guy Chlorine") for sku in skus]  # Default to Pool Guy Chlorine if SKU not found
    
    # Calculate business loss
    loss = business_loss(y_pred, y_true, products)
    return -loss  # Negative because GridSearchCV maximizes score

custom_scorer = make_scorer(custom_cost_score, greater_is_better=True, needs_proba=False, needs_threshold=False)

def load_and_aggregate(path: str) -> pd.DataFrame:
    # Load raw transactional data
    df = pd.read_excel(path, parse_dates=[DATE_COL])
    # Aggregate to panel time series: count unique transactions per SKU per day
    aggs = { 'Transaction_ID': 'nunique' }
    # include exogenous first-values
    for col in EXOG_FEATURES:
        aggs[col] = 'first'
    panel = (
        df.groupby([ID_COL, DATE_COL])
          .agg(aggs)
          .reset_index()
          .rename(columns={'Transaction_ID': TARGET})
    )
    panel = panel.sort_values([ID_COL, DATE_COL])
    return panel

def explore_data(df: pd.DataFrame):
    print(df.head())
    print(df.info())
    print(df.describe())
    # Plot a sample series
    sample = df[ID_COL].unique()[0]
    sns.lineplot(data=df[df[ID_COL] == sample], x=DATE_COL, y=TARGET)
    plt.title(f"{TARGET} over time for SKU {sample}")
    plt.show()

def preprocess(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    # Ensure continuous date index per SKU
    df = df.set_index(DATE_COL)
    out = []
    for sku, grp in df.groupby(ID_COL):
        grp = grp.resample(freq).asfreq()
        # Interpolate missing target
        grp[TARGET] = grp[TARGET].interpolate(method='time')
        # Forward-fill exogenous
        grp[EXOG_FEATURES] = grp[EXOG_FEATURES].ffill()
        grp[ID_COL] = sku
        out.append(grp)
    result = pd.concat(out).reset_index()
    return result

def create_features(df: pd.DataFrame, lags: list = [1,7,14], rolling_windows: list = [7,14]):
    df = df.copy()
    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(ID_COL)[TARGET].shift(lag)
    # Rolling window stats
    for window in rolling_windows:
        grp = df.groupby(ID_COL)[TARGET]
        df[f'rmean_{window}'] = grp.shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        df[f'rstd_{window}'] = grp.shift(1).rolling(window).std().reset_index(level=0, drop=True)
    # Drop NaNs from new features
    df = df.dropna()
    # Feature list
    FEATURES = [c for c in df.columns if c not in [ID_COL, DATE_COL, TARGET]]
    return df, FEATURES

def train_test_split_time(df: pd.DataFrame, test_year: int = 2023):
    """
    Split data based on year boundary.
    All data before test_year goes to training, test_year goes to test.
    """
    train_mask = df[DATE_COL].dt.year < test_year
    test_mask = df[DATE_COL].dt.year == test_year  # Changed to only use 2023 as test
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"\nTrain period: {train_df[DATE_COL].min()} to {train_df[DATE_COL].max()}")
    print(f"Test period: {test_df[DATE_COL].min()} to {test_df[DATE_COL].max()}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    return train_df, test_df

def run_cv(train_df: pd.DataFrame, FEATURES: list, param_grid: dict,
           cv_splits: int = 5, randomized: bool = True, n_iter: int = 20):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    model = lgb.LGBMRegressor(objective='regression')
    if randomized:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring=custom_scorer,
            verbose=1,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring=custom_scorer,
            verbose=1,
            n_jobs=-1
        )
    search.fit(train_df[FEATURES], train_df[TARGET])
    print("Best params:", search.best_params_)
    print("Best score:", search.best_score_)
    return search

def forecast_and_evaluate(model, test_df: pd.DataFrame, FEATURES: list):
    preds = model.predict(test_df[FEATURES])
    y_true = test_df[TARGET]
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    
    # Calculate business loss
    products = [SKU_TO_PRODUCT.get(sku, "Pool Guy Chlorine") for sku in test_df[ID_COL]]
    business_cost = business_loss(preds, y_true, products)
    
    print(f"Test MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    print(f"Business Loss: ${business_cost:.2f} per prediction")
    
    # Add per-SKU analysis
    print("\nPer-SKU Analysis:")
    print("-" * 50)
    for sku in test_df[ID_COL].unique():
        sku_mask = test_df[ID_COL] == sku
        sku_actual = y_true[sku_mask]
        sku_pred = preds[sku_mask]
        sku_dates = test_df[DATE_COL][sku_mask]
        
        avg_actual = sku_actual.mean()
        avg_pred = sku_pred.mean()
        sku_mae = mean_absolute_error(sku_actual, sku_pred)
        
        # Calculate SKU-specific business loss
        product_name = SKU_TO_PRODUCT.get(sku, "Pool Guy Chlorine")
        sku_business_loss = business_loss(sku_pred, sku_actual, [product_name] * len(sku_pred))
        
        print(f"SKU {sku} ({product_name}):")
        print(f"  Average Actual Orders: {avg_actual:.2f}")
        print(f"  Average Predicted Orders: {avg_pred:.2f}")
        print(f"  SKU-specific MAE: {sku_mae:.2f}")
        print(f"  Error as % of Average: {(sku_mae/avg_actual*100):.1f}%")
        print(f"  Business Loss: ${sku_business_loss:.2f} per prediction")
        print("-" * 50)
        
        # Create detailed visualizations for Case Chlorine (SKU 102.0)
        if sku == 102.0:
            # 1. Prediction vs Actual Time Series
            plt.figure(figsize=(15, 6))
            plt.plot(sku_dates, sku_actual, label='Actual Orders', color='blue', alpha=0.7)
            plt.plot(sku_dates, sku_pred, label='Predicted Orders', color='red', alpha=0.7)
            plt.title(f'Case Chlorine (SKU 102.0): Actual vs Predicted Orders\nMAE: {sku_mae:.2f}, Business Loss: ${sku_business_loss:.2f}/prediction')
            plt.xlabel('Date')
            plt.ylabel('Number of Orders')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('case_chlorine_predictions.png')
            plt.close()
            
            # 2. Loss Function Curve
            plt.figure(figsize=(12, 6))
            errors = sku_pred - sku_actual
            costs = []
            for error in errors:
                if error > 0:  # Overstock
                    cost = error * PRODUCT_COSTS["Case Chlorine"]["over"]
                else:  # Understock
                    cost = -error * PRODUCT_COSTS["Case Chlorine"]["under"]
                costs.append(cost)
            
            # Plot error distribution and costs
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Error distribution
            ax1.hist(errors, bins=20, alpha=0.7, color='blue')
            ax1.set_title('Distribution of Prediction Errors\nfor Case Chlorine')
            ax1.set_xlabel('Prediction Error (Predicted - Actual)')
            ax1.set_ylabel('Frequency')
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            # Cost distribution
            ax2.hist(costs, bins=20, alpha=0.7, color='green')
            ax2.set_title('Distribution of Business Costs\nfor Case Chlorine')
            ax2.set_xlabel('Cost ($)')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('case_chlorine_loss_analysis.png')
            plt.close()
    
    # Plot per SKU or aggregate and save to file
    plt.figure(figsize=(12,6))
    plt.plot(test_df[DATE_COL], y_true, label='Actual')
    plt.plot(test_df[DATE_COL], preds, label='Forecast')
    plt.title('Forecast vs Actual')
    plt.legend()
    plt.savefig('forecast_vs_actual.png')
    plt.close()
    
    return preds

if __name__ == '__main__':
    print("Starting data analysis pipeline...")
    
    # Load & analyze top products
    print("\n1. Loading and analyzing top products...")
    df_raw = load_and_analyze_data('final_total_clean.xlsx')
    print("Data loaded successfully. Shape:", df_raw.shape)
    print(f"Date range: {df_raw[DATE_COL].min()} to {df_raw[DATE_COL].max()}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    df_processed = preprocess(df_raw)
    print("Preprocessed shape:", df_processed.shape)
    
    # Create features
    print("\n3. Creating features...")
    df_features, FEATURES = create_features(df_processed)
    print("Features created. Total features:", len(FEATURES))
    print("Feature list:", FEATURES)
    
    # Split data
    print("\n4. Splitting data into train and test sets...")
    train_df, test_df = train_test_split_time(df_features, test_year=2023)
    
    # Define model parameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [7, 15, 31],
        'min_child_samples': [20, 50, 100]
    }
    
    # Train model with cross-validation
    print("\n5. Training model with cross-validation...")
    cv_results = run_cv(train_df, FEATURES, param_grid, cv_splits=5, randomized=True, n_iter=20)
    
    # Make predictions and evaluate
    print("\n6. Making predictions and evaluating...")
    predictions = forecast_and_evaluate(cv_results.best_estimator_, test_df, FEATURES)
    
    # Save the model
    print("\n7. Saving the best model...")
    joblib.dump(cv_results.best_estimator_, 'best_model.joblib')
    print("Model saved as 'best_model.joblib'") 