import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import lightgbm as lgb

# =========================
# 1. CONFIGURATION & UTILS
# =========================

# Column settings
ID_COL = 'SKU'               # Entity identifier for panel
DATE_COL = 'Date'            # Date column
TARGET = 'orders'            # Forecast target

# Exogenous features to carry through from raw data
EXOG_FEATURES = [
    'tempmax', 'tempmin', 'temp', 'humidity',
    'precip', 'windspeedmean', 'Fuel_Price',
    'Median_HH_Income', 'Unemployment_Rate'
]

# Custom scoring scaffold
def custom_cost_score(y_true, y_pred):
    """
    Custom cost-based scoring function (negative for maximization).
    Replace this logic with business-specific cost calculations.
    """
    cost = np.abs(y_true - y_pred)
    return -np.sum(cost)

custom_scorer = make_scorer(custom_cost_score, greater_is_better=True)

# =========================
# 2. DATA LOADING & AGGREGATION
# =========================

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

# =========================
# 3. DATA INSPECTION
# =========================

def explore_data(df: pd.DataFrame):
    print(df.head())
    print(df.info())
    print(df.describe())
    # Plot a sample series
    sample = df[ID_COL].unique()[0]
    sns.lineplot(data=df[df[ID_COL] == sample], x=DATE_COL, y=TARGET)
    plt.title(f"{TARGET} over time for SKU {sample}")
    plt.show()

# =========================
# 4. PREPROCESSING & RESAMPLING
# =========================

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

# =========================
# 5. FEATURE ENGINEERING
# =========================

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

# =========================
# 6. TRAIN/TEST SPLIT
# =========================

def train_test_split_time(df: pd.DataFrame, test_size: float = 0.2):
    train_parts, test_parts = [], []
    for sku, grp in df.groupby(ID_COL):
        n = len(grp)
        split = int(n * (1 - test_size))
        train_parts.append(grp.iloc[:split])
        test_parts.append(grp.iloc[split:])
    return pd.concat(train_parts), pd.concat(test_parts)

# =========================
# 7. MODELING & CV
# =========================

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

# =========================
# 8. FORECAST & EVALUATION
# =========================

def forecast_and_evaluate(model, test_df: pd.DataFrame, FEATURES: list):
    preds = model.predict(test_df[FEATURES])
    y_true = test_df[TARGET]
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    print(f"Test MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    # Plot per SKU or aggregate
    plt.figure(figsize=(12,6))
    plt.plot(test_df[DATE_COL], y_true, label='Actual')
    plt.plot(test_df[DATE_COL], preds, label='Forecast')
    plt.title('Forecast vs Actual')
    plt.legend(); plt.show()
    return preds

# =========================
# 9. MAIN PIPELINE
# =========================
if __name__ == '__main__':
    # Load & aggregate
    df_raw = load_and_aggregate('final_total_clean.xlsx')
    explore_data(df_raw)

    # Preprocess
    df_prep = preprocess(df_raw, freq='D')

    # Features
    df_feat, FEATURES = create_features(df_prep)

    # Split
    train_df, test_df = train_test_split_time(df_feat, test_size=0.2)

    # Hyperparameters
    param_grid = {
        'num_leaves': [31, 61, 127],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300, 500]
    }

    # CV & Fit
    cv_model = run_cv(train_df, FEATURES, param_grid, cv_splits=5)

    # Forecast & Evaluate
    preds = forecast_and_evaluate(cv_model, test_df, FEATURES)

    # Save results
    model_path = 'orders_forecast_model.joblib'
    import joblib; joblib.dump(cv_model, model_path)
    print(f"Model saved to {model_path}")
