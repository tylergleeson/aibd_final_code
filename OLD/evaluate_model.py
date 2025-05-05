import pandas as pd
import numpy as np
import joblib
from business_optimized_model import create_features, weighted_quantile_loss, EXOG_FEATURES, ID_COL, DATE_COL, TARGET
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load and prepare the data."""
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
    
    return daily_df

def prepare_features(df):
    """Prepare features for prediction, using same process as training."""
    # Create enhanced features
    daily_features = create_features(df)
    
    # Use same feature selection as training
    feature_cols = [col for col in daily_features.columns 
                   if col != TARGET and col != DATE_COL and pd.api.types.is_numeric_dtype(daily_features[col])]
    
    return daily_features[feature_cols]

def evaluate_model():
    """Evaluate model performance on 2023 test data."""
    df = load_data()
    model = joblib.load('business_optimized_model.joblib')
    
    # Filter for 2023 test data
    test_df = df[df.index.year == 2023].copy()
    
    # Create and prepare features
    features_df = prepare_features(test_df)
    
    print(f"\nNumber of features used for prediction: {features_df.shape[1]}")
    print("Features:", ", ".join(features_df.columns))
    
    # Make predictions
    predictions = model.predict(features_df)
    actuals = test_df[TARGET].values
    
    # Calculate loss
    loss = weighted_quantile_loss(actuals, predictions)
    
    # Calculate basic metrics
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions)**2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # Calculate over/under prediction statistics
    over_predictions = predictions > actuals
    under_predictions = predictions < actuals
    
    over_prediction_count = np.sum(over_predictions)
    under_prediction_count = np.sum(under_predictions)
    
    over_prediction_avg = np.mean(predictions[over_predictions] - actuals[over_predictions]) if any(over_predictions) else 0
    under_prediction_avg = np.mean(actuals[under_predictions] - predictions[under_predictions]) if any(under_predictions) else 0
    
    # Print results
    print("\nModel Evaluation Results (2023 Test Data):")
    print("=" * 50)
    print(f"Weighted Quantile Loss: {loss:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.1f}%")
    print("\nPrediction Distribution:")
    print(f"Over-predictions: {over_prediction_count} ({over_prediction_count/len(actuals)*100:.1f}%)")
    print(f"Under-predictions: {under_prediction_count} ({under_prediction_count/len(actuals)*100:.1f}%)")
    print(f"Perfect predictions: {len(actuals) - over_prediction_count - under_prediction_count}")
    print(f"\nAverage over-prediction amount: {over_prediction_avg:.2f}")
    print(f"Average under-prediction amount: {under_prediction_avg:.2f}")
    
    # Calculate monthly metrics
    monthly_metrics = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Date': test_df.index
    }).set_index('Date')
    
    monthly_metrics['Error'] = monthly_metrics['Predicted'] - monthly_metrics['Actual']
    monthly_metrics['AbsError'] = np.abs(monthly_metrics['Error'])
    monthly_metrics['OverPrediction'] = monthly_metrics['Error'] > 0
    
    monthly_stats = monthly_metrics.resample('M').agg({
        'Actual': 'sum',
        'Predicted': 'sum',
        'Error': ['mean', 'std'],
        'AbsError': 'mean',
        'OverPrediction': 'mean'
    })
    
    print("\nMonthly Performance:")
    print("=" * 50)
    print(monthly_stats)
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    plt.hist(predictions - actuals, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution (2023)')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.savefig('prediction_error_distribution_2023.png')
    plt.close()
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([0, max(actuals)], [0, max(actuals)], 'r--')
    plt.title('Actual vs Predicted Orders (2023)')
    plt.xlabel('Actual Orders')
    plt.ylabel('Predicted Orders')
    plt.savefig('actual_vs_predicted_2023.png')
    plt.close()
    
    # Plot time series comparison
    plt.figure(figsize=(15, 6))
    plt.plot(test_df.index, actuals, label='Actual', alpha=0.7)
    plt.plot(test_df.index, predictions, label='Predicted', alpha=0.7)
    plt.title('Time Series Comparison (2023)')
    plt.xlabel('Date')
    plt.ylabel('Orders')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_series_comparison_2023.png')
    plt.close()

if __name__ == "__main__":
    evaluate_model() 