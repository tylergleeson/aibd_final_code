import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime, timedelta
import holidays
from business_optimized_model import create_features, weighted_quantile_loss, is_peak_season

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def get_temperature_band(temp):
    """Determine temperature band."""
    if temp <= 60:
        return "Cold (≤60°F)"
    elif temp <= 74:
        return "Normal (61-74°F)"
    elif temp <= 84:
        return "Hot (75-84°F)"
    else:
        return "Very Hot (≥85°F)"

def get_holiday_period(date, us_holidays):
    """Determine holiday period (pre-holiday, holiday, post-holiday, or normal)."""
    # Convert to datetime if needed
    date = pd.Timestamp(date)
    
    # Find closest holiday
    holiday_dates = sorted(us_holidays.keys())
    closest_holiday = min(holiday_dates, key=lambda x: abs((pd.Timestamp(x) - date).days))
    days_diff = (pd.Timestamp(closest_holiday) - date).days
    
    if days_diff == 0:
        return "Holiday"
    elif 0 < days_diff <= 3:  # 3 days before holiday
        return "Pre-Holiday"
    elif -3 <= days_diff < 0:  # 3 days after holiday
        return "Post-Holiday"
    else:
        return "Normal"

def evaluate_model():
    """Comprehensive evaluation of the business optimized model."""
    print("Loading data...")
    df = pd.read_excel('final_total_clean.xlsx')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for SKU 1 and 2023
    df_2023 = df[(df['SKU'] == 1) & (df['Date'].dt.year == 2023)].copy()
    
    # Create complete date range for 2023
    date_range_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    daily_df = pd.DataFrame(index=date_range_2023)
    daily_df.index.name = 'Date'
    
    # Aggregate quantities by date
    quantities = df_2023.groupby('Date')['Quantity'].sum()
    daily_df = daily_df.join(quantities)
    daily_df['Quantity'] = daily_df['Quantity'].fillna(0)
    
    # Add weather and price features
    for col in ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 
                'windspeedmean', 'Fuel_Price', 'Avg_Price']:
        values = df_2023.groupby('Date')[col].first()
        daily_df[col] = values
        daily_df[col] = daily_df[col].ffill().bfill()  # Fill any missing values
    
    # Create features for prediction
    features = create_features(daily_df)
    
    # Load model and get expected features
    print("\nLoading model and making predictions...")
    model = joblib.load('business_optimized_model.joblib')
    expected_features = model.feature_name_
    
    # Ensure we only use the features the model expects
    features = features[expected_features]
    
    # Make predictions
    predictions = model.predict(features)
    
    # Add temperature bands and holiday periods
    daily_df['Temperature_Band'] = daily_df['temp'].apply(get_temperature_band)
    us_holidays_2023 = holidays.US(years=2023)
    daily_df['Holiday_Period'] = daily_df.index.map(lambda x: get_holiday_period(x, us_holidays_2023))
    
    # Calculate various metrics
    print("\nCalculating performance metrics...")
    
    # Basic metrics
    mae = mean_absolute_error(daily_df['Quantity'], predictions)
    rmse = np.sqrt(mean_squared_error(daily_df['Quantity'], predictions))
    mape = np.mean(np.abs((daily_df['Quantity'] - predictions) / 
                         (daily_df['Quantity'] + 1e-6))) * 100
    
    # Business-specific loss
    business_loss = weighted_quantile_loss(daily_df['Quantity'].values, predictions)
    
    # Temperature band performance
    temp_band_metrics = {}
    for band in ['Cold (≤60°F)', 'Normal (61-74°F)', 'Hot (75-84°F)', 'Very Hot (≥85°F)']:
        mask = daily_df['Temperature_Band'] == band
        if sum(mask) > 0:  # Only calculate if we have data for this band
            band_predictions = predictions[mask]
            band_actuals = daily_df.loc[mask, 'Quantity'].values
            band_over_pred = band_predictions > band_actuals
            
            temp_band_metrics[band] = {
                'MAE': mean_absolute_error(band_actuals, band_predictions),
                'RMSE': np.sqrt(mean_squared_error(band_actuals, band_predictions)),
                'Business_Loss': weighted_quantile_loss(band_actuals, band_predictions),
                'Over_predictions': sum(band_over_pred) / len(band_over_pred) * 100,
                'Under_predictions': sum(~band_over_pred) / len(band_over_pred) * 100,
                'Average_Safety_Stock': np.mean(band_predictions - band_actuals)
            }
    
    # Holiday period performance
    holiday_metrics = {}
    for period in ['Pre-Holiday', 'Holiday', 'Post-Holiday', 'Normal']:
        mask = daily_df['Holiday_Period'] == period
        if sum(mask) > 0:  # Only calculate if we have data for this period
            period_predictions = predictions[mask]
            period_actuals = daily_df.loc[mask, 'Quantity'].values
            
            holiday_metrics[period] = {
                'MAE': mean_absolute_error(period_actuals, period_predictions),
                'RMSE': np.sqrt(mean_squared_error(period_actuals, period_predictions)),
                'Business_Loss': weighted_quantile_loss(period_actuals, period_predictions)
            }
    
    # Separate peak vs off-peak season performance
    peak_mask = daily_df.index.month.isin(range(5, 10))  # May through September
    peak_loss = weighted_quantile_loss(
        daily_df.loc[peak_mask, 'Quantity'].values,
        predictions[peak_mask]
    )
    offpeak_loss = weighted_quantile_loss(
        daily_df.loc[~peak_mask, 'Quantity'].values,
        predictions[~peak_mask]
    )
    
    # Calculate over/under prediction statistics
    over_pred = predictions > daily_df['Quantity'].values
    under_pred = predictions < daily_df['Quantity'].values
    
    # Print metrics
    print("\nOverall Performance Metrics:")
    print(f"Business Loss: ${business_loss:.2f} per prediction")
    print(f"Peak Season Loss: ${peak_loss:.2f} per prediction")
    print(f"Off-Peak Season Loss: ${offpeak_loss:.2f} per prediction")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"\nPrediction Bias:")
    print(f"Overpredictions: {sum(over_pred)} days ({sum(over_pred)/len(over_pred)*100:.1f}%)")
    print(f"Underpredictions: {sum(under_pred)} days ({sum(under_pred)/len(under_pred)*100:.1f}%)")
    
    print("\nTemperature Band Performance:")
    for band, metrics in temp_band_metrics.items():
        print(f"\n{band}:")
        print(f"- MAE: {metrics['MAE']:.2f} units")
        print(f"- RMSE: {metrics['RMSE']:.2f} units")
        print(f"- Business Loss: ${metrics['Business_Loss']:.2f} per prediction")
        print(f"- Over-predictions: {metrics['Over_predictions']:.1f}%")
        print(f"- Under-predictions: {metrics['Under_predictions']:.1f}%")
        print(f"- Average Safety Stock: {metrics['Average_Safety_Stock']:.2f} units")
    
    print("\nHoliday Period Performance:")
    for period, metrics in holiday_metrics.items():
        print(f"\n{period} Period:")
        print(f"- MAE: {metrics['MAE']:.2f} units")
        print(f"- RMSE: {metrics['RMSE']:.2f} units")
        print(f"- Business Loss: ${metrics['Business_Loss']:.2f} per prediction")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': daily_df.index,
        'Actual_Quantity': daily_df['Quantity'].values,
        'Predicted_Quantity': predictions,
        'Temperature': daily_df['temp'].values,
        'Temperature_Band': daily_df['Temperature_Band'],
        'Holiday_Period': daily_df['Holiday_Period'],
        'Is_Peak_Season': [is_peak_season(d.month) for d in daily_df.index],
        'Is_Holiday': [d in holidays.US() for d in daily_df.index],
        'Prediction_Error': predictions - daily_df['Quantity'].values,
        'Business_Loss': [
            1.78 * (pred - actual) if pred > actual else 11.54 * (actual - pred)
            for pred, actual in zip(predictions, daily_df['Quantity'].values)
        ]
    })
    
    # Save results to CSV
    results_df.to_csv('2023_evaluation_results.csv', index=False)
    print("\nSaved detailed results to '2023_evaluation_results.csv'")
    
    # Create visualization
    plt.figure(figsize=(15, 15))
    
    # Plot actual vs predicted
    plt.subplot(3, 1, 1)
    plt.plot(daily_df.index, daily_df['Quantity'], 
             label='Actual Orders', color='#2ecc71', linewidth=2)
    plt.plot(daily_df.index, predictions, 
             label='Predicted Orders', color='#e74c3c', linewidth=2, alpha=0.8)
    
    # Add peak season shading
    for month in range(5, 10):  # May through September
        peak_dates = daily_df.index[daily_df.index.month == month]
        plt.axvspan(peak_dates[0], peak_dates[-1], color='gray', alpha=0.1)
    
    plt.title('Pool Guy Chlorine (SKU 1) - 2023 Predictions vs Actual Orders', 
              fontsize=14, pad=20)
    plt.ylabel('Order Quantity', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot temperature band performance
    plt.subplot(3, 1, 2)
    temp_band_losses = [metrics['Business_Loss'] for metrics in temp_band_metrics.values()]
    temp_bands = list(temp_band_metrics.keys())
    plt.bar(temp_bands, temp_band_losses, color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])
    plt.title('Business Loss by Temperature Band', fontsize=14, pad=20)
    plt.ylabel('Business Loss ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot holiday period performance
    plt.subplot(3, 1, 3)
    holiday_losses = [metrics['Business_Loss'] for metrics in holiday_metrics.values()]
    holiday_periods = list(holiday_metrics.keys())
    plt.bar(holiday_periods, holiday_losses, color=['#9b59b6', '#e67e22', '#1abc9c', '#34495e'])
    plt.title('Business Loss by Holiday Period', fontsize=14, pad=20)
    plt.ylabel('Business Loss ($)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('2023_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to '2023_evaluation_results.png'")

if __name__ == "__main__":
    evaluate_model() 