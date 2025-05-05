import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from business_optimized_model import create_features, calculate_safety_stock, EXOG_FEATURES
import joblib
from sklearn.preprocessing import LabelEncoder
import holidays

# Set style
plt.style.use('ggplot')
sns.set_style("whitegrid")

def encode_categorical(df):
    """Encode categorical variables for prediction"""
    categorical_columns = ['Product', 'conditions', 'description', 'holiday_type']
    
    # Drop columns that aren't needed for prediction
    for col in ['Product', 'conditions', 'description']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Encode holiday_type
    if 'holiday_type' in df.columns:
        le = LabelEncoder()
        df['holiday_type'] = le.fit_transform(df['holiday_type'].astype(str))
    
    return df

def plot_2023_predictions():
    # Load data
    df = pd.read_excel('final_total_clean.xlsx')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for SKU 1 and 2023
    df = df[df['SKU'] == 1]
    df_2023 = df[df['Date'].dt.year == 2023].copy()
    
    # Set date as index for feature creation
    df_2023.set_index('Date', inplace=True)
    
    # Create features
    features = create_features(df_2023)
    
    # Handle categorical variables
    features = encode_categorical(features)
    
    # Load model
    model = joblib.load('business_optimized_model.joblib')
    
    # Get predictions
    predictions = model.predict(features)
    
    # Calculate safety stock for each day
    safety_stock = []
    for idx in df_2023.index:
        temp = df_2023.loc[idx, 'temp']
        is_peak = 5 <= idx.month <= 9
        is_holiday = idx in holidays.US()
        errors = np.abs(df_2023['Quantity'] - predictions)  # Using absolute errors
        ss = calculate_safety_stock(errors, temp, is_peak, is_holiday)
        safety_stock.append(ss)
    
    # Add safety stock to predictions
    final_predictions = predictions + np.array(safety_stock)
    
    # Reset index for plotting
    df_2023.reset_index(inplace=True)
    
    # Create plot
    plt.figure(figsize=(15, 6))
    
    # Plot actual values
    plt.plot(df_2023['Date'], df_2023['Quantity'], 
             label='Actual Orders', color='#2ecc71', linewidth=2)
    
    # Plot predicted values
    plt.plot(df_2023['Date'], final_predictions, 
             label='Predicted Orders', color='#e74c3c', linewidth=2, alpha=0.8)
    
    # Add shaded area for peak season
    peak_season_mask = (df_2023['Date'].dt.month >= 5) & (df_2023['Date'].dt.month <= 9)
    plt.fill_between(df_2023['Date'], plt.ylim()[0], plt.ylim()[1],
                    where=peak_season_mask, color='gray', alpha=0.1, label='Peak Season')
    
    # Customize plot
    plt.title('2023 Pool Guy Chlorine: Actual vs Predicted Order Quantities', 
              fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Order Quantity', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('2023_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_2023_predictions() 