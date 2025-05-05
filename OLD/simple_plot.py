import pandas as pd
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv('2023_orders.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Create plot
plt.figure(figsize=(15, 6))

# Plot actual values
plt.plot(df['Date'], df['Actual_Orders'], 
         label='Actual Orders', color='#2ecc71', linewidth=2)

# Plot predicted values
plt.plot(df['Date'], df['Predicted_Orders'], 
         label='Predicted Orders', color='#e74c3c', linewidth=2, alpha=0.8)

# Add shaded area for peak season (May-September)
peak_season_mask = (df['Date'].dt.month >= 5) & (df['Date'].dt.month <= 9)
plt.fill_between(df['Date'], plt.ylim()[0], plt.ylim()[1],
                where=peak_season_mask, color='gray', alpha=0.1, label='Peak Season')

# Customize plot
plt.title('2023 Pool Guy Chlorine: Actual vs Predicted Orders', fontsize=14, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Order Quantity', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Rotate x-axis labels
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('2023_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created plot: 2023_predictions_vs_actual.png") 