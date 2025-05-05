import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Set style
plt.style.use('ggplot')

# Read data
df = pd.read_excel('final_total_clean.xlsx', parse_dates=['Date'])

# Filter for SKU 1
sku1_df = df[df['SKU'] == 1.0].copy()

# Aggregate by date
daily_orders = sku1_df.groupby('Date')['Quantity'].sum().reset_index()

# Create year and day of year columns
daily_orders['Year'] = daily_orders['Date'].dt.year
daily_orders['DayOfYear'] = daily_orders['Date'].dt.dayofyear

# Calculate early year statistics (first 28 days) for each year
early_year_stats = daily_orders[daily_orders['DayOfYear'] <= 28].groupby('Year').agg({
    'Quantity': ['mean', 'std', 'sum', 'count']
}).round(2)

print("\nEarly Year Comparison (First 28 days):")
print(early_year_stats)

# Create the main plot
plt.figure(figsize=(15, 10))

# Create two subplots
plt.subplot(2, 1, 1)

# Create complete date range including 2025
min_date = daily_orders['Date'].min()
max_date = datetime(2025, 12, 31)
all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
complete_df = pd.DataFrame({'Date': all_dates})

# Merge with actual data
complete_df = complete_df.merge(daily_orders, on='Date', how='left')

# Plot the full timeline
plt.plot(complete_df['Date'], complete_df['Quantity'], 
         label='Daily Orders', color='#2E86C1', alpha=0.7)

# Add 30-day moving average
ma_30 = complete_df['Quantity'].rolling(window=30, center=True).mean()
plt.plot(complete_df['Date'], ma_30, 
         label='30-day Moving Average', color='#E74C3C', linewidth=2)

# Customize the plot
plt.title('Pool Guy Chlorine (SKU 1) Daily Orders Over Time', fontsize=14, pad=20)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Add vertical lines for year transitions
for year in [2024, 2025]:
    plt.axvline(x=datetime(year, 1, 1), color='gray', linestyle='--', alpha=0.5)
    plt.text(datetime(year, 1, 1), plt.ylim()[1], str(year), 
             rotation=90, va='top', ha='right', alpha=0.5)

# Subplot for early year comparison
plt.subplot(2, 1, 2)

# Plot early year patterns for the last few years
years_to_compare = [2022, 2023, 2024, 2025]
colors = ['#2E86C1', '#E74C3C', '#27AE60', '#8E44AD']

for year, color in zip(years_to_compare, colors):
    year_data = daily_orders[daily_orders['Year'] == year]
    early_year_data = year_data[year_data['DayOfYear'] <= 28]
    if not early_year_data.empty:
        plt.plot(early_year_data['DayOfYear'], early_year_data['Quantity'], 
                label=f'{year}', color=color, alpha=0.7, marker='o')

plt.title('Early Year Comparison (First 28 Days)', fontsize=14, pad=20)
plt.xlabel('Day of Year', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('sku1_orders_history.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate year-over-year change for early year periods
print("\nYear-over-Year Change in Early Year Orders (First 28 days):")
early_year_totals = early_year_stats['Quantity']['sum']
for i in range(len(early_year_totals)-1):
    year = early_year_totals.index[i]
    next_year = early_year_totals.index[i+1]
    if year >= 2022:  # Only show recent years
        change = ((early_year_totals[next_year] - early_year_totals[year]) / early_year_totals[year]) * 100
        print(f"{year} to {next_year}: {change:.1f}% change")

# Compare 2025 with historical patterns
print("\n2025 Early Year Analysis:")
avg_2022_2023 = early_year_stats.loc[[2022, 2023], ('Quantity', 'mean')].mean()
current_2025 = early_year_stats.loc[2025, ('Quantity', 'mean')]
percent_diff = ((current_2025 - avg_2022_2023) / avg_2022_2023) * 100
print(f"2025 vs 2022-2023 Average: {percent_diff:.1f}% difference") 