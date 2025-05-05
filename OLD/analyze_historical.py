import pandas as pd
import numpy as np
from datetime import datetime

# Constants
ID_COL = 'SKU'
DATE_COL = 'Date'
TARGET = 'orders'
POOL_GUY_CHLORINE_SKU = 1.0

# Load historical data
print("Loading historical data...")
df = pd.read_excel('final_total_clean.xlsx', parse_dates=[DATE_COL])

# Filter for Pool Guy Chlorine
df = df[df[ID_COL] == POOL_GUY_CHLORINE_SKU]

# Create a complete date range
min_date = df[DATE_COL].min()
max_date = df[DATE_COL].max()
all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
print(f"\nDate range in data: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

# Create a DataFrame with all dates
daily_df = pd.DataFrame(index=all_dates)
daily_df.index.name = DATE_COL

# Aggregate quantities by date
quantities_by_date = df.groupby(DATE_COL)['Quantity'].sum()

# Merge with all dates to identify missing days (which would be days with zero orders)
daily_df = daily_df.join(quantities_by_date)

# Fill NaN values with 0 (these are days with no orders)
daily_df['Quantity'] = daily_df['Quantity'].fillna(0)

# Basic statistics
print("\nPool Guy Chlorine (SKU 1) Quantity Analysis:")
print("\nQuantity Distribution:")
print(daily_df['Quantity'].describe())

# Count days with zero orders
zero_quantity_days = (daily_df['Quantity'] == 0).sum()
total_days = len(daily_df)
print(f"\nDays with zero quantity: {zero_quantity_days} ({(zero_quantity_days/total_days)*100:.1f}% of days)")

# Print days with zero quantity
if zero_quantity_days > 0:
    print("\nDates with zero quantity:")
    zero_dates = daily_df[daily_df['Quantity'] == 0].index
    for date in zero_dates:
        print(date.strftime('%Y-%m-%d'))

# Print top 10 highest quantity days
print("\nTop 10 Highest Quantity Days:")
top_10_days = daily_df.nlargest(10, 'Quantity')
for date, row in top_10_days.iterrows():
    print(f"{date.strftime('%Y-%m-%d')}: {int(row['Quantity'])} units")

# Print bottom 10 quantity days (excluding zeros)
print("\nBottom 10 Non-Zero Quantity Days:")
bottom_10_days = daily_df[daily_df['Quantity'] > 0].nsmallest(10, 'Quantity')
for date, row in bottom_10_days.iterrows():
    print(f"{date.strftime('%Y-%m-%d')}: {int(row['Quantity'])} units") 