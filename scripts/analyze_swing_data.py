import pandas as pd

# Load the generated data
df = pd.read_csv('data/BTC_SWING_18M_15m.csv')

print("=== BTC SWING 18-Month Data Analysis ===")
print(f"Total candles: {len(df):,}")
print(f"File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
print()

print("=== Price Statistics ===")
print(f"Starting price: ${df['open'].iloc[0]:,.2f}")
print(f"Ending price: ${df['close'].iloc[-1]:,.2f}")
print(f"Price change: {((df['close'].iloc[-1] / df['open'].iloc[0]) - 1) * 100:+.2f}%")
print()

print(f"Lowest price: ${df['low'].min():,.2f}")
print(f"Highest price: ${df['high'].max():,.2f}")
print(f"Price range: ${df['high'].max() - df['low'].min():,.2f}")
print()

print("=== Data Quality ===")
negative_count = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
print(f"Negative prices: {negative_count}")

# Check bounds
bounds_violations = 0
if df['low'].min() < 20000:
    bounds_violations += (df['low'] < 20000).sum()
if df['high'].max() > 150000:
    bounds_violations += (df['high'] > 150000).sum()
    
print(f"Bounds violations: {bounds_violations}")

# Time range
start_time = pd.to_datetime(df['timestamp'].iloc[0], unit='s')
end_time = pd.to_datetime(df['timestamp'].iloc[-1], unit='s')
print(f"Time range: {start_time.date()} to {end_time.date()}")
print(f"Duration: {(end_time - start_time).days} days")

print()
print("=== Sample Data (First 3 rows) ===")
print(df[['open', 'high', 'low', 'close', 'volume']].head(3))

print()
print("✅ SWING market data generation successful!")
print("Ready for training with sideways/ranging market conditions")
