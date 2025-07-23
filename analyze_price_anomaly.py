import pandas as pd
import numpy as np

# Load trade data
print("Loading trade data...")
trades = pd.read_csv('episodes/episode_01_20250721_234711/logs/trades_episode_01_20250721_234711_env0.csv')

# Get a sample anomalous trade
sample = trades[trades['trade_id'] == 'TRADE_00013'].iloc[0]
print(f'Trade ID: {sample["trade_id"]}')
print(f'Entry datetime: {sample["entry_datetime"]}')
print(f'Entry price: {sample["entry_price"]}')
print(f'Side: {sample["side"]}')
print(f'Status: {sample["status"]}')
print(f'Position size: {sample["position_size"]}')

# Load market data
print("\nLoading market data...")
market = pd.read_csv('data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv')

# Convert to datetime
trade_time = pd.to_datetime(sample['entry_datetime'])
market['datetime'] = pd.to_datetime(market['timestamp'], unit='s')

# Find closest market data
market['time_diff'] = abs(market['datetime'] - trade_time)
closest_idx = market['time_diff'].idxmin()
closest_market = market.iloc[closest_idx]

print(f'\nMarket data for same time:')
print(f'Market time: {closest_market["datetime"]}')
print(f'Market close: {closest_market["close"]}')
print(f'Market high: {closest_market["high"]}')
print(f'Market low: {closest_market["low"]}')

trade_price = float(sample['entry_price'])
market_price = float(closest_market['close'])
diff_pct = ((trade_price - market_price) / market_price) * 100
print(f'\nPrice Analysis:')
print(f'Trade price: ${trade_price:,.2f}')
print(f'Market price: ${market_price:,.2f}')
print(f'Difference: {diff_pct:.2f}%')

# Check if trade price is within market high/low range
market_high = float(closest_market['high'])
market_low = float(closest_market['low'])
print(f'Market range: ${market_low:,.2f} - ${market_high:,.2f}')
print(f'Trade price within range: {market_low <= trade_price <= market_high}')
