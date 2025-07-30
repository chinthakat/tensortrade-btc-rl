"""
Debug script to test ATR calculation and dynamic stop-loss logic
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv

# Create test data with known volatility patterns
np.random.seed(42)
n_samples = 100

# Low volatility period
low_vol_returns = np.random.normal(0.0, 0.005, 50)  # 0.5% volatility
# High volatility period  
high_vol_returns = np.random.normal(0.0, 0.03, 50)   # 3% volatility

all_returns = np.concatenate([low_vol_returns, high_vol_returns])

# Generate price data
initial_price = 50000.0
prices = [initial_price]
for i in range(1, n_samples):
    prices.append(prices[-1] * (1 + all_returns[i]))

prices = np.array(prices)

# Generate OHLC with realistic spreads
highs = prices * (1 + np.random.uniform(0.002, 0.008, n_samples))
lows = prices * (1 - np.random.uniform(0.002, 0.008, n_samples))
opens = np.roll(prices, 1)
opens[0] = prices[0]
volumes = np.random.uniform(100, 1000, n_samples)

timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15T')

df = pd.DataFrame({
    'timestamp': [int(ts.timestamp()) for ts in timestamps],
    'open': opens,
    'high': highs,
    'low': lows,
    'close': prices,
    'volume': volumes
})

print("Raw ATR Calculation Test")
print("=" * 40)

# Create environment
env = FuturesTradingEnv(
    df=df,
    initial_equity=10000.0,
    use_dynamic_stops=True,
    atr_stop_loss_multiplier=2.0,
    atr_take_profit_multiplier=3.0,
    min_stop_loss_pct=0.001,    # Very low minimum
    max_stop_loss_pct=0.20,     # High maximum
    window_size=30
)

env.reset()

# Check ATR values directly from the feature data
print("\nATR Values from Feature Data:")
print("-" * 30)
for i in range(30, min(70, len(env.feature_columns)), 5):
    try:
        atr_value = env.feature_columns.iloc[i]['atr']
        price = env.price_data.iloc[i]['close']
        atr_pct = (atr_value / price) * 100
        
        # Determine volatility period
        period = "Low Vol" if i < 50 else "High Vol"
        
        print(f"Step {i:2d} | {period} | Price: ${price:6.0f} | ATR: ${atr_value:6.2f} | ATR%: {atr_pct:5.2f}%")
        
    except Exception as e:
        print(f"Step {i}: Error - {e}")

print("\nDynamic Stop Calculation Test:")
print("-" * 35)

# Test dynamic stop calculation
for i in range(35, min(60, len(env.price_data)), 10):
    env.current_step = i
    
    # Take a position to trigger stop calculation
    action = {'leverage': 10.0, 'risk_percentage': 0.5}
    env.step(action)
    
    if abs(env.position_size) > 0.001:
        current_price = env.price_data.iloc[env.current_step]['close']
        atr_value = env._get_current_atr(current_price)
        atr_pct = (atr_value / current_price) * 100
        
        # Calculate what the dynamic stops should be
        stop_pct, tp_pct = env._calculate_dynamic_stops(atr_value, current_price)
        
        period = "Low Vol" if i < 50 else "High Vol"
        
        print(f"Step {i:2d} | {period} | ATR: ${atr_value:6.2f} ({atr_pct:4.2f}%) | "
              f"Stop: {stop_pct*100:4.2f}% | TP: {tp_pct*100:4.2f}%")
        
        # Get the actual stops info
        stops_info = env.get_dynamic_stops_info()
        actual_stop = stops_info.get('dynamic_stop_loss_pct', 0) * 100
        actual_tp = stops_info.get('dynamic_take_profit_pct', 0) * 100
        
        print(f"       Actual: Stop: {actual_stop:4.2f}% | TP: {actual_tp:4.2f}%")
        print()

print("ATR Bounds Testing:")
print("-" * 20)

# Test with different multipliers to see bounds in action
test_multipliers = [1.0, 2.0, 3.0, 4.0, 5.0]

for mult in test_multipliers:
    env.atr_stop_loss_multiplier = mult
    env.atr_take_profit_multiplier = mult * 1.5
    
    # Use high volatility step
    env.current_step = 60
    current_price = env.price_data.iloc[env.current_step]['close']
    atr_value = env._get_current_atr(current_price)
    
    stop_pct, tp_pct = env._calculate_dynamic_stops(atr_value, current_price)
    
    print(f"Multiplier {mult}x: Stop {stop_pct*100:5.2f}% | TP {tp_pct*100:5.2f}%")
