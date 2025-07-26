#!/usr/bin/env python3
"""
Simple test to check for entry price errors during normal operation
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trading_environment import FuturesTradingEnv

# Set logging to only show warnings and errors
logging.basicConfig(level=logging.WARNING)

def create_simple_data():
    """Create minimal test data"""
    dates = pd.date_range('2024-01-01', periods=50, freq='15min')
    prices = [50000 + i * 10 for i in range(len(dates))]  # Simple ascending prices
    
    df = pd.DataFrame({
        'timestamp': dates.astype('int64') // 10**9,
        'open': prices, 'high': prices, 'low': prices, 'close': prices,
        'volume': [1000] * len(dates),
        'returns': [0.001] * len(dates),
        'rsi': [50] * len(dates),
        'ema_10': prices, 'ema_20': prices,
        'macd': [0] * len(dates),
        'adx': [50] * len(dates),
        'atr': [p * 0.01 for p in prices],
        'volume_ratio': [1.0] * len(dates)
    })
    return df

def test_normal_operation():
    """Test normal trading operation"""
    
    print("Testing normal trading operation...")
    
    df = create_simple_data()
    env = FuturesTradingEnv(df=df, initial_equity=10000.0, max_leverage=2.0)
    
    observation = env.reset()
    
    entry_price_errors = 0
    steps_completed = 0
    
    # Run a simple trading session
    for i in range(20):
        # Simple alternating actions
        action = 0.1 if i % 2 == 0 else -0.1
        
        obs, reward, terminated, truncated, info = env.step(action)
        steps_completed += 1
        
        # Check for entry price issues
        if env.position_size != 0 and env.entry_price <= 0:
            entry_price_errors += 1
            print(f"Step {i}: Entry price error - position_size={env.position_size:.6f}, entry_price={env.entry_price}")
        
        if terminated or truncated:
            break
    
    print(f"Completed {steps_completed} steps")
    print(f"Entry price errors: {entry_price_errors}")
    print(f"Final position size: {env.position_size:.6f}")
    print(f"Final entry price: ${env.entry_price:.2f}")
    print(f"Final equity: ${env.equity:.2f}")
    
    if entry_price_errors == 0:
        print("✅ No entry price errors detected!")
        return True
    else:
        print(f"❌ {entry_price_errors} entry price errors detected")
        return False

if __name__ == "__main__":
    success = test_normal_operation()
    sys.exit(0 if success else 1)
