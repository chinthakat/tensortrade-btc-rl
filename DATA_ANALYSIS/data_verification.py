#!/usr/bin/env python3
"""
Data Verification Script
========================

Checks for mismatches between CSV data and trading environment price data.
"""

import pandas as pd
import sys
import os

# Add the parent directory to sys.path to import the trading environment
sys.path.append('../')

from trading_environment import FuturesTradingEnv

def compare_data_sources():
    """Compare raw CSV data with trading environment price data"""
    
    # Load raw CSV data
    csv_file = "../data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
    print(f"Loading raw CSV data from: {csv_file}")
    
    raw_df = pd.read_csv(csv_file)
    print(f"Raw CSV shape: {raw_df.shape}")
    print(f"Raw CSV columns: {list(raw_df.columns)}")
    
    # Initialize trading environment 
    print("\nInitializing trading environment...")
    env = FuturesTradingEnv(
        df=raw_df,
        initial_equity=10000,
        window_size=10
    )
    
    print(f"Environment df shape: {env.df.shape}")
    print(f"Environment price_data shape: {env.price_data.shape}")
    print(f"Environment df columns: {list(env.df.columns)}")
    print(f"Environment price_data columns: {list(env.price_data.columns)}")
    
    # Compare specific rows around step 594
    test_steps = [590, 591, 592, 593, 594, 595, 596, 597, 598]
    
    print("\n" + "="*80)
    print("DETAILED COMPARISON FOR STEPS 590-598")
    print("="*80)
    
    for step in test_steps:
        if step < len(raw_df) and step < len(env.price_data):
            raw_row = raw_df.iloc[step]
            env_row = env.price_data.iloc[step]
            
            print(f"\nStep {step}:")
            print(f"Raw CSV    - Close: {raw_row['close']:>12.2f}, High: {raw_row['high']:>12.2f}, Low: {raw_row['low']:>12.2f}, Timestamp: {raw_row['timestamp']}")
            print(f"Environment - Close: {env_row['close']:>12.2f}, High: {env_row['high']:>12.2f}, Low: {env_row['low']:>12.2f}, Timestamp: {env_row['timestamp']}")
            
            # Check for differences
            close_diff = abs(raw_row['close'] - env_row['close'])
            high_diff = abs(raw_row['high'] - env_row['high'])
            low_diff = abs(raw_row['low'] - env_row['low'])
            
            if close_diff > 0.01 or high_diff > 0.01 or low_diff > 0.01:
                print(f"⚠️  MISMATCH! Close diff: {close_diff:.4f}, High diff: {high_diff:.4f}, Low diff: {low_diff:.4f}")
            else:
                print("✅ Data matches")
    
    # Test the _safe_get_price_data method directly
    print("\n" + "="*80)
    print("TESTING _safe_get_price_data METHOD")
    print("="*80)
    
    test_step = 594
    if test_step < len(env.price_data):
        safe_close = env._safe_get_price_data(test_step, 'close', 0.0)
        safe_high = env._safe_get_price_data(test_step, 'high', 0.0)
        safe_low = env._safe_get_price_data(test_step, 'low', 0.0)
        
        print(f"Step {test_step} via _safe_get_price_data:")
        print(f"Close: {safe_close:.2f}, High: {safe_high:.2f}, Low: {safe_low:.2f}")
        
        # Compare with direct access
        direct_close = env.price_data.iloc[test_step]['close']
        direct_high = env.price_data.iloc[test_step]['high']
        direct_low = env.price_data.iloc[test_step]['low']
        
        print(f"Step {test_step} via direct access:")
        print(f"Close: {direct_close:.2f}, High: {direct_high:.2f}, Low: {direct_low:.2f}")
        
        if abs(safe_close - direct_close) > 0.01:
            print("⚠️  MISMATCH between _safe_get_price_data and direct access!")
        else:
            print("✅ _safe_get_price_data matches direct access")
    
    # Check if there are any shifts or transformations applied
    print("\n" + "="*80)
    print("CHECKING FOR DATA TRANSFORMATIONS")
    print("="*80)
    
    # Check if window_size causes any offset
    print(f"Environment window_size: {env.window_size}")
    print(f"Environment current_step after reset: {env.current_step}")
    
    # Reset environment and check current price
    obs, info = env.reset()
    current_price = env._safe_get_price_data(env.current_step, 'close', 0.0)
    print(f"After reset, current_step: {env.current_step}, current_price: {current_price:.2f}")
    
    # Check if this matches the expected CSV row
    if env.current_step < len(raw_df):
        expected_price = raw_df.iloc[env.current_step]['close']
        print(f"Expected price from CSV row {env.current_step}: {expected_price:.2f}")
        
        if abs(current_price - expected_price) > 0.01:
            print("⚠️  MISMATCH! Environment current_price differs from CSV!")
        else:
            print("✅ Environment current_price matches CSV")

if __name__ == "__main__":
    compare_data_sources()
