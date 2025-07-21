#!/usr/bin/env python3
"""
Test Price Alignment Fix
=======================

Quick test to verify that the price alignment fix works correctly.
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv

def test_price_alignment():
    """Test if trades now use correct timestep prices"""
    
    print("🧪 Testing Price Alignment Fix")
    print("=" * 40)
    
    # Load minimal data for testing
    market_data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
    
    print("Loading market data...")
    df = pd.read_csv(market_data_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Create environment
    print("Creating trading environment...")
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000,
        window_size=10,
        log_file=None
    )
    print(f"✓ Environment created")
    
    # Reset environment
    obs, info = env.reset()
    print(f"✓ Environment reset, starting at step: {env.current_step}")
    
    # Take a few steps and check price alignment
    print(f"\nTesting price alignment during steps:")
    print("-" * 40)
    
    for i in range(5):
        # Store step info before action
        step_before = env.current_step
        price_before = env._safe_get_price_data(step_before, 'close', 0.0)
        timestamp_before = env._safe_get_price_data(step_before, 'timestamp', 0)
        
        # Take action (BUY with small leverage)
        action = [0.1]  # Small long position
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check step after
        step_after = env.current_step
        price_after = env._safe_get_price_data(step_after, 'close', 0.0)
        timestamp_after = env._safe_get_price_data(step_after, 'timestamp', 0)
        
        print(f"Step {i+1}:")
        print(f"  Before: Step {step_before}, Price ${price_before:,.2f}, Time {pd.to_datetime(timestamp_before, unit='s')}")
        print(f"  After:  Step {step_after}, Price ${price_after:,.2f}, Time {pd.to_datetime(timestamp_after, unit='s')}")
        print(f"  Step increment: {step_after - step_before}")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Price alignment test completed!")
    print(f"Current step should increment by 1 each time.")

if __name__ == "__main__":
    test_price_alignment()
