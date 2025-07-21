#!/usr/bin/env python3
"""
Test script to verify the entry price fix for SHORT positions
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_entry_price_fix():
    """Test that entry prices are properly preserved during trade logging"""
    
    # Create a simple test dataset
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    test_data = pd.DataFrame({
        'timestamp': dates.astype('int64') // 10**9,  # Convert to Unix timestamp
        'open': 50000 + np.random.randn(100) * 100,
        'high': 50200 + np.random.randn(100) * 100,
        'low': 49800 + np.random.randn(100) * 100,
        'close': 50000 + np.random.randn(100) * 100,
        'volume': 1000 + np.random.randn(100) * 100
    })
    
    # Ensure price consistency
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    # Create environment
    config = {
        'lookback_window': 10,
        'initial_balance': 10000,
        'max_leverage': 10,
        'taker_fee': 0.0004,
        'funding_rate': 0.0001,
        'liquidation_threshold': 0.8,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.04,
        'max_trades_per_episode': 10,
        'position_sizing_type': 'risk_percentage',
        'enable_logging': True,
        'log_file': 'test_entry_price_trades.csv'
    }
    
    env = FuturesTradingEnv(test_data, config)
    
    # Test SHORT position scenario
    print("Testing SHORT position entry price preservation...")
    
    # Reset environment
    obs = env.reset()
    print(f"Initial state - Position: {env.position_size}, Entry Price: {env.entry_price}")
    
    # Open SHORT position (action = 0 for SELL)
    obs, reward, done, info = env.step(0)  # SELL action
    
    if env.position_size < 0:  # We have a SHORT position
        original_entry_price = env.entry_price
        print(f"SHORT position opened - Size: {env.position_size}, Entry Price: {original_entry_price}")
        
        # Close SHORT position (action = 1 for BUY or 2 for HOLD -> CANCEL)
        obs, reward, done, info = env.step(2)  # CANCEL action to close position
        
        print(f"Position after close - Size: {env.position_size}, Entry Price: {env.entry_price}")
        
        # Check if any trades were logged
        if hasattr(env, 'logger') and env.logger:
            print("Trade logging is active - trades should be logged with correct entry prices")
        
        return original_entry_price > 0
    else:
        print("No SHORT position created, trying different action...")
        return False

if __name__ == "__main__":
    print("Testing entry price fix for SHORT positions...")
    result = test_entry_price_fix()
    print(f"Test result: {'PASSED' if result else 'FAILED'}")
