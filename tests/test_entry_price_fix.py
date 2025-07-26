#!/usr/bin/env python3
"""
Test Entry Price Validation Fix
Ensures entry prices are never zero for open positions.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create simple test data"""
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    np.random.seed(42)
    
    prices = []
    base_price = 50000
    for i in range(len(dates)):
        # Add some price variation
        change = np.random.normal(0, 0.001)
        new_price = base_price * (1 + change)
        prices.append(max(new_price, 1000))  # Minimum $1000
        base_price = new_price
    
    df = pd.DataFrame({
        'timestamp': dates.astype('int64') // 10**9,
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, len(dates)),
        'returns': [0] + [np.random.normal(0, 0.001) for _ in range(len(dates)-1)],
        'rsi': np.random.uniform(30, 70, len(dates)),
        'ema_10': prices,
        'ema_20': prices,
        'macd': np.random.normal(0, 10, len(dates)),
        'adx': np.random.uniform(20, 80, len(dates)),
        'atr': [p * 0.02 for p in prices],
        'volume_ratio': np.random.uniform(0.5, 2.0, len(dates))
    })
    
    return df

def test_entry_price_validation():
    """Test that entry price validation prevents zero entry prices"""
    
    print("🧪 TESTING ENTRY PRICE VALIDATION")
    print("=" * 50)
    
    df = create_test_data()
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=5.0
    )
    
    observation = env.reset()
    
    print(f"Initial state: position_size={env.position_size}, entry_price={env.entry_price}")
    
    # Test creating positions and validating entry prices
    entry_price_errors = 0
    valid_positions_created = 0
    
    for i in range(20):
        # Create random trading actions
        action = np.random.uniform(-0.5, 0.5)  # Small leverage values
        
        prev_position_size = env.position_size
        prev_entry_price = env.entry_price
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if a new position was created
        if env.position_size != 0 and prev_position_size == 0:
            valid_positions_created += 1
            print(f"Step {i}: New position created")
            print(f"  Position size: {env.position_size:.6f}")
            print(f"  Entry price: ${env.entry_price:.2f}")
            print(f"  Current step: {env.current_step}")
            
            # Validate entry price
            if env.entry_price <= 0:
                entry_price_errors += 1
                print(f"  ❌ INVALID ENTRY PRICE: {env.entry_price}")
            else:
                print(f"  ✅ Valid entry price: ${env.entry_price:.2f}")
        
        # Check if position state is consistent
        if env.position_size != 0:
            if env.entry_price <= 0:
                entry_price_errors += 1
                print(f"Step {i}: ❌ Position exists but entry_price={env.entry_price}")
        
        if terminated or truncated:
            break
    
    print(f"\n=== RESULTS ===")
    print(f"Valid positions created: {valid_positions_created}")
    print(f"Entry price errors: {entry_price_errors}")
    
    if entry_price_errors == 0:
        print("✅ NO ENTRY PRICE ERRORS DETECTED!")
        print("✅ Entry price validation is working correctly")
        return True
    else:
        print(f"❌ {entry_price_errors} ENTRY PRICE ERRORS DETECTED")
        print("❌ Entry price validation needs improvement")
        return False

if __name__ == "__main__":
    success = test_entry_price_validation()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
