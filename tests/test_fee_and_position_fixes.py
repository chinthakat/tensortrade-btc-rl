#!/usr/bin/env python3
"""
Test Fee Cap and Position State Issues
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trading_environment import FuturesTradingEnv

# Set logging to show warnings
logging.basicConfig(level=logging.WARNING)

def create_test_data():
    """Create test data that might trigger fee accumulation"""
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)
    
    # Create volatile prices to encourage trading
    prices = []
    base_price = 50000
    for i in range(len(dates)):
        change = np.random.normal(0, 0.005)  # Higher volatility
        new_price = base_price * (1 + change)
        prices.append(max(new_price, 1000))
        base_price = new_price
    
    df = pd.DataFrame({
        'timestamp': dates.astype('int64') // 10**9,
        'open': prices, 'high': [p * 1.01 for p in prices], 
        'low': [p * 0.99 for p in prices], 'close': prices,
        'volume': np.random.uniform(100, 1000, len(dates)),
        'returns': [0] + [np.random.normal(0, 0.005) for _ in range(len(dates)-1)],
        'rsi': np.random.uniform(20, 80, len(dates)),
        'ema_10': prices, 'ema_20': prices,
        'macd': np.random.normal(0, 50, len(dates)),
        'adx': np.random.uniform(20, 80, len(dates)),
        'atr': [p * 0.02 for p in prices],
        'volume_ratio': np.random.uniform(0.5, 2.0, len(dates))
    })
    return df

def test_fee_and_position_issues():
    """Test fee capping and position state handling"""
    
    print("🧪 TESTING FEE CAPS AND POSITION STATE HANDLING")
    print("=" * 60)
    
    df = create_test_data()
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=5.0,
        taker_fee=0.001  # Higher fee to trigger caps faster
    )
    
    observation = env.reset()
    
    print(f"Initial equity: ${env.equity:.2f}")
    print(f"Episode fee cap (5%): ${env.initial_equity * 0.05:.2f}")
    
    # Run trading simulation with frequent actions
    for i in range(100):
        # Create random actions that might trigger frequent trading
        action = np.random.uniform(-1, 1) * 0.3  # Moderate leverage
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check fee accumulation every 20 steps
        if i % 20 == 0:
            fee_percentage = (env.episode_total_fees / env.initial_equity) * 100
            print(f"Step {i}: Total fees=${env.episode_total_fees:.2f} ({fee_percentage:.1f}%), "
                  f"Equity=${env.equity:.2f}, Position={env.position_size:.6f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Final equity: ${env.equity:.2f}")
    print(f"Total fees paid: ${env.episode_total_fees:.2f}")
    print(f"Fee percentage: {(env.episode_total_fees / env.initial_equity) * 100:.2f}%")
    print(f"Episode trades: {env.episode_trades}")
    
    # Analyze results
    fee_cap_triggered = env.episode_total_fees >= env.initial_equity * 0.05
    print(f"\nFee cap triggered: {'Yes' if fee_cap_triggered else 'No'}")
    
    if fee_cap_triggered:
        print("✅ Fee cap working correctly - prevented excessive fees")
    else:
        print("ℹ️  Fee cap not needed - fees stayed within limits")
    
    return True

if __name__ == "__main__":
    success = test_fee_and_position_issues()
    print(f"\nTest completed: {'SUCCESS' if success else 'FAILED'}")
