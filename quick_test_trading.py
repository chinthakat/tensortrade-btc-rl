#!/usr/bin/env python3
"""
Quick Test Training Session
==========================

Run a short training session to generate trade data with the fixed price alignment
to verify that the anomalies are resolved.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv, TradeLogger

def quick_test_trading():
    """Run a quick trading test to generate new trade data"""
    
    print("🧪 Quick Test Trading Session")
    print("=" * 40)
    
    # Load data
    print("Loading market data...")
    df = pd.read_csv("data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv")
    print(f"✓ Loaded {len(df)} records")
    
    # Create unique test session ID
    session_id = f"test_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"episodes/{session_id}/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logger
    log_file = f"{log_dir}/trades_{session_id}.csv"
    logger = TradeLogger(log_file=log_file)
    
    # Create environment
    print("Creating trading environment...")
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000,
        window_size=10,
        log_file=log_file
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"✓ Environment reset, starting at step: {env.current_step}")
    
    # Run a series of trading actions
    print(f"\nRunning test trading actions:")
    print("-" * 30)
    
    actions_taken = 0
    for i in range(100):  # Take 100 steps
        # Alternate between different actions to generate trades
        if i % 20 == 0:
            action = [0.5]  # Long position
            action_type = "BUY"
        elif i % 20 == 10:
            action = [-0.5]  # Short position
            action_type = "SELL"
        else:
            action = [0.0]  # Hold
            action_type = "HOLD"
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if action_type != "HOLD":
            actions_taken += 1
            current_price = env._safe_get_price_data(env.current_step, 'close', 0.0)
            timestamp = env._safe_get_price_data(env.current_step, 'timestamp', 0)
            print(f"  Step {i+1}: {action_type} at ${current_price:,.2f} (timestamp: {pd.to_datetime(timestamp, unit='s')})")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Test session completed!")
    print(f"Actions taken: {actions_taken}")
    print(f"Trade log saved: {log_file}")
    
    # Quick analysis of generated trades
    if os.path.exists(log_file):
        trades = pd.read_csv(log_file)
        print(f"Trades logged: {len(trades)}")
        
        # Check for any obvious anomalies
        valid_entries = trades[trades['entry_price'] > 0]
        if len(valid_entries) > 0:
            print(f"Valid entry prices: {len(valid_entries)}")
            print(f"Price range: ${valid_entries['entry_price'].min():,.2f} - ${valid_entries['entry_price'].max():,.2f}")
        else:
            print("⚠️  No valid entry prices found")
    
    return log_file

if __name__ == "__main__":
    trade_file = quick_test_trading()
    print(f"\nTrade file generated: {trade_file}")
    print("You can now run the trade anomaly analyzer on this new file to verify the fix.")
