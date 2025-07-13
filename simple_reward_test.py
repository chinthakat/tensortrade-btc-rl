#!/usr/bin/env python3
"""
Simple test to verify reward logging fix
"""

import pandas as pd
import os
from trading_environment import FuturesTradingEnv

def main():
    print("Testing reward calculation and logging...")
    
    # Load data
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows of data")
    
    # Create log directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "simple_reward_test.csv")
    
    # Create environment
    env = FuturesTradingEnv(
        data=df,
        initial_balance=10000.0,
        transaction_cost=0.001,
        leverage=10,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        risk_free_rate=0.02,
        max_position_size=1.0,
        holding_cost_rate=0.0001,
        volatility_lookback=20,
        max_consecutive_losses=5,
        severe_drawdown_threshold=0.20,
        moderate_drawdown_threshold=0.10,
        log_file=log_file
    )
    
    # Test environment
    obs = env.reset()
    print(f"Environment reset. Balance: {env.balance}")
    
    # Execute some steps
    actions = [1, 0, 0, 2, 0]  # Buy, Hold, Hold, Sell, Hold
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.6f}, Position={env.position_size:.3f}")
        if hasattr(env, 'current_trade_reward'):
            print(f"  Current trade reward: {env.current_trade_reward:.6f}")
        if done:
            break
    
    # Check log file
    if os.path.exists(log_file):
        print(f"\nLog file created: {log_file}")
        with open(log_file, 'r') as f:
            content = f.read()
            print(f"Log file size: {len(content)} characters")
            if content:
                lines = content.split('\n')
                if len(lines) > 0:
                    print(f"Header: {lines[0]}")
                if len(lines) > 1:
                    for i, line in enumerate(lines[1:4], 1):
                        if line.strip():
                            print(f"Line {i}: {line}")
    else:
        print(f"Log file not found: {log_file}")

if __name__ == "__main__":
    main()
