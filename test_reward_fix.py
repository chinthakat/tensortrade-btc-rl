#!/usr/bin/env python3
"""
Test script to verify reward logging is working properly
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv

def test_reward_logging():
    print("Testing reward logging fix...")
    
    # Create environment
    env = FuturesTradingEnv(
        data_file='data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv',
        initial_balance=10000.0,
        episode_name='test_reward_fix'
    )
    
    # Reset environment
    obs = env.reset()
    print(f"Environment reset. Initial equity: {env.equity:.2f}")
    
    # Test a few steps
    print("\nTesting step rewards:")
    for i in range(10):
        # Alternate between hold (0) and buy (1) actions
        action = 0 if i % 2 == 0 else 1
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i:2d}: Action={action}, Reward={reward:8.6f}, Equity={env.equity:8.2f}")
        
        if done or truncated:
            print(f"Episode ended at step {i}")
            break
    
    # Check if any trades were logged
    log_file = f"episodes/episode_test_reward_fix/logs/trades_episode_test_reward_fix_env0.csv"
    if os.path.exists(log_file):
        print(f"\nChecking log file: {log_file}")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"Log file has {len(lines)} lines")
            if len(lines) > 1:
                # Print header and first few data lines
                print("Header:", lines[0].strip())
                for i, line in enumerate(lines[1:6], 1):
                    if 'close_reward' in line:
                        parts = line.strip().split(',')
                        try:
                            reward_idx = lines[0].strip().split(',').index('close_reward')
                            reward_val = parts[reward_idx] if reward_idx < len(parts) else 'N/A'
                            print(f"Line {i}: close_reward = {reward_val}")
                        except (ValueError, IndexError):
                            print(f"Line {i}: {line.strip()[:100]}...")
    else:
        print(f"\nNo log file found at: {log_file}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_reward_logging()
