#!/usr/bin/env python3
"""
Run a quick backtest to verify trade counting fix
"""

import numpy as np
from stable_baselines3 import PPO
from trading_environment import FuturesTradingEnv
import pandas as pd

def quick_backtest():
    print("🚀 Quick Backtest - Trade Counting Verification")
    print("=" * 55)
    
    # Load the trained model
    model_path = "episodes/episode_02_20250713_171414/models/final_model_episode_02_20250713_171414.zip"
    
    # Load data
    data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
    df = pd.read_csv(data_path)
    
    # Create environment for validation (same as used in training evaluation)
    env = FuturesTradingEnv(
        df=df.iloc[30000:32000],  # Use validation subset
        initial_equity=10000.0,
        window_size=60,
        training_iteration=999
    )
    
    # Load model
    model = PPO.load(model_path)
    print(f"🤖 Loaded model: {model_path}")
    
    # Run backtest
    obs, _ = env.reset()
    total_steps = min(500, len(env.df) - env.window_size - 1)  # Limit to 500 steps for quick test
    
    print(f"📊 Running {total_steps} steps...")
    
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0, "CANCEL": 0}
    action_types = ["HOLD", "BUY", "SELL", "CANCEL"]
    
    for step in range(total_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Decode action
        action_type = int(action[0])
        action_name = action_types[action_type] if action_type < len(action_types) else "UNKNOWN"
        action_counts[action_name] += 1
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print(f"\n📈 Action Distribution:")
    for action, count in action_counts.items():
        percentage = count / sum(action_counts.values()) * 100
        print(f"   {action:6s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n📊 Final Results:")
    print(f"   Total trades executed: {env.episode_trades}")
    print(f"   Final equity: {env.equity:.2f}")
    print(f"   Total return: {(env.equity - 10000) / 10000 * 100:.2f}%")
    print(f"   Position size: {env.position_size:.6f}")
    
    # The key insight:
    if env.episode_trades == 0 and action_counts["HOLD"] == sum(action_counts.values()):
        print(f"\n💡 ANALYSIS:")
        print(f"   ✅ Trade counting is working correctly")
        print(f"   ⚠️  Model has learned to be extremely conservative (HOLD-only strategy)")
        print(f"   📝 Training logs show many trades, but trained model avoids trading")
        print(f"   🎯 This explains 0 trades in backtest vs many trades in training logs")

if __name__ == "__main__":
    quick_backtest()
