#!/usr/bin/env python3
"""
Test script to verify trade counting fix
"""

import numpy as np
from stable_baselines3 import PPO
from trading_environment import FuturesTradingEnv
import json
import pandas as pd

def test_trade_counting():
    print("🔍 Testing Trade Counting Fix")
    print("=" * 50)
    
    # Load data
    data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} data points")
    
    # Create environment
    env = FuturesTradingEnv(
        df=df.iloc[15000:15500],  # Use subset for testing
        initial_equity=10000.0,
        window_size=60,
        training_iteration=999
    )
    
    print(f"🏛️ Environment created with {len(env.df)} data points")
    
    # Test manual trades to verify counting
    obs, _ = env.reset()
    print(f"🚀 Environment reset. Initial episode_trades: {env.episode_trades}")
    
    # Force some trades by directly calling _execute_efficient_trade
    current_price = env._safe_get_price_data(env.current_step, 'close', 50000)
    print(f"💰 Current price: {current_price:.2f}")
    
    # Trade 1: Open long position
    print("\n📈 Trade 1: Opening long position...")
    env._execute_efficient_trade(0.1, current_price)  # Buy 0.1 BTC
    print(f"   Episode trades: {env.episode_trades}, Position: {env.position_size:.6f}")
    
    # Step forward
    action = np.array([0, 1.0, 0.1])  # HOLD action
    obs, reward, terminated, truncated, info = env.step(action)
    current_price = env._safe_get_price_data(env.current_step, 'close', 50000)
    
    # Trade 2: Increase position
    print("\n📈 Trade 2: Increasing position...")
    env._execute_efficient_trade(0.2, current_price)  # Increase to 0.2 BTC
    print(f"   Episode trades: {env.episode_trades}, Position: {env.position_size:.6f}")
    
    # Step forward
    obs, reward, terminated, truncated, info = env.step(action)
    current_price = env._safe_get_price_data(env.current_step, 'close', 50000)
    
    # Trade 3: Flip to short
    print("\n📉 Trade 3: Flipping to short position...")
    env._execute_efficient_trade(-0.15, current_price)  # Flip to short 0.15 BTC
    print(f"   Episode trades: {env.episode_trades}, Position: {env.position_size:.6f}")
    
    # Step forward
    obs, reward, terminated, truncated, info = env.step(action)
    current_price = env._safe_get_price_data(env.current_step, 'close', 50000)
    
    # Trade 4: Close position
    print("\n🔄 Trade 4: Closing position...")
    env._execute_efficient_trade(0.0, current_price)  # Close position
    print(f"   Episode trades: {env.episode_trades}, Position: {env.position_size:.6f}")
    
    print(f"\n📊 Final Results:")
    print(f"   Total episode trades: {env.episode_trades}")
    print(f"   Expected trades: 4")
    print(f"   ✅ Trade counting: {'WORKING' if env.episode_trades == 4 else 'BROKEN'}")
    
    # Test with backtest info
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Info episode_trades: {info.get('episode_trades', 'NOT_FOUND')}")

if __name__ == "__main__":
    test_trade_counting()
