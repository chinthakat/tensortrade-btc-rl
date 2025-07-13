#!/usr/bin/env python3
"""
Debug script to analyze why no trades are being executed
"""

import numpy as np
from stable_baselines3 import PPO
from trading_environment import FuturesTradingEnv
import json
import pandas as pd

def debug_trading_actions():
    print("🔍 Debugging Trading Actions")
    print("=" * 50)
    
    # Load the latest trained model
    model_path = "episodes/episode_02_20250713_171414/models/final_model_episode_02_20250713_171414.zip"
    
    try:
        # Load configuration
        with open('configs/ppo_production.json', 'r') as f:
            config = json.load(f)
        
        # Load data
        data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
        df = pd.read_csv(data_path)
        print(f"📊 Loaded {len(df)} data points")
        
        # Create environment
        env = FuturesTradingEnv(
            df=df.iloc[10000:12000],  # Use subset for testing
            initial_equity=10000.0,
            window_size=60,
            training_iteration=999
        )
        
        # Load model
        model = PPO.load(model_path)
        print(f"🤖 Loaded model from: {model_path}")
        
        # Test some actions
        obs, _ = env.reset()
        
        print("\n📈 Testing Actions:")
        print("-" * 40)
        
        action_types = ["HOLD", "BUY", "SELL", "CANCEL"]
        
        for step in range(20):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Decode action
            action_type = int(action[0])
            leverage = action[1]
            risk_percentage = action[2]
            
            action_name = action_types[action_type] if action_type < len(action_types) else "UNKNOWN"
            
            print(f"Step {step:2d}: {action_name:6s} | Leverage: {leverage:6.3f} | Risk: {risk_percentage:6.3f}")
            
            # Check current state
            current_price = env._safe_get_price_data(env.current_step, 'close', 50000)
            position_value = env.position_size * current_price if current_price > 0 else 0
            
            # Calculate what would happen
            if action_name in ["BUY", "SELL"]:
                # Calculate target position
                risk_equity = env.equity * risk_percentage
                target_position_value = leverage * risk_equity
                target_position_size = target_position_value / current_price if current_price > 0 else 0
                trade_size = target_position_size - env.position_size
                
                print(f"         Current pos: {env.position_size:8.6f} | Target pos: {target_position_size:8.6f} | Trade size: {trade_size:8.6f}")
                print(f"         Equity: {env.equity:8.2f} | Risk equity: {risk_equity:8.2f} | Position value: {position_value:8.2f}")
                
                # Check if trade would execute
                would_execute = abs(trade_size) > 0.001
                print(f"         Would execute: {'YES' if would_execute else 'NO'} (threshold: 0.001)")
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
                
            print()
    
        print(f"\n📊 Final Stats:")
        print(f"   Episode trades: {env.episode_trades}")
        print(f"   Final equity: {env.equity:.2f}")
        print(f"   Position size: {env.position_size:.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_trading_actions()
