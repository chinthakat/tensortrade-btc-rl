#!/usr/bin/env python3
"""
Test script for enhanced action space with HOLD/CANCEL actions
"""

import numpy as np
import pandas as pd
from trading_environment import FuturesTradingEnv

def test_enhanced_actions():
    """Test the enhanced action space with HOLD, BUY, SELL, CANCEL"""
    
    print("Testing Enhanced Action Space with HOLD/CANCEL actions...")
    
    # Load sample data
    try:
        data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
        df = pd.read_csv(data_path)
        print(f"Loaded data: {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Test advanced action space
    print("\n1. Testing Advanced Action Space (HOLD/BUY/SELL/CANCEL)...")
    try:
        env = FuturesTradingEnv(
            df=df, 
            initial_equity=10000,
            use_advanced_action_space=True,  # Enable enhanced actions
            max_leverage=5.0
        )
        
        print(f"Action space: {env.action_space}")
        print(f"Trading threshold: {getattr(env, 'trading_threshold', 'N/A')}")
        
        # Test environment reset
        obs = env.reset()
        print(f"Initial observation shape: {obs[0]['market_features'].shape}")
        
        # Test different action types
        actions_to_test = [
            {"action_type": 0, "leverage": 2.0, "risk_percentage": 0.1},  # HOLD
            {"action_type": 1, "leverage": 3.0, "risk_percentage": 0.05},  # BUY
            {"action_type": 0, "leverage": 1.0, "risk_percentage": 0.02},  # HOLD
            {"action_type": 2, "leverage": 2.5, "risk_percentage": 0.03},  # SELL
            {"action_type": 3, "leverage": 1.0, "risk_percentage": 1.0},   # CANCEL
        ]
        
        for i, action in enumerate(actions_to_test):
            obs, reward, done, truncated, info = env.step(action)
            action_name = ["HOLD", "BUY", "SELL", "CANCEL"][action["action_type"]]
            print(f"Step {i+1}: {action_name} -> Reward: {reward:.6f}, Equity: {env.equity:.2f}, Action Type: {env.last_action_type}")
            
            if done or truncated:
                break
        
        print("Advanced action space test: PASSED")
        
    except Exception as e:
        print(f"Advanced action space test FAILED: {e}")
        return False
    
    # Test legacy action space with threshold
    print("\n2. Testing Legacy Action Space with Trading Threshold...")
    try:
        env_legacy = FuturesTradingEnv(
            df=df, 
            initial_equity=10000,
            use_advanced_action_space=False,  # Use legacy with threshold
            max_leverage=5.0
        )
        
        print(f"Legacy action space: {env_legacy.action_space}")
        print(f"Trading threshold: {env_legacy.trading_threshold}")
        
        obs = env_legacy.reset()
        
        # Test actions with threshold
        legacy_actions = [0.05, -0.05, 2.0, -3.0, 0.08]  # Some below threshold, some above
        
        for i, action in enumerate(legacy_actions):
            obs, reward, done, truncated, info = env_legacy.step([action])
            print(f"Step {i+1}: Leverage {action:.2f} -> Action: {env_legacy.last_action_type}, Reward: {reward:.6f}")
            
            if done or truncated:
                break
        
        print("Legacy action space test: PASSED")
        
    except Exception as e:
        print(f"Legacy action space test FAILED: {e}")
        return False
    
    print("\n✅ All tests passed! Enhanced action space is working correctly.")
    return True

if __name__ == "__main__":
    test_enhanced_actions()
