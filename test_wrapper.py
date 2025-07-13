#!/usr/bin/env python3
"""
Test script for the updated action space wrapper
"""

import numpy as np
import pandas as pd
from trading_environment import FuturesTradingEnv
from action_space_wrapper import DictToBoxActionWrapper

def test_wrapper():
    """Test the updated action space wrapper"""
    
    print("Testing Updated Action Space Wrapper...")
    
    # Load sample data
    try:
        data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
        df = pd.read_csv(data_path)
        print(f"Loaded data: {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Test enhanced action space environment
    print("\n1. Testing Enhanced Action Space Environment...")
    env = FuturesTradingEnv(
        df=df.head(1000), 
        initial_equity=10000,
        use_advanced_action_space=True,
        max_leverage=5.0
    )
    
    print(f"Original action space: {env.action_space}")
    
    # Wrap environment
    wrapped_env = DictToBoxActionWrapper(env)
    print(f"Wrapped action space: {wrapped_env.action_space}")
    
    # Test reset
    obs = wrapped_env.reset()
    print(f"Reset successful, observation shape: {obs[0]['market_features'].shape}")
    
    # Test different Box actions and see how they convert
    test_actions = [
        np.array([-1.0, 0.0, 0.5]),  # Should map to HOLD
        np.array([-0.5, 0.8, 0.2]),  # Should map to HOLD/BUY
        np.array([0.0, -0.3, 0.8]),   # Should map to BUY/SELL
        np.array([0.5, 0.5, 0.1]),   # Should map to SELL  
        np.array([1.0, 0.2, 0.9])    # Should map to CANCEL
    ]
    
    for i, box_action in enumerate(test_actions):
        # Convert Box action to Dict action
        dict_action = wrapped_env.action(box_action)
        
        print(f"\nTest {i+1}:")
        print(f"  Box input: {box_action}")
        print(f"  Dict output: {dict_action}")
        
        # Test environment step
        obs, reward, done, truncated, info = wrapped_env.step(box_action)
        action_names = ["HOLD", "BUY", "SELL", "CANCEL"]
        action_type = dict_action['action_type']
        action_name = action_names[action_type] if action_type < len(action_names) else f"UNKNOWN({action_type})"
        
        print(f"  Result: {action_name}, Reward: {reward:.6f}")
        
        if done or truncated:
            break
    
    print("\n✅ Action space wrapper test completed successfully!")
    return True

if __name__ == "__main__":
    test_wrapper()
