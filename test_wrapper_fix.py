#!/usr/bin/env python3
"""
Test the wrapper attribute delegation fix
"""

import pandas as pd
from trading_environment import FuturesTradingEnv
from action_space_wrapper import DictToBoxActionWrapper

def test_wrapper_attributes():
    """Test that wrapper properly delegates attributes"""
    
    print("Testing wrapper attribute delegation...")
    
    # Load sample data
    try:
        data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
        df = pd.read_csv(data_path)
        print(f"Loaded data: {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Create environment
    env = FuturesTradingEnv(
        df=df.head(1000), 
        initial_equity=10000,
        use_advanced_action_space=True,
        max_leverage=5.0
    )
    
    print(f"Original env has price_data: {hasattr(env, 'price_data')}")
    print(f"Original env price_data shape: {env.price_data.shape}")
    
    # Wrap environment
    wrapped_env = DictToBoxActionWrapper(env)
    
    # Test attribute access
    try:
        print(f"Wrapped env has price_data: {hasattr(wrapped_env, 'price_data')}")
        print(f"Wrapped env price_data shape: {wrapped_env.price_data.shape}")
        print(f"Wrapped env current_step: {wrapped_env.current_step}")
        print("✅ Attribute delegation working!")
        return True
    except Exception as e:
        print(f"❌ Attribute delegation failed: {e}")
        return False

if __name__ == "__main__":
    test_wrapper_attributes()
