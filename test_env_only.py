#!/usr/bin/env python3
"""Minimal test to isolate the NoneType error"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
from improved_reward_configs import TREND_RIDER_CONFIG

def test_environment_only():
    """Test just the environment creation and reset"""
    print("Creating test data...")
    
    # Create minimal test data
    data = []
    base_price = 50000
    for i in range(35):
        price = base_price + i + np.random.normal(0, 10)
        data.append({
            'timestamp': 1672531200 + i * 900,  # 15min intervals
            'open': price - 5,
            'high': price + 10,
            'low': price - 10,
            'close': price,
            'volume': 1000 + np.random.normal(0, 100)
        })
    
    df = pd.DataFrame(data)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Null values:\n{df.isnull().sum()}")
    print(f"Data sample:\n{df.head()}")
    
    try:
        print("\nCreating environment...")
        env = FuturesTradingEnv(
            df=df,
            window_size=30,
            initial_equity=10000.0,
            reward_config=TREND_RIDER_CONFIG
        )
        print("✅ Environment created successfully")
        
        print("\nResetting environment...")
        obs = env.reset()
        print(f"✅ Environment reset successful, observation type: {type(obs)}")
        
        if isinstance(obs, tuple):
            print(f"Observation shape: {obs[0].shape if hasattr(obs[0], 'shape') else 'No shape'}")
        else:
            print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")
            
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment_only()
