"""
Quick test to verify observation space conversion is working
"""

import sys
import numpy as np
import pandas as pd
from trading_environment import FuturesTradingEnv
from dict_to_box_wrapper import DictToBoxObservationWrapper
from stable_baselines3 import PPO
from improved_reward_configs import TREND_RIDER_CONFIG

def test_observation_fix():
    """Test if the observation space conversion fixes the model prediction"""
    
    # Create sample market data (100 rows as required)
    print("Creating sample market data...")
    np.random.seed(42)
    
    # Create realistic OHLCV data
    data = []
    base_price = 45000.0
    for i in range(100):
        # Random walk price
        change = np.random.normal(0, 0.002)  # 0.2% std dev
        base_price *= (1 + change)
        
        high = base_price * (1 + abs(np.random.normal(0, 0.001)))
        low = base_price * (1 - abs(np.random.normal(0, 0.001)))
        volume = np.random.uniform(1000, 5000)
        
        data.append({
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i),
            'open': base_price,
            'high': high,
            'low': low,
            'close': base_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"Created market data with {len(df)} rows")
    
    # Test 1: Create environment and check observation format
    print("\n=== Test 1: Environment Observation Format ===")
    try:
        env = FuturesTradingEnv(
            df=df,
            window_size=30,
            initial_equity=10000,
            reward_config=TREND_RIDER_CONFIG
        )
        
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        print(f"Original observation type: {type(obs)}")
        if isinstance(obs, dict):
            for key, value in obs.items():
                print(f"  {key}: {type(value)} shape {value.shape}")
        else:
            print(f"  Observation shape: {obs.shape}")
            
    except Exception as e:
        print(f"Error creating environment: {e}")
        return False
    
    # Test 2: Wrap environment and check converted observation
    print("\n=== Test 2: Wrapped Environment Observation Format ===")
    try:
        wrapped_env = DictToBoxObservationWrapper(env)
        
        obs = wrapped_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        print(f"Wrapped observation type: {type(obs)}")
        print(f"Wrapped observation shape: {obs.shape}")
        print(f"Wrapped observation space: {wrapped_env.observation_space}")
        
    except Exception as e:
        print(f"Error wrapping environment: {e}")
        return False
    
    # Test 3: Load model and test prediction
    print("\n=== Test 3: Model Prediction Test ===")
    try:
        model_path = "models/best_model.zip"
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path)
        
        print("Testing model prediction...")
        action, _states = model.predict(obs, deterministic=True)
        
        print(f"Model prediction successful!")
        print(f"Action type: {type(action)}")
        print(f"Action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
        print(f"Action values: {action}")
        
        return True
        
    except Exception as e:
        print(f"Error with model prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_observation_fix()
    if success:
        print("\n✅ SUCCESS: Observation space fix is working!")
    else:
        print("\n❌ FAILED: Observation space fix needs more work")
