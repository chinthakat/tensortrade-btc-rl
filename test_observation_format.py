"""Test the observation format to understand the exact issue"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from trading_environment import FuturesTradingEnv
from action_space_wrapper import wrap_environment_for_algorithm
from improved_reward_configs import TREND_RIDER_CONFIG

def test_observation_format():
    """Test what format the observation is in"""
    
    # Create sample data
    dates = pd.date_range(start='2025-01-01', periods=100, freq='15min')
    np.random.seed(42)
    
    initial_price = 50000
    returns = np.random.normal(0, 0.001, 100)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates.astype(np.int64) // 10**9,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, 100)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.002, 100))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.002, 100))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    try:
        # Create environment
        print("Creating environment...")
        env = FuturesTradingEnv(
            df=df,
            window_size=30,
            initial_equity=10000.0,
            reward_config=TREND_RIDER_CONFIG
        )
        
        print("Environment created successfully")
        
        # Check observation space
        print(f"\nOriginal observation space: {env.observation_space}")
        print(f"Observation space type: {type(env.observation_space)}")
        
        # Reset and check observation
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        print(f"\nOriginal observation type: {type(obs)}")
        if isinstance(obs, dict):
            print(f"Observation keys: {list(obs.keys())}")
            for key, value in obs.items():
                print(f"  {key}: shape={getattr(value, 'shape', 'no shape')}, type={type(value)}")
        elif hasattr(obs, 'shape'):
            print(f"Observation shape: {obs.shape}")
        
        # Wrap environment
        print("\nWrapping environment...")
        wrapped_env = wrap_environment_for_algorithm(env, "PPO")
        
        # Check wrapped observation space
        print(f"\nWrapped observation space: {wrapped_env.observation_space}")
        print(f"Wrapped observation space type: {type(wrapped_env.observation_space)}")
        
        # Reset wrapped environment
        wrapped_obs = wrapped_env.reset()
        if isinstance(wrapped_obs, tuple):
            wrapped_obs = wrapped_obs[0]
        
        print(f"\nWrapped observation type: {type(wrapped_obs)}")
        if isinstance(wrapped_obs, dict):
            print(f"Wrapped observation keys: {list(wrapped_obs.keys())}")
            for key, value in wrapped_obs.items():
                print(f"  {key}: shape={getattr(value, 'shape', 'no shape')}, type={type(value)}")
        elif hasattr(wrapped_obs, 'shape'):
            print(f"Wrapped observation shape: {wrapped_obs.shape}")
        
        # Try to load model and predict
        print("\nTesting model prediction...")
        model = PPO.load("models/best_model.zip")
        
        print("Trying to predict with original observation...")
        try:
            action, _ = model.predict(obs, deterministic=True)
            print(f"✅ Prediction with original obs successful: {action}")
        except Exception as e:
            print(f"❌ Prediction with original obs failed: {e}")
        
        print("Trying to predict with wrapped observation...")
        try:
            action, _ = model.predict(wrapped_obs, deterministic=True)
            print(f"✅ Prediction with wrapped obs successful: {action}")
        except Exception as e:
            print(f"❌ Prediction with wrapped obs failed: {e}")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_observation_format()
