"""Debug script to identify the exact location of the NoneType error"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from trading_environment import FuturesTradingEnv
from action_space_wrapper import wrap_environment_for_algorithm
from improved_reward_configs import TREND_RIDER_CONFIG
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_environment_creation():
    """Test environment creation with sample data"""
    
    # Create sample data similar to what's being used
    dates = pd.date_range(start='2025-01-01', periods=100, freq='15min')
    np.random.seed(42)
    
    # Generate realistic BTCUSDT data
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
    
    print("Sample data created:")
    print(df.head())
    print(f"Data shape: {df.shape}")
    print(f"Any nulls in data: {df.isnull().any().any()}")
    
    try:
        # Create environment
        print("\nCreating environment...")
        env = FuturesTradingEnv(
            df=df,
            window_size=30,
            initial_equity=10000.0,
            reward_config=TREND_RIDER_CONFIG
        )
        
        print("Environment created successfully")
        
        # Try to reset
        print("\nResetting environment...")
        result = env.reset()
        
        # Handle both old and new gym API
        if isinstance(result, tuple):
            obs, info = result
            print(f"Reset returned tuple (new API)")
        else:
            obs = result
            info = {}
            print(f"Reset returned observation only (old API)")
            
        print(f"Reset successful, observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        print(f"Observation type: {type(obs)}")
        if isinstance(obs, np.ndarray):
            print(f"Observation min: {obs.min()}, max: {obs.max()}")
        
        # Check internal state
        print(f"\nChecking internal data after reset:")
        print(f"Current step: {env.current_step}")
        print(f"Data shape: {env.df.shape}")
        
        # Try a step
        print("\nTrying a step...")
        action = np.array([0, 1.0, 0.01])  # HOLD action
        step_result = env.step(action)
        
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            truncated = False
        else:
            obs, reward, done, truncated, info = step_result
            
        print(f"Step successful, reward: {reward}")
        
        # Now test with the wrapped environment
        print("\n\nTesting with wrapped environment...")
        wrapped_env = wrap_environment_for_algorithm(env, algo_type='PPO')
        print("Wrapped environment created")
        
        # Reset wrapped environment
        wrapped_result = wrapped_env.reset()
        if isinstance(wrapped_result, tuple):
            wrapped_obs, _ = wrapped_result
        else:
            wrapped_obs = wrapped_result
            
        print(f"Wrapped reset successful, observation shape: {wrapped_obs.shape}")
        
        # Try prediction with a dummy model
        print("\nTesting model prediction...")
        model = PPO.load("models/best_model.zip")
        
        # Get action from model
        action, _ = model.predict(wrapped_obs, deterministic=True)
        print(f"Model prediction successful, action: {action}")
        
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to identify the exact line
        if 'env' in locals() and hasattr(env, 'df'):
            print(f"\nDataFrame info:")
            print(env.df.info())
            
            # Check for None values in calculated columns
            for col in env.df.columns:
                null_count = env.df[col].isnull().sum()
                if null_count > 0:
                    print(f"Column '{col}' has {null_count} null values")

if __name__ == "__main__":
    test_environment_creation()