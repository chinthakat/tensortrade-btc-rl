"""Test the exact environment creation used in binance_integration.py"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from trading_environment import FuturesTradingEnv
from action_space_wrapper import wrap_environment_for_algorithm
from improved_reward_configs import TREND_RIDER_CONFIG
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_binance_environment():
    """Test environment creation exactly as in binance_integration.py"""
    
    # Create data similar to what binance_integration uses
    data = []
    base_price = 50000
    
    for i in range(100):
        timestamp = 1735689600 + i * 900  # 15-minute intervals
        price = base_price + np.random.uniform(-100, 100)
        
        data.append({
            'timestamp': timestamp,
            'open': price,
            'high': price + np.random.uniform(0, 50),
            'low': price - np.random.uniform(0, 50),
            'close': price + np.random.uniform(-25, 25),
            'volume': np.random.uniform(100, 1000)
        })
    
    df = pd.DataFrame(data)
    
    # Prepare model input (similar to prepare_model_input method)
    market_data = df.tail(30).copy()
    
    print(f"Market data shape: {market_data.shape}")
    print(f"Market data columns: {market_data.columns.tolist()}")
    
    # Get balance (simulate)
    balance = 1000.0
    
    # Create environment exactly as in get_model_signal
    try:
        print("\nCreating environment...")
        env = FuturesTradingEnv(
            df=market_data,
            window_size=30,
            initial_equity=balance,
            reward_config=TREND_RIDER_CONFIG
        )
        
        print("Environment created successfully")
        
        # Wrap for PPO
        print("\nWrapping environment...")
        env = wrap_environment_for_algorithm(env, algo_type='PPO')
        print("Environment wrapped successfully")
        
        # Reset
        print("\nResetting environment...")
        reset_result = env.reset()
        
        if isinstance(reset_result, tuple):
            obs, info = reset_result
            print("Reset returned tuple")
        else:
            obs = reset_result
            print("Reset returned observation only")
            
        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")
        
        # Test model prediction
        print("\nLoading model...")
        model = PPO.load("models/best_model.zip")
        print("Model loaded")
        
        print("\nGetting prediction...")
        action, _ = model.predict(obs, deterministic=True)
        print(f"Prediction successful! Action: {action}")
        
        # Parse action
        print("\nParsing action...")
        action_type = int(action[0])
        leverage = float(action[1])
        position_size = float(action[2])
        
        print(f"Action type: {action_type}")
        print(f"Leverage: {leverage}")
        print(f"Position size: {position_size}")
        
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_binance_environment()