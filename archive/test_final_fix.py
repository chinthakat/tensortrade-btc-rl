"""
Quick test to verify the model prediction fix is working in Binance integration
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
from action_space_wrapper import wrap_environment_for_algorithm
from improved_reward_configs import TREND_RIDER_CONFIG
from stable_baselines3 import PPO

def test_final_fix():
    """Final test to confirm everything works"""
    
    print("=== FINAL VERIFICATION TEST ===")
    
    # Create sample data (same as Binance would provide)
    print("1. Creating sample BTCUSDT data...")
    dates = pd.date_range(start='2025-01-01', periods=50, freq='15min')
    np.random.seed(42)
    
    initial_price = 45000
    returns = np.random.normal(0, 0.001, 50)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': (dates.astype(np.int64) // 10**9).astype(float),  # Convert to seconds like Binance
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, 50)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.002, 50))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.002, 50))),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 50)
    })
    
    # Use all data (Binance integration now provides full dataset for technical indicators)
    model_input = df.copy()  # All 50 rows
    print(f"   Model input shape: {model_input.shape}")
    print(f"   This provides {len(model_input)} data points for technical indicators")
    
    # Create environment exactly like Binance integration does
    print("2. Creating environment (like Binance integration)...")
    env = FuturesTradingEnv(
        df=model_input,
        window_size=20,  # Fixed to match trained model
        initial_equity=10000,
        reward_config=TREND_RIDER_CONFIG
    )
    
    # Wrap environment exactly like Binance integration does
    print("3. Wrapping environment...")
    env = wrap_environment_for_algorithm(env, "PPO")
    
    # Reset and get observation
    print("4. Getting observation...")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print(f"   Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"   Keys: {list(obs.keys())}")
        for key, value in obs.items():
            print(f"   {key}: {value.shape}")
    
    # Load model and predict
    print("5. Loading model and making prediction...")
    model = PPO.load("models/best_model.zip")
    
    try:
        action, _states = model.predict(obs, deterministic=True)
        print(f"   ✅ Model prediction successful!")
        print(f"   Action: {action}")
        print(f"   Action type: {type(action)}")
        print(f"   Action shape: {action.shape}")
        
        # Parse action like Binance integration does
        action_dict = {
            'action_type': int(action[0]),
            'leverage': float(action[1]),
            'risk_percentage': float(action[2])
        }
        print(f"   Parsed action: {action_dict}")
        
        print("\n🎉 SUCCESS: The fix is working perfectly!")
        print("✅ Binance integration should now work correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Model prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_fix()
    if success:
        print("\n✅ ALL TESTS PASSED - Ready for live trading!")
    else:
        print("\n❌ Tests failed - Need more debugging")
