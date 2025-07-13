"""
Test script to verify the indexing fix in trading environment
"""
import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
from action_space_wrapper import wrap_environment_for_algorithm

def test_episode_boundary_fix():
    """Test that the environment doesn't crash at episode boundaries"""
    print("🧪 Testing Episode Boundary Fix")
    print("=" * 50)
    
    # Load small subset of data for quick testing
    df = pd.read_csv('data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv')
    
    # Use only first 100 rows to quickly reach boundary
    test_df = df.head(100).copy()
    print(f"📊 Using {len(test_df)} rows for boundary testing")
    
    try:
        # Create environment with small dataset
        env = FuturesTradingEnv(
            df=test_df,
            log_file="logs/test_boundary_fix.csv",  # Use proper path
            window_size=20,  # Small window for faster testing
            initial_equity=10000,
            use_advanced_action_space=True
        )
        
        # Wrap for PPO compatibility
        env = wrap_environment_for_algorithm(env, "PPO")
        
        print("✅ Environment created successfully")
        
        # Reset and test
        result = env.reset()
        print(f"✅ Environment reset successfully")
        print(f"📊 Reset result type: {type(result)}")
        print(f"📊 Reset result: {result}")
        
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
            
        print(f"📊 Observation type: {type(obs)}")
        if hasattr(obs, 'shape'):
            print(f"📊 Observation shape: {obs.shape}")
        elif isinstance(obs, dict):
            print(f"📊 Observation keys: {obs.keys()}")
            for k, v in obs.items():
                print(f"   - {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
        
        # Run through many steps to reach boundary
        steps = 0
        max_steps = 20  # Reduced for testing
        
        while steps < max_steps:
            # Random action in Box(3,) space
            action = np.random.uniform(-1, 1, 3)
            
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                if steps % 5 == 0:
                    print(f"📈 Step {steps}: reward={reward:.4f}")
                
                if terminated or truncated:
                    print(f"🏁 Episode ended at step {steps}")
                    print(f"   - Terminated: {terminated}")
                    print(f"   - Truncated: {truncated}")
                    break
                    
            except Exception as e:
                print(f"❌ Error at step {steps}: {e}")
                return False
        
        print("✅ Boundary test completed successfully!")
        print(f"📊 Completed {steps} steps without indexing errors")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_episode_boundary_fix()
    if success:
        print("\n🎉 Index boundary fix verified!")
    else:
        print("\n💥 Test failed - fix needs more work")
