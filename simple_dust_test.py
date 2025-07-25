"""
Simple test for dust position filter
"""
print("Starting dust filter test...")

try:
    import pandas as pd
    print("✓ Pandas imported")
    
    import numpy as np
    print("✓ NumPy imported")
    
    from trading_environment import FuturesTradingEnv
    print("✓ Trading environment imported")
    
    # Load minimal data for testing
    df = pd.read_csv('data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv')
    print(f"✓ Data loaded: {len(df)} rows")
    
    # Create environment
    env = FuturesTradingEnv(
        df=df.head(1000),  # Use first 1000 rows only
        initial_equity=1000.0,
        max_leverage=10.0,
        taker_fee=0.0004,
        enable_funding_costs=False,
        use_advanced_action_space=False,
        window_size=30
    )
    print("✓ Environment created")
    
    # Reset and test
    obs, info = env.reset()
    print("✓ Environment reset")
    
    # Test dust position
    print("\n=== DUST POSITION TEST ===")
    action = np.array([0.01])  # Very small leverage
    obs, reward, done, truncated, info = env.step(action)
    
    print(f"Action: {action[0]:.3f}")
    print(f"Position size: {env.position_size:.8f} BTC")
    if hasattr(env, 'current_price'):
        print(f"Position value: ${abs(env.position_size * env.current_price):.2f}")
    print(f"Dust penalty: {getattr(env, 'dust_position_penalty', 0.0):.6f}")
    print(f"Reward: {reward:.6f}")
    
    print("\n✅ Dust filter test completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error in test: {e}")
    import traceback
    traceback.print_exc()
