"""
Test the dust position filter functionality
"""
import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv

# Load test data
df = pd.read_csv('data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv')

# Create environment
env = FuturesTradingEnv(
    df=df,
    initial_equity=1000.0,  # Smaller equity for easier dust testing
    max_leverage=10.0,
    taker_fee=0.0004,
    enable_funding_costs=False,
    use_advanced_action_space=False,
    window_size=30
)

# Reset environment
obs, info = env.reset()

print("Testing Dust Position Filter")
print("=" * 50)

# Test 1: Try to create a dust position (very small leverage)
print("\n1. Testing dust position creation (leverage=0.005):")
action = np.array([0.005])  # Very small leverage
obs, reward, done, truncated, info = env.step(action)

print(f"Position size: {env.position_size:.8f} BTC")
print(f"Position value: ${env.position_size * env.current_price:.2f}")
print(f"Dust penalty: {getattr(env, 'dust_position_penalty', 0.0):.6f}")
print(f"Reward: {reward:.6f}")

# Test 2: Try normal position
print("\n2. Testing normal position creation (leverage=1.0):")
env.reset()  # Reset for clean test
action = np.array([1.0])  # Normal leverage
obs, reward, done, truncated, info = env.step(action)

print(f"Position size: {env.position_size:.8f} BTC")
print(f"Position value: ${env.position_size * env.current_price:.2f}")
print(f"Dust penalty: {getattr(env, 'dust_position_penalty', 0.0):.6f}")
print(f"Reward: {reward:.6f}")

# Test 3: Monitor position sizes over multiple steps
print("\n3. Testing multiple small positions:")
env.reset()
dust_positions = 0
normal_positions = 0

for i in range(10):
    # Alternate between very small and normal actions
    if i % 2 == 0:
        action = np.array([0.002])  # Dust-level
    else:
        action = np.array([0.5])    # Normal
    
    obs, reward, done, truncated, info = env.step(action)
    
    if abs(env.position_size) > 0:
        position_value = abs(env.position_size * env.current_price)
        if position_value < 20:  # Under $20
            dust_positions += 1
            print(f"Step {i+1}: DUST - Position: {env.position_size:.8f} BTC (${position_value:.2f})")
        else:
            normal_positions += 1
            print(f"Step {i+1}: NORMAL - Position: {env.position_size:.8f} BTC (${position_value:.2f})")
    else:
        print(f"Step {i+1}: NO POSITION")

print(f"\nSummary:")
print(f"Dust positions created: {dust_positions}")
print(f"Normal positions created: {normal_positions}")
print(f"Filter effectiveness: {((10 - dust_positions) / 10) * 100:.1f}%")

print("\nDust position filter test completed!")
