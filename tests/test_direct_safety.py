#!/usr/bin/env python3
"""
Direct Safety Intervention Test
Forces extreme conditions to trigger safety penalties.
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def test_safety_triggers():
    """Directly test if safety mechanisms can be triggered"""
    print("🔬 DIRECT SAFETY INTERVENTION TESTING")
    print("=" * 60)
    
    # Load data
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    df = pd.read_csv(data_file)
    
    # Create environment with very high leverage to force triggers
    env = FuturesTradingEnv(df=df, initial_equity=1000.0, max_leverage=100.0)  # Lower equity, higher max leverage
    state = env.reset()
    
    print(f"Environment: ${env.equity:.2f} equity, {env.max_leverage}x max leverage")
    
    # Get current price to calculate extreme positions
    current_price = env._safe_get_price_data(env.current_step, 'close')
    print(f"Current BTC price: ${current_price:.2f}")
    
    # Test extreme leverage that should trigger safety systems
    extreme_leverage = 150.0  # Way beyond max_leverage
    
    print(f"\n🚨 Testing {extreme_leverage}x leverage (beyond {env.max_leverage}x limit):")
    
    # Calculate what this would request
    risk_percentage = 1.0
    risk_equity = env.equity * risk_percentage
    requested_position_value = extreme_leverage * risk_equity
    max_safe_position_value = env.equity * env.max_leverage * 0.5  # Our safety limit
    
    print(f"   Requested position value: ${requested_position_value:,.2f}")
    print(f"   Max safe position value: ${max_safe_position_value:,.2f}")
    print(f"   Excess: ${requested_position_value - max_safe_position_value:,.2f}")
    
    if requested_position_value > max_safe_position_value:
        print("   ✅ Should trigger POSITION_SIZE_LIMITED")
    
    # Execute the extreme action
    action = np.array([extreme_leverage], dtype=np.float32)
    state, reward, done, truncated, info = env.step(action)
    
    safety_penalty = info.get('safety_intervention_penalty', 0.0)
    print(f"\n📊 Results:")
    print(f"   Safety penalty: -{safety_penalty:.4f}")
    print(f"   Total reward: {reward:.4f}")
    print(f"   Position change: {env.position_size:.6f} BTC")
    
    if safety_penalty > 0:
        print("   ✅ Safety penalty applied successfully!")
    else:
        print("   ❌ No safety penalty - investigating...")
        
        # Debug the safety calculation
        print(f"\n🔍 Debug info:")
        print(f"   Current equity: ${env.equity:.2f}")
        print(f"   Max leverage: {env.max_leverage}x")
        print(f"   Action leverage: {extreme_leverage}x")
        
        # Check if leverage is actually being clipped before our safety checks
        clipped_leverage = np.clip(extreme_leverage, -env.max_leverage, env.max_leverage)
        print(f"   Clipped leverage: {clipped_leverage}x")
        
        if clipped_leverage != extreme_leverage:
            print("   🔍 Leverage was clipped before safety checks!")
            print("   💡 Need to add penalty for requesting beyond max_leverage")

if __name__ == "__main__":
    test_safety_triggers()
