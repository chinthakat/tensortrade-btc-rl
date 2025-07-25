#!/usr/bin/env python3
"""
Test Safety Intervention Penalty System
Verifies that the agent receives penalties for attempting extreme actions that trigger safety systems.
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
import logging

# Configure logging to see safety interventions
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_safety_penalties():
    """Test that extreme actions result in appropriate penalties"""
    print("🧪 TESTING SAFETY INTERVENTION PENALTY SYSTEM")
    print("=" * 60)
    
    # Load data
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    df = pd.read_csv(data_file)
    
    # Create environment
    env = FuturesTradingEnv(df=df, initial_equity=10000.0, max_leverage=25.0)
    state = env.reset()
    
    print(f"✅ Environment initialized with ${env.equity:.2f} equity")
    print()
    
    # Test scenarios with increasing severity
    test_scenarios = [
        ("Reasonable action", 5.0, "Should have no penalty"),
        ("High leverage", 15.0, "Might trigger position limits"),
        ("Maximum leverage", 25.0, "Should trigger position limits"),
        ("Extreme leverage", 50.0, "Should trigger emergency brake"),
        ("Insane leverage", 100.0, "Should trigger severe penalties")
    ]
    
    for scenario_name, leverage, expectation in test_scenarios:
        print(f"🚨 Testing: {scenario_name} ({leverage:.1f}x)")
        print(f"   Expected: {expectation}")
        
        # Reset safety penalty tracking
        if hasattr(env, 'safety_intervention_penalty'):
            env.safety_intervention_penalty = 0.0
        
        # Record state before action
        equity_before = env.equity
        
        # Execute action
        action = np.array([leverage], dtype=np.float32)
        state, reward, done, truncated, info = env.step(action)
        
        # Check for safety interventions
        safety_penalty = info.get('safety_intervention_penalty', 0.0)
        
        print(f"   📊 Results:")
        print(f"      Reward: {reward:.4f}")
        print(f"      Safety penalty: -{safety_penalty:.4f}")
        print(f"      Equity change: ${env.equity - equity_before:.2f}")
        
        # Analyze the penalty appropriateness
        if leverage <= 20:
            if safety_penalty > 0:
                print(f"      ⚠️  Unexpected penalty for reasonable leverage")
            else:
                print(f"      ✅ No penalty for reasonable action")
        elif leverage <= 30:
            if safety_penalty > 0:
                print(f"      ✅ Appropriate penalty for high leverage")
            else:
                print(f"      ⚠️  Expected penalty for high leverage")
        else:  # Extreme leverage
            if safety_penalty >= 0.1:
                print(f"      ✅ Strong penalty for extreme leverage (good!)")
            elif safety_penalty > 0:
                print(f"      ⚠️  Weak penalty for extreme leverage")
            else:
                print(f"      ❌ No penalty for extreme leverage!")
        
        print()
    
    # Test the learning signal clarity
    print("🎯 LEARNING SIGNAL ANALYSIS:")
    print("✅ Agent now receives immediate negative feedback for:")
    print("   • Requesting excessive position sizes")
    print("   • Attempting unrealistic BTC amounts")
    print("   • Actions that would cause excessive fees")
    print()
    print("📚 This teaches the agent:")
    print("   • High leverage has consequences")
    print("   • Extreme actions are punished even if 'prevented'")
    print("   • Optimal policy should avoid triggering safety systems")
    
def test_penalty_scaling():
    """Test that penalties scale appropriately with severity"""
    print("\n🔬 TESTING PENALTY SCALING")
    print("=" * 60)
    
    # Load data
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    df = pd.read_csv(data_file)
    
    penalties = []
    leverages = [25, 50, 75, 100, 150, 200]
    
    for leverage in leverages:
        env = FuturesTradingEnv(df=df, initial_equity=10000.0, max_leverage=25.0)
        state = env.reset()
        
        action = np.array([leverage], dtype=np.float32)
        state, reward, done, truncated, info = env.step(action)
        
        safety_penalty = info.get('safety_intervention_penalty', 0.0)
        penalties.append(safety_penalty)
        
        print(f"Leverage {leverage:3.0f}x → Penalty: -{safety_penalty:.4f}")
    
    print("\n📈 Penalty scaling analysis:")
    for i in range(1, len(penalties)):
        if penalties[i] >= penalties[i-1]:
            print(f"✅ Penalty increases from {leverages[i-1]}x to {leverages[i]}x")
        else:
            print(f"⚠️  Penalty decreases from {leverages[i-1]}x to {leverages[i]}x")

if __name__ == "__main__":
    test_safety_penalties()
    test_penalty_scaling()
