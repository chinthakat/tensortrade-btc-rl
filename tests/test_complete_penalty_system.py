#!/usr/bin/env python3
"""
Complete Safety Penalty System Test
Demonstrates how the agent is now penalized for extreme actions.
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
import logging

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def test_complete_penalty_system():
    """Test the complete penalty system for extreme actions"""
    print("🎯 COMPLETE SAFETY PENALTY SYSTEM TEST")
    print("=" * 60)
    
    # Load data
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    df = pd.read_csv(data_file)
    
    # Test different leverage scenarios
    scenarios = [
        ("Conservative", 5.0, "No penalty expected"),
        ("Reasonable", 15.0, "No penalty expected"),
        ("At limit", 25.0, "No penalty expected"),
        ("Slightly over", 30.0, "Small penalty expected"),
        ("Moderate excess", 50.0, "Moderate penalty expected"),
        ("Large excess", 100.0, "Large penalty expected"),
        ("Extreme excess", 200.0, "Maximum penalty expected")
    ]
    
    print("📊 PENALTY ANALYSIS:")
    print("Leverage | Reward   | Extreme Penalty | Safety Penalty | Total Penalty")
    print("-" * 75)
    
    results = []
    
    for scenario_name, leverage, expectation in scenarios:
        # Create fresh environment for each test
        env = FuturesTradingEnv(df=df, initial_equity=10000.0, max_leverage=25.0)
        state = env.reset()
        
        # Execute action
        action = np.array([leverage], dtype=np.float32)
        state, reward, done, truncated, info = env.step(action)
        
        # Extract penalty information
        extreme_penalty = info.get('extreme_leverage_penalty', 0.0)
        safety_penalty = info.get('safety_intervention_penalty', 0.0)
        total_penalty = extreme_penalty + safety_penalty
        
        results.append({
            'scenario': scenario_name,
            'leverage': leverage,
            'reward': reward,
            'extreme_penalty': extreme_penalty,
            'safety_penalty': safety_penalty,
            'total_penalty': total_penalty,
            'expectation': expectation
        })
        
        print(f"{leverage:7.1f}x | {reward:7.4f} | {extreme_penalty:14.4f} | {safety_penalty:13.4f} | {total_penalty:12.4f}")
    
    print("\n🎯 LEARNING SIGNAL ANALYSIS:")
    print("=" * 60)
    
    # Analyze the results
    no_penalty = [r for r in results if r['total_penalty'] == 0]
    small_penalty = [r for r in results if 0 < r['total_penalty'] <= 0.1]
    large_penalty = [r for r in results if r['total_penalty'] > 0.1]
    
    print(f"✅ No penalty (reasonable actions): {len(no_penalty)} scenarios")
    for r in no_penalty:
        print(f"   - {r['leverage']:.1f}x leverage: No penalty (good)")
    
    print(f"\n⚠️  Small penalty (mild excess): {len(small_penalty)} scenarios")
    for r in small_penalty:
        print(f"   - {r['leverage']:.1f}x leverage: -{r['total_penalty']:.4f} penalty")
    
    print(f"\n🚨 Large penalty (extreme excess): {len(large_penalty)} scenarios")
    for r in large_penalty:
        print(f"   - {r['leverage']:.1f}x leverage: -{r['total_penalty']:.4f} penalty")
    
    print(f"\n📚 AGENT LEARNING OUTCOMES:")
    print("✅ The agent now learns that:")
    print("   • Reasonable leverage (≤25x) has no penalty")
    print("   • Excessive leverage requests are immediately punished")
    print("   • Extreme leverage (100x+) receives severe penalties")
    print("   • The penalty scales with the severity of the excess")
    print("   • Optimal policy should stay within reasonable bounds")
    
    print(f"\n🎲 REINFORCEMENT LEARNING IMPACT:")
    print("✅ Clear gradient signals:")
    penalty_gradient = [r['total_penalty'] for r in results]
    for i in range(1, len(penalty_gradient)):
        if penalty_gradient[i] >= penalty_gradient[i-1]:
            print(f"   • {results[i-1]['leverage']:.0f}x → {results[i]['leverage']:.0f}x: Penalty increases ✅")
        else:
            print(f"   • {results[i-1]['leverage']:.0f}x → {results[i]['leverage']:.0f}x: Penalty decreases ❓")

if __name__ == "__main__":
    test_complete_penalty_system()
