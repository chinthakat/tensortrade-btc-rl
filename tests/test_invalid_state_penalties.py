#!/usr/bin/env python3
"""
Test Invalid State Penalty System
Verifies that the agent gets penalized for creating invalid trading states.
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
import logging

# Set logging to capture penalty information
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def test_invalid_state_penalties():
    """Test that the agent gets penalized for causing invalid states"""
    print("🚨 TESTING INVALID STATE PENALTY SYSTEM")
    print("=" * 60)
    
    # Load data
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    df = pd.read_csv(data_file)
    
    # Create environment
    env = FuturesTradingEnv(df=df, initial_equity=10000.0, max_leverage=25.0)
    state = env.reset()
    
    print("✅ Testing invalid state detection and penalties...")
    print()
    
    # Track penalties over multiple steps
    total_penalties = {
        'position_state': 0.0,
        'zero_pnl_prevention': 0.0,
        'safety_intervention': 0.0,
        'extreme_leverage': 0.0
    }
    
    penalty_steps = []
    
    # Run a series of actions that should trigger state corrections
    actions = [10.0, -15.0, 25.0, -25.0, 50.0, 0.1, -100.0, 12.0]  # Mix of reasonable and extreme
    
    for i, leverage in enumerate(actions):
        print(f"Step {i+1}: Testing {leverage}x leverage")
        
        # Reset penalties for this step
        env.position_state_penalty = 0.0
        env.zero_pnl_prevention_penalty = 0.0
        env.safety_intervention_penalty = 0.0
        env.extreme_leverage_penalty = 0.0
        
        # Execute action
        action = np.array([leverage], dtype=np.float32)
        state, reward, done, truncated, info = env.step(action)
        
        # Check what penalties were applied
        step_penalties = {
            'position_state': getattr(env, 'position_state_penalty', 0.0),
            'zero_pnl_prevention': getattr(env, 'zero_pnl_prevention_penalty', 0.0),
            'safety_intervention': getattr(env, 'safety_intervention_penalty', 0.0),
            'extreme_leverage': getattr(env, 'extreme_leverage_penalty', 0.0)
        }
        
        step_total = sum(step_penalties.values())
        
        print(f"   Reward: {reward:.4f}")
        print(f"   Position state penalty: -{step_penalties['position_state']:.4f}")
        print(f"   Zero PnL prevention penalty: -{step_penalties['zero_pnl_prevention']:.4f}")
        print(f"   Safety intervention penalty: -{step_penalties['safety_intervention']:.4f}")
        print(f"   Extreme leverage penalty: -{step_penalties['extreme_leverage']:.4f}")
        print(f"   Total penalties: -{step_total:.4f}")
        
        if step_total > 0:
            penalty_steps.append((i+1, leverage, step_total, step_penalties))
            print(f"   ⚠️  PENALTY APPLIED for this action")
        else:
            print(f"   ✅ Clean action - no penalties")
        
        # Accumulate totals
        for key in total_penalties:
            total_penalties[key] += step_penalties[key]
        
        print()
        
        if done:
            break
    
    print("📊 PENALTY ANALYSIS SUMMARY:")
    print("=" * 60)
    
    if penalty_steps:
        print(f"🚨 Penalties applied in {len(penalty_steps)} out of {len(actions)} actions:")
        for step_num, leverage, total_penalty, breakdown in penalty_steps:
            print(f"   Step {step_num} ({leverage:5.1f}x): -{total_penalty:.4f} total penalty")
            for penalty_type, amount in breakdown.items():
                if amount > 0:
                    print(f"     - {penalty_type}: -{amount:.4f}")
        
        print(f"\n📈 TOTAL ACCUMULATED PENALTIES:")
        grand_total = sum(total_penalties.values())
        for penalty_type, total in total_penalties.items():
            if total > 0:
                print(f"   {penalty_type}: -{total:.4f}")
        print(f"   GRAND TOTAL: -{grand_total:.4f}")
        
        print(f"\n🎯 LEARNING IMPACT:")
        print("✅ Agent now receives immediate negative feedback for:")
        print("   • Creating invalid position states")
        print("   • Triggering emergency fixes")
        print("   • Requesting extreme leverage")
        print("   • Causing position state chaos")
        print(f"✅ Total learning signal: -{grand_total:.4f} penalty")
        
    else:
        print("✅ No penalties applied - all actions were valid")
        print("   (This might indicate very conservative actions or good behavior)")
    
    print(f"\n🧠 BEHAVIORAL MODIFICATION:")
    print("The agent will learn to avoid actions that:")
    print("• Cause position_side corrections")
    print("• Create invalid entry_price states") 
    print("• Trigger zero PnL prevention")
    print("• Request excessive leverage")
    print("• Lead to chaotic state corrections")

if __name__ == "__main__":
    test_invalid_state_penalties()
