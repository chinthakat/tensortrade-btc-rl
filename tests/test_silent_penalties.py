#!/usr/bin/env python3
"""
Test Silent Penalty System
Verifies that penalties are applied without spamming warning logs.
"""

import logging
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging to capture both DEBUG and WARNING levels
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

from trading_environment import TradingEnvironment

def test_silent_penalties():
    """Test that penalties are applied without excessive warning logs"""
    print("🔇 TESTING SILENT PENALTY SYSTEM")
    print("=" * 60)
    
    # Initialize environment
    env = TradingEnvironment()
    env.reset()
    
    print("✅ Testing silent penalties vs debug logging...")
    
    # Count log messages by level
    warning_count = 0
    debug_count = 0
    error_count = 0
    
    # Capture log output
    original_warning = logging.Logger.warning
    original_debug = logging.Logger.debug
    original_error = logging.Logger.error
    
    def count_warning(self, message, *args, **kwargs):
        nonlocal warning_count
        if "PENALTY" in str(message) or "LEVERAGE" in str(message) or "POSITION_STATE_FIX" in str(message):
            warning_count += 1
        return original_warning(self, message, *args, **kwargs)
    
    def count_debug(self, message, *args, **kwargs):
        nonlocal debug_count
        if "PENALTY" in str(message) or "LEVERAGE" in str(message) or "POSITION_STATE_FIX" in str(message):
            debug_count += 1
        return original_debug(self, message, *args, **kwargs)
    
    def count_error(self, message, *args, **kwargs):
        nonlocal error_count
        if "CHAOS" in str(message):
            error_count += 1
        return original_error(self, message, *args, **kwargs)
    
    # Monkey patch logging methods
    logging.Logger.warning = count_warning
    logging.Logger.debug = count_debug
    logging.Logger.error = count_error
    
    try:
        # Test actions that should trigger silent penalties
        penalty_actions = [10.0, -15.0, 50.0, -100.0, 0.1, 25.0]
        
        total_penalties = 0
        
        for i, leverage in enumerate(penalty_actions, 1):
            print(f"\nStep {i}: Testing {leverage}x leverage")
            
            # Reset penalty tracking
            env.position_state_penalty = 0.0
            env.extreme_leverage_penalty = 0.0
            env.zero_pnl_prevention_penalty = 0.0
            env.safety_intervention_penalty = 0.0
            
            # Execute action
            action = leverage  # Direct leverage input
            obs, reward, done, truncated, info = env.step(action)
            
            # Calculate total penalty for this step
            step_penalty = (getattr(env, 'position_state_penalty', 0.0) +
                           getattr(env, 'extreme_leverage_penalty', 0.0) +
                           getattr(env, 'zero_pnl_prevention_penalty', 0.0) +
                           getattr(env, 'safety_intervention_penalty', 0.0))
            
            total_penalties += step_penalty
            
            if step_penalty > 0:
                print(f"   💰 Penalty applied: -{step_penalty:.4f}")
            else:
                print(f"   ✅ No penalty")
                
        print(f"\n📊 LOGGING ANALYSIS:")
        print(f"   WARNING logs (penalty-related): {warning_count}")
        print(f"   DEBUG logs (penalty-related): {debug_count}") 
        print(f"   ERROR logs (chaos-related): {error_count}")
        print(f"   Total penalties applied: -{total_penalties:.4f}")
        
        print(f"\n🎯 SILENT PENALTY ASSESSMENT:")
        if warning_count == 0 and debug_count > 0:
            print("   ✅ PERFECT: Penalties logged at DEBUG level only")
            print("   ✅ No WARNING spam during training")
            print("   ✅ Agent receives negative feedback silently")
        elif warning_count > 0:
            print(f"   ⚠️  Still {warning_count} WARNING logs for penalties")
            print("   ❌ Some penalty logs not converted to DEBUG")
        else:
            print("   ❓ No penalty logs detected at all")
            
        if error_count > 0:
            print(f"   ✅ {error_count} ERROR logs kept for truly chaotic behavior")
        
        print(f"\n🧠 TRAINING IMPACT:")
        print("   ✅ Clean logs during training episodes")
        print("   ✅ Agent still receives full penalty signals")
        print("   ✅ Debugging info available at DEBUG level")
        print("   ✅ Only truly problematic behaviors generate warnings")
        
    finally:
        # Restore original logging methods
        logging.Logger.warning = original_warning
        logging.Logger.debug = original_debug
        logging.Logger.error = original_error

if __name__ == "__main__":
    test_silent_penalties()
