#!/usr/bin/env python3
"""
Live Trading Environment Silent Penalty Test
Tests the actual trading environment for silent penalty implementation.
"""

import logging
import sys
import os
import numpy as np
from io import StringIO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_actual_trading_environment():
    """Test the actual trading environment with silent penalties"""
    print("🎯 TESTING ACTUAL TRADING ENVIRONMENT SILENT PENALTIES")
    print("=" * 65)
    
    # Configure logging to capture different levels
    log_capture = StringIO()
    
    # Set up logging to capture both WARNING and DEBUG
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True,
        handlers=[
            logging.StreamHandler(log_capture),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        from trading_environment import TradingEnvironment
        
        # Initialize environment
        print("✅ Initializing trading environment...")
        env = TradingEnvironment()
        env.reset()
        
        print("\n🔍 Testing penalty scenarios...")
        
        # Test scenarios that should trigger penalties
        test_scenarios = [
            ("Extreme leverage (50x)", 50.0),
            ("Negative extreme leverage (-75x)", -75.0),
            ("Small leverage (0.1x)", 0.1),
            ("Normal leverage (15x)", 15.0),
            ("Very extreme leverage (200x)", 200.0)
        ]
        
        warning_count = 0
        debug_count = 0
        penalty_count = 0
        
        for scenario_name, leverage in test_scenarios:
            print(f"\n   Testing: {scenario_name}")
            
            # Clear log capture for this test
            log_capture.seek(0)
            log_capture.truncate(0)
            
            # Reset penalty tracking
            env.position_state_penalty = 0.0
            env.extreme_leverage_penalty = 0.0
            env.zero_pnl_prevention_penalty = 0.0
            env.safety_intervention_penalty = 0.0
            
            # Execute action that should trigger penalties
            try:
                obs, reward, done, truncated, info = env.step(leverage)
                
                # Check if penalties were applied
                total_penalty = (getattr(env, 'position_state_penalty', 0.0) +
                               getattr(env, 'extreme_leverage_penalty', 0.0) +
                               getattr(env, 'zero_pnl_prevention_penalty', 0.0) +
                               getattr(env, 'safety_intervention_penalty', 0.0))
                
                if total_penalty > 0:
                    penalty_count += 1
                    print(f"      💰 Penalty applied: -{total_penalty:.4f}")
                else:
                    print(f"      ✅ No penalty")
                
                # Check log content for this scenario
                log_content = log_capture.getvalue()
                scenario_warnings = log_content.count("WARNING")
                scenario_debugs = log_content.count("DEBUG")
                
                # Count penalty-related logs specifically
                penalty_warnings = (log_content.count("WARNING - EXTREME_LEVERAGE") +
                                  log_content.count("WARNING - POSITION_STATE") +
                                  log_content.count("WARNING - ZERO_PNL") +
                                  log_content.count("WARNING - SAFETY"))
                
                penalty_debugs = (log_content.count("DEBUG - EXTREME_LEVERAGE") +
                                log_content.count("DEBUG - POSITION_STATE") +
                                log_content.count("DEBUG - ZERO_PNL") +
                                log_content.count("DEBUG - SAFETY"))
                
                warning_count += penalty_warnings
                debug_count += penalty_debugs
                
                if total_penalty > 0 and penalty_warnings > 0:
                    print(f"      ⚠️  {penalty_warnings} WARNING logs (should be DEBUG!)")
                elif total_penalty > 0 and penalty_debugs > 0:
                    print(f"      ✅ {penalty_debugs} DEBUG logs (silent penalties working!)")
                elif total_penalty > 0:
                    print(f"      🔇 Silent penalty (no logs)")
                    
            except Exception as e:
                print(f"      ❌ Error in scenario: {e}")
        
        print(f"\n📊 SILENT PENALTY ANALYSIS:")
        print(f"   Scenarios tested: {len(test_scenarios)}")
        print(f"   Penalties triggered: {penalty_count}")
        print(f"   WARNING penalty logs: {warning_count}")
        print(f"   DEBUG penalty logs: {debug_count}")
        
        print(f"\n🎯 ASSESSMENT:")
        if warning_count == 0 and penalty_count > 0:
            print("   ✅ PERFECT: Silent penalties working!")
            print("   ✅ No WARNING spam during penalties")
            print("   ✅ Agent receives penalties without log noise")
            print("   ✅ Clean training experience")
        elif warning_count > 0:
            print(f"   ⚠️  {warning_count} WARNING logs still present")
            print("   ❌ Some penalties not converted to silent mode")
            print("   💡 May need additional logging level adjustments")
        else:
            print("   ❓ No penalties triggered in test scenarios")
            print("   💡 May need to adjust test scenarios")
        
        print(f"\n🚀 TRAINING READINESS:")
        if warning_count == 0:
            print("   ✅ Ready for clean, professional training")
            print("   ✅ Logs will be focused on important events")
            print("   ✅ Penalty system fully functional")
        else:
            print("   ⚠️  May have some log noise during training")
            print("   💡 Consider setting log level to INFO for training")
            
    except ImportError as e:
        print(f"❌ Could not import trading environment: {e}")
        print("💡 Make sure you're in the correct environment")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_actual_trading_environment()
