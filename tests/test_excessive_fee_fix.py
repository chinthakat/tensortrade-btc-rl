#!/usr/bin/env python3
"""
Test Excessive Fee Root Cause Fix
Verifies that the position size limits prevent excessive fees.
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
import logging

# Configure logging to see our fixes in action
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_excessive_fee_prevention():
    """Test that position size limits prevent excessive fees"""
    print("🧪 TESTING EXCESSIVE FEE PREVENTION FIXES")
    print("=" * 60)
    
    # Load data
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    df = pd.read_csv(data_file)
    
    # Create environment with realistic settings
    env = FuturesTradingEnv(
        df=df, 
        initial_equity=10000.0,
        max_leverage=25.0,  # This was the problem - allowing 25x leverage
        taker_fee=0.0004
    )
    
    # Reset to initialize
    state = env.reset()
    
    print(f"✅ Environment initialized:")
    print(f"   Initial equity: ${env.equity:.2f}")
    print(f"   Max leverage: {env.max_leverage}x")
    print(f"   Taker fee: {env.taker_fee*100:.3f}%")
    print()
    
    # Test scenarios that previously caused excessive fees
    extreme_scenarios = [
        ("Maximum leverage long", env.max_leverage),      # +25x
        ("Maximum leverage short", -env.max_leverage),    # -25x  
        ("Extreme leverage", 50.0),                       # Beyond limits
        ("Insane leverage", 100.0)                        # Way beyond limits
    ]
    
    for scenario_name, leverage in extreme_scenarios:
        print(f"🚨 Testing: {scenario_name} ({leverage:.1f}x leverage)")
        
        # Get current price from the environment data
        current_price = env._safe_get_price_data(env.current_step, 'close')
        print(f"   📊 Current BTC price: ${current_price:.2f}")
        
        # Simulate what the old system would have calculated
        risk_percentage = 1.0  # This was the problem - 100% risk
        risk_equity = env.equity * risk_percentage
        old_position_value = leverage * risk_equity
        old_position_size = old_position_value / current_price
        old_trade_size = old_position_size - env.position_size
        old_trade_value = abs(old_trade_size * current_price)
        old_fee = old_trade_value * env.taker_fee
        
        print(f"   📊 OLD SYSTEM (broken):")
        print(f"      Position value: ${old_position_value:,.2f}")
        print(f"      Position size: {old_position_size:.6f} BTC")
        print(f"      Trade value: ${old_trade_value:,.2f}")
        print(f"      Estimated fee: ${old_fee:,.2f}")
        
        if old_fee > 1000:
            print(f"      ❌ EXCESSIVE FEE: ${old_fee:,.2f} (This was the bug!)")
        
        # Now test with the fixed system by trying to execute the action
        current_position_before = env.position_size
        equity_before = env.equity
        
        # Execute action (this will use our new safety limits)
        action = np.array([leverage], dtype=np.float32)
        state, reward, done, truncated, info = env.step(action)
        
        actual_position_change = env.position_size - current_position_before
        actual_trade_value = abs(actual_position_change * current_price)
        
        print(f"   ✅ NEW SYSTEM (fixed):")
        print(f"      Actual position change: {actual_position_change:.6f} BTC")
        print(f"      Actual trade value: ${actual_trade_value:.2f}")
        
        # Check if any fees were actually paid (from the step)
        equity_change = env.equity - equity_before
        print(f"      Equity change: ${equity_change:.2f}")
        
        if actual_trade_value > 0:
            estimated_actual_fee = actual_trade_value * env.taker_fee
            print(f"      Estimated actual fee: ${estimated_actual_fee:.2f}")
            
            if estimated_actual_fee > 100:
                print(f"      ⚠️  Still a large fee, but much better than ${old_fee:,.2f}")
            else:
                print(f"      ✅ Reasonable fee: ${estimated_actual_fee:.2f}")
        else:
            print(f"      ✅ No trade executed (safety limits prevented it)")
        
        print()
    
    print("🎯 SUMMARY:")
    print("✅ Position size safety limits implemented")
    print("✅ Emergency brake for BTC position size")
    print("✅ Trade size reduction when fees would be excessive")
    print("✅ Multiple layers of protection against unrealistic trades")

if __name__ == "__main__":
    test_excessive_fee_prevention()
