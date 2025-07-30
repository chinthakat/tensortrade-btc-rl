#!/usr/bin/env python3
"""
Excessive Fee Diagnostic Script
Identifies the root cause of unrealistic trading fees in the system.
"""

import sys
import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
import logging

# Configure logging to capture detailed fee information
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_fee_scenarios():
    """Test various trading scenarios to identify excessive fee causes"""
    print("🔍 DIAGNOSING EXCESSIVE FEE ROOT CAUSES")
    print("=" * 60)
    
    # Load data first
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded data: {len(df)} rows")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Create environment for testing
    env = FuturesTradingEnv(df=df, initial_equity=10000.0)
    
    # Reset to initialize
    state = env.reset()
    
    # Test scenarios that might cause excessive fees
    scenarios = [
        ("Normal small trade", 0.1),
        ("Normal medium trade", 0.5), 
        ("Large position", 1.0),
        ("Extreme position", 2.0),
        ("Position flip", -1.0),
        ("Tiny position", 0.001),
        ("Very large position", 5.0)
    ]
    
    print(f"Initial state: equity=${env.equity:.2f}, BTC=${env.btc_balance:.6f}")
    print(f"Current price: ${env.current_price:.2f}")
    print(f"Taker fee rate: {env.taker_fee*100:.3f}%")
    print()
    
    for scenario_name, target_position in scenarios:
        print(f"🧪 Testing: {scenario_name} (target position: {target_position})")
        
        # Calculate what the trade would be
        current_position = env.position_size
        trade_size = target_position - current_position
        trade_value = abs(trade_size * env.current_price)
        expected_fee = trade_value * env.taker_fee
        
        print(f"  Current position: {current_position:.6f}")
        print(f"  Target position: {target_position:.6f}")
        print(f"  Trade size: {trade_size:.6f}")
        print(f"  Trade value: ${trade_value:.2f}")
        print(f"  Expected fee: ${expected_fee:.2f}")
        
        # Check if this would trigger excessive fees
        if expected_fee > 1000:
            print(f"  ❌ EXCESSIVE FEE DETECTED! ${expected_fee:.2f}")
            print(f"     - Fee rate: {(expected_fee/trade_value)*100:.2f}%")
            print(f"     - This is the problem!")
            
            # Investigate further
            if abs(trade_size) > 10:
                print(f"     - ROOT CAUSE: Unrealistic trade size {trade_size:.6f}")
            if trade_value > env.equity * 10:
                print(f"     - ROOT CAUSE: Trade value ${trade_value:.2f} >> equity ${env.equity:.2f}")
            if env.taker_fee > 0.01:
                print(f"     - ROOT CAUSE: Excessive taker fee rate {env.taker_fee*100:.2f}%")
        elif expected_fee > 100:
            print(f"  ⚠️  High fee: ${expected_fee:.2f}")
        else:
            print(f"  ✅ Reasonable fee: ${expected_fee:.2f}")
        
        print()

def investigate_problematic_trades():
    """Investigate the specific trades that caused issues"""
    print("🔍 INVESTIGATING PROBLEMATIC TRADES FROM USER DATA")
    print("=" * 60)
    
    # From the user's verify_trades.py data
    problematic_trades = [
        {'id': 'TRADE_02571', 'fees': 3200.59, 'reason': 'CANCEL_ACTION'},
        {'id': 'TRADE_02622', 'fees': 3283.74, 'reason': 'CANCEL_ACTION'},
        {'id': 'TRADE_03056', 'fees': 3640.81, 'reason': 'CANCEL_ACTION'},
        {'id': 'TRADE_00441', 'fees': 782.51, 'reason': 'CANCEL_ACTION'},
        {'id': 'TRADE_00438', 'fees': 778.43, 'reason': 'CANCEL_ACTION'}
    ]
    
    total_excessive_fees = sum(trade['fees'] for trade in problematic_trades)
    print(f"Total excessive fees: ${total_excessive_fees:.2f}")
    print(f"Average excessive fee: ${total_excessive_fees/len(problematic_trades):.2f}")
    print()
    
    # What would cause $3000+ fees?
    print("💡 REVERSE ENGINEERING EXCESSIVE FEES:")
    btc_price = 45000  # Approximate BTC price
    taker_fee = 0.001  # 0.1% taker fee
    
    for trade in problematic_trades:
        required_trade_value = trade['fees'] / taker_fee
        required_btc_amount = required_trade_value / btc_price
        
        print(f"{trade['id']}: ${trade['fees']:.2f} fee")
        print(f"  Required trade value: ${required_trade_value:.2f}")
        print(f"  Required BTC amount: {required_btc_amount:.6f}")
        print(f"  Reason: {trade['reason']}")
        
        if required_btc_amount > 100:
            print(f"  ❌ UNREALISTIC: Trading {required_btc_amount:.2f} BTC (>100 BTC)")
        elif required_btc_amount > 10:
            print(f"  ⚠️  VERY HIGH: Trading {required_btc_amount:.2f} BTC")
        else:
            print(f"  ✅ Reasonable: Trading {required_btc_amount:.6f} BTC")
        print()

def check_environment_parameters():
    """Check if environment parameters are causing issues"""
    print("🔍 CHECKING ENVIRONMENT PARAMETERS")
    print("=" * 60)
    
    # Load data first
    data_file = 'data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv'
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded data: {len(df)} rows")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    env = FuturesTradingEnv(df=df, initial_equity=10000.0)
    
    print(f"Taker fee rate: {env.taker_fee*100:.3f}%")
    print(f"Initial equity: ${env.initial_equity:.2f}")
    print(f"Max leverage: {env.max_leverage}x")
    
    # Check if taker fee is reasonable
    if env.taker_fee > 0.01:  # >1%
        print(f"❌ EXCESSIVE TAKER FEE: {env.taker_fee*100:.2f}% is unrealistic")
    elif env.taker_fee > 0.005:  # >0.5%
        print(f"⚠️  HIGH TAKER FEE: {env.taker_fee*100:.2f}% is higher than typical")
    else:
        print(f"✅ Reasonable taker fee: {env.taker_fee*100:.3f}%")
    
    # Check initial equity
    if env.initial_equity < 1000:
        print(f"⚠️  Low initial equity: ${env.initial_equity:.2f} - fees will be disproportionate")
    else:
        print(f"✅ Reasonable initial equity: ${env.initial_equity:.2f}")
    
    # Check leverage - high leverage can cause massive positions
    if env.max_leverage > 50:
        print(f"❌ EXCESSIVE LEVERAGE: {env.max_leverage}x can cause massive positions")
    elif env.max_leverage > 20:
        print(f"⚠️  HIGH LEVERAGE: {env.max_leverage}x increases risk of large trades")
    else:
        print(f"✅ Reasonable leverage: {env.max_leverage}x")

def run_diagnostics():
    """Run comprehensive fee diagnostics"""
    try:
        check_environment_parameters()
        print()
        investigate_problematic_trades()
        print()
        test_fee_scenarios()
        
        print("🎯 RECOMMENDED ACTIONS:")
        print("1. Check if action space allows unrealistic position sizes")
        print("2. Verify position size calculations are bounded properly")
        print("3. Look for phantom trades that execute massive positions")
        print("4. Check if CANCEL_ACTION is executing trades instead of canceling")
        print("5. Verify price data doesn't have extreme outliers")
        
    except Exception as e:
        print(f"❌ Error during diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostics()
