#!/usr/bin/env python3
"""
Verification script for trade logging fixes
Tests the 4 outstanding issues:
1. Duration calculation consistency
2. PnL attribution only on closure
3. Clear close_reason values  
4. Proper timestamp handling
"""

import pandas as pd
import numpy as np
from trading_environment import TradingEnvironment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_environment():
    """Create a test trading environment with sample data"""
    # Create minimal test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
        'open': 50000 + np.random.randn(100) * 100,
        'high': 50000 + np.random.randn(100) * 100 + 50,
        'low': 50000 + np.random.randn(100) * 100 - 50,
        'close': 50000 + np.random.randn(100) * 100,
        'volume': np.random.uniform(100, 1000, 100),
        'atr': np.random.uniform(50, 200, 100)
    })
    
    # Add technical indicators
    test_data['sma_20'] = test_data['close'].rolling(20).mean()
    test_data['ema_12'] = test_data['close'].ewm(span=12).mean()
    test_data['rsi'] = 50 + np.random.randn(100) * 15  # Mock RSI
    test_data['macd'] = np.random.randn(100) * 10
    test_data['bb_upper'] = test_data['close'] + 100
    test_data['bb_lower'] = test_data['close'] - 100
    test_data['obv'] = np.cumsum(np.random.randn(100) * 1000)
    
    # Fill NaN values
    test_data = test_data.fillna(method='bfill').fillna(method='ffill')
    
    # Save test data
    test_data.to_csv('test_data_verification.csv', index=False)
    
    # Create environment
    env = TradingEnvironment(
        data_file='test_data_verification.csv',
        initial_balance=10000,
        leverage=1.0,
        taker_fee=0.001,
        use_percentage_action=True
    )
    
    return env

def test_duration_fix():
    """Test that trade duration is calculated correctly"""
    print("\n=== Testing Duration Calculation Fix ===")
    
    env = create_test_environment()
    obs, _ = env.reset()
    
    # Open a long position
    action = [0.5, 0.0, 0.0, 0.0]  # 50% long
    obs, reward, done, truncated, info = env.step(action)
    trade_start_step = env.current_step
    print(f"Opened long position at step {trade_start_step}")
    
    # Wait several steps
    for i in range(5):
        action = [0.0, 0.0, 0.0, 0.0]  # Hold
        obs, reward, done, truncated, info = env.step(action)
    
    # Close the position
    action = [0.0, 1.0, 0.0, 0.0]  # Close long
    obs, reward, done, truncated, info = env.step(action)
    close_step = env.current_step
    
    print(f"Closed position at step {close_step}")
    print(f"Expected duration: {(close_step - trade_start_step) * 0.25} hours")
    
    # Check the logged trade
    if hasattr(env, 'logger') and env.logger.trades:
        last_trade = env.logger.trades[-1]
        print(f"Logged duration: {last_trade['trade_duration_hours']} hours")
        print(f"Duration calculation: {'✓ CORRECT' if last_trade['trade_duration_hours'] > 0 else '✗ FAILED'}")
    
    return env

def test_pnl_attribution():
    """Test that PnL only appears on final closure"""
    print("\n=== Testing PnL Attribution Fix ===")
    
    env = create_test_environment()
    obs, _ = env.reset()
    
    # Open a position
    action = [0.3, 0.0, 0.0, 0.0]  # 30% long
    obs, reward, done, truncated, info = env.step(action)
    print("Opened long position (30%)")
    
    # Check OPEN trade log
    if hasattr(env, 'logger') and env.logger.trades:
        open_trade = env.logger.trades[-1]
        print(f"OPEN trade PnL: {open_trade['net_pnl']} (should be 0.0)")
        print(f"OPEN PnL check: {'✓ CORRECT' if open_trade['net_pnl'] == 0.0 else '✗ FAILED'}")
    
    # Adjust position (increase to 50%)
    action = [0.5, 0.0, 0.0, 0.0]  # 50% long
    obs, reward, done, truncated, info = env.step(action)
    print("Adjusted position to 50%")
    
    # Check ADJUST trade log
    if hasattr(env, 'logger') and env.logger.trades:
        adjust_trade = env.logger.trades[-1]
        print(f"ADJUST trade PnL: {adjust_trade['net_pnl']} (should be 0.0)")
        print(f"ADJUST PnL check: {'✓ CORRECT' if adjust_trade['net_pnl'] == 0.0 else '✗ FAILED'}")
        print(f"ADJUST status: {adjust_trade['status']} (should be OPEN)")
        print(f"ADJUST status check: {'✓ CORRECT' if adjust_trade['status'] == 'OPEN' else '✗ FAILED'}")
    
    # Close position
    action = [0.0, 1.0, 0.0, 0.0]  # Close long
    obs, reward, done, truncated, info = env.step(action)
    print("Closed position")
    
    # Check CLOSE trade log
    if hasattr(env, 'logger') and env.logger.trades:
        close_trade = env.logger.trades[-1]
        print(f"CLOSE trade PnL: {close_trade['net_pnl']} (should be non-zero)")
        print(f"CLOSE PnL check: {'✓ CORRECT' if close_trade['net_pnl'] != 0.0 else '✗ FAILED'}")
        print(f"CLOSE status: {close_trade['status']} (should be CLOSED)")
        print(f"CLOSE status check: {'✓ CORRECT' if close_trade['status'] == 'CLOSED' else '✗ FAILED'}")
    
    return env

def test_close_reason_clarity():
    """Test that close_reason values are meaningful"""
    print("\n=== Testing Close Reason Clarity ===")
    
    env = create_test_environment()
    obs, _ = env.reset()
    
    # Open position
    action = [0.4, 0.0, 0.0, 0.0]  # 40% long
    obs, reward, done, truncated, info = env.step(action)
    
    # Check OPEN trade close_reason
    if hasattr(env, 'logger') and env.logger.trades:
        open_trade = env.logger.trades[-1]
        print(f"OPEN trade close_reason: '{open_trade['close_reason']}' (should be empty)")
        print(f"OPEN close_reason check: {'✓ CORRECT' if open_trade['close_reason'] == '' else '✗ FAILED'}")
    
    # Adjust position
    action = [0.6, 0.0, 0.0, 0.0]  # 60% long
    obs, reward, done, truncated, info = env.step(action)
    
    # Check ADJUST trade close_reason
    if hasattr(env, 'logger') and env.logger.trades:
        adjust_trade = env.logger.trades[-1]
        print(f"ADJUST trade close_reason: '{adjust_trade['close_reason']}' (should be empty)")
        print(f"ADJUST close_reason check: {'✓ CORRECT' if adjust_trade['close_reason'] == '' else '✗ FAILED'}")
    
    # Close position
    action = [0.0, 1.0, 0.0, 0.0]  # Close long
    obs, reward, done, truncated, info = env.step(action)
    
    # Check CLOSE trade close_reason
    if hasattr(env, 'logger') and env.logger.trades:
        close_trade = env.logger.trades[-1]
        print(f"CLOSE trade close_reason: '{close_trade['close_reason']}' (should be CLOSE_LONG)")
        print(f"CLOSE close_reason check: {'✓ CORRECT' if close_trade['close_reason'] == 'CLOSE_LONG' else '✗ FAILED'}")
    
    return env

def test_timestamp_handling():
    """Test that timestamps are handled correctly"""
    print("\n=== Testing Timestamp Handling ===")
    
    env = create_test_environment()
    obs, _ = env.reset()
    
    # Open position
    action = [0.3, 0.0, 0.0, 0.0]  # 30% long
    obs, reward, done, truncated, info = env.step(action)
    open_step = env.current_step
    
    # Get the entry datetime from the OPEN trade
    if hasattr(env, 'logger') and env.logger.trades:
        open_trade = env.logger.trades[-1]
        entry_datetime = open_trade['entry_datetime']
        print(f"OPEN trade entry_datetime: {entry_datetime}")
    
    # Wait a few steps then adjust
    for i in range(3):
        action = [0.0, 0.0, 0.0, 0.0]  # Hold
        obs, reward, done, truncated, info = env.step(action)
    
    # Adjust position
    action = [0.5, 0.0, 0.0, 0.0]  # 50% long
    obs, reward, done, truncated, info = env.step(action)
    adjust_step = env.current_step
    
    # Check ADJUST trade timestamps
    if hasattr(env, 'logger') and env.logger.trades:
        adjust_trade = env.logger.trades[-1]
        adjust_entry_datetime = adjust_trade['entry_datetime']
        print(f"ADJUST trade entry_datetime: {adjust_entry_datetime}")
        print(f"Timestamp preservation: {'✓ CORRECT' if adjust_entry_datetime == entry_datetime else '✗ FAILED'}")
        print(f"ADJUST close_datetime: '{adjust_trade['close_datetime']}' (should be empty)")
        print(f"ADJUST close_datetime check: {'✓ CORRECT' if adjust_trade['close_datetime'] == '' else '✗ FAILED'}")
    
    # Close position
    action = [0.0, 1.0, 0.0, 0.0]  # Close long
    obs, reward, done, truncated, info = env.step(action)
    close_step = env.current_step
    
    # Check CLOSE trade timestamps
    if hasattr(env, 'logger') and env.logger.trades:
        close_trade = env.logger.trades[-1]
        close_entry_datetime = close_trade['entry_datetime']
        close_datetime = close_trade['close_datetime']
        print(f"CLOSE trade entry_datetime: {close_entry_datetime}")
        print(f"CLOSE trade close_datetime: {close_datetime}")
        print(f"Entry timestamp preservation: {'✓ CORRECT' if close_entry_datetime == entry_datetime else '✗ FAILED'}")
        print(f"Different timestamps: {'✓ CORRECT' if close_datetime != close_entry_datetime else '✗ FAILED'}")
    
    return env

def test_flip_operation():
    """Test FLIP operation logging"""
    print("\n=== Testing FLIP Operation ===")
    
    env = create_test_environment()
    obs, _ = env.reset()
    
    # Open long position
    action = [0.4, 0.0, 0.0, 0.0]  # 40% long
    obs, reward, done, truncated, info = env.step(action)
    initial_trade_id = env.trade_id
    print(f"Opened long position, trade_id: {initial_trade_id}")
    
    # Wait a few steps
    for i in range(4):
        action = [0.0, 0.0, 0.0, 0.0]  # Hold
        obs, reward, done, truncated, info = env.step(action)
    
    # Flip to short position
    action = [0.0, 0.0, 0.4, 0.0]  # 40% short (flip)
    obs, reward, done, truncated, info = env.step(action)
    new_trade_id = env.trade_id
    
    print(f"Flipped to short position, new trade_id: {new_trade_id}")
    print(f"Trade ID increment: {'✓ CORRECT' if new_trade_id == initial_trade_id + 1 else '✗ FAILED'}")
    
    # Check the last two trades (should be CLOSE_LONG and OPEN_SHORT)
    if hasattr(env, 'logger') and len(env.logger.trades) >= 2:
        close_trade = env.logger.trades[-2]  # Second to last
        open_trade = env.logger.trades[-1]   # Last
        
        print(f"FLIP CLOSE: action={close_trade['entry_action']}, status={close_trade['status']}, duration={close_trade['trade_duration_hours']}")
        print(f"FLIP OPEN: action={open_trade['entry_action']}, status={open_trade['status']}, duration={open_trade['trade_duration_hours']}")
        
        # Verify CLOSE trade
        close_correct = (close_trade['entry_action'] == 'CLOSE_LONG' and 
                        close_trade['status'] == 'CLOSED' and
                        close_trade['trade_duration_hours'] > 0)
        print(f"FLIP CLOSE check: {'✓ CORRECT' if close_correct else '✗ FAILED'}")
        
        # Verify OPEN trade
        open_correct = (open_trade['entry_action'] == 'OPEN_SHORT' and
                       open_trade['status'] == 'OPEN' and
                       open_trade['trade_duration_hours'] == 0 and
                       open_trade['close_reason'] == '')
        print(f"FLIP OPEN check: {'✓ CORRECT' if open_correct else '✗ FAILED'}")
    
    return env

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("COMPREHENSIVE TRADE LOGGING VERIFICATION")
    print("=" * 60)
    
    try:
        # Run all individual tests
        test_duration_fix()
        test_pnl_attribution()
        test_close_reason_clarity()
        test_timestamp_handling()
        test_flip_operation()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("All tests completed. Check individual results above.")
        print("✓ = CORRECT, ✗ = FAILED")
        print("\nIf any tests show ✗ FAILED, review the specific issue.")
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up test file
    import os
    if os.path.exists('test_data_verification.csv'):
        os.remove('test_data_verification.csv')

if __name__ == "__main__":
    run_comprehensive_test()
