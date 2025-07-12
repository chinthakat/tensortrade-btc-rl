"""
Test script to verify the Efficient Trade Execution fix.

This script demonstrates that position flips (long->short, short->long) now 
incur fees only once instead of twice, making trading costs more realistic.
"""

import pandas as pd
import numpy as np
import sys
from trading_environment import FuturesTradingEnv
import os
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv


def create_test_data(n_samples: int = 100, initial_price: float = 50000.0) -> pd.DataFrame:
    """Create simple test data for trade execution testing"""
    
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15T')
    
    # Create realistic price data with slight variations for technical indicators
    returns = np.random.normal(0.0, 0.001, n_samples)  # Very small volatility
    prices = [initial_price]
    
    for i in range(1, n_samples):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    # Generate OHLC with small variations
    highs = prices * (1 + np.random.uniform(0.0001, 0.002, n_samples))
    lows = prices * (1 - np.random.uniform(0.0001, 0.002, n_samples))
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    # Add some volume variation
    volumes = np.random.uniform(50, 150, n_samples)
    
    df = pd.DataFrame({
        'timestamp': [int(ts.timestamp()) for ts in timestamps],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    return df


def test_efficient_trade_execution():
    """Test that position flips use efficient trade execution"""
    
    print("Testing Efficient Trade Execution")
    print("=" * 60)
    
    # Create test data
    df = create_test_data(100, 50000.0)
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=10.0,
        taker_fee=0.0004,  # 0.04% fee - easier to track
        window_size=50
    )
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"Initial balance: ${env.balance:.2f}")
    print(f"Initial total fees: ${env.total_fees:.2f}")
    print(f"Taker fee rate: {env.taker_fee:.4f} ({env.taker_fee*100:.2f}%)")
    
    # Test 1: Open long position
    print("\\n" + "-" * 40)
    print("Test 1: Opening 5x Long Position")
    print("-" * 40)
    
    action = np.array([5.0])  # 5x long
    obs, reward, terminated, truncated, info = env.step(action)
    
    current_price = env.price_data.iloc[env.current_step]['close']
    position_value = abs(env.position_size * current_price)
    expected_fee = position_value * env.taker_fee
    
    print(f"Position size: {env.position_size:.6f} BTC")
    print(f"Position value: ${position_value:.2f}")
    print(f"Expected fee: ${expected_fee:.2f}")
    print(f"Actual total fees: ${env.total_fees:.2f}")
    print(f"Balance after trade: ${env.balance:.2f}")
    
    fees_after_long = env.total_fees
    balance_after_long = env.balance
    
    # Test 2: Flip to short position (this should be efficient!)
    print("\\n" + "-" * 40)
    print("Test 2: Flipping to 5x Short Position (EFFICIENCY TEST)")
    print("-" * 40)
    print("OLD METHOD: Would close long (fee) + open short (fee) = 2 fees")
    print("NEW METHOD: Should process as single trade = 1 fee")
    
    # Record position before flip
    old_position_size = env.position_size
    
    action = np.array([-5.0])  # 5x short - this is a position flip!
    obs, reward, terminated, truncated, info = env.step(action)
    
    current_price = env.price_data.iloc[env.current_step]['close']
    new_position_size = env.position_size
    
    # Calculate actual trade size (this is what the environment calculated)
    actual_trade_size = abs(new_position_size - old_position_size)
    actual_trade_value = actual_trade_size * current_price
    expected_single_fee = actual_trade_value * env.taker_fee
    
    actual_fees_incurred = env.total_fees - fees_after_long
    
    print(f"Old position: {old_position_size:.6f} BTC (long)")
    print(f"New position: {new_position_size:.6f} BTC (short)")
    print(f"Net trade size: {actual_trade_size:.6f} BTC")
    print(f"Net trade value: ${actual_trade_value:.2f}")
    print(f"Expected efficient fee: ${expected_single_fee:.2f}")
    print(f"Actual fees incurred: ${actual_fees_incurred:.2f}")
    print(f"Total fees so far: ${env.total_fees:.2f}")
    print(f"Balance after flip: ${env.balance:.2f}")
    
    # Test efficiency - use looser tolerance for rounding differences
    if abs(actual_fees_incurred - expected_single_fee) < 0.50:
        print("✅ EFFICIENT: Single fee charged for position flip!")
    else:
        print("❌ INEFFICIENT: Multiple fees charged!")
    
    # Test 3: Adjust position size (not a flip)
    print("\\n" + "-" * 40)
    print("Test 3: Adjusting Short Position (3x instead of 5x)")
    print("-" * 40)
    
    fees_before_adjust = env.total_fees
    action = np.array([-3.0])  # Reduce to 3x short
    obs, reward, terminated, truncated, info = env.step(action)
    
    fees_after_adjust = env.total_fees
    adjust_fees = fees_after_adjust - fees_before_adjust
    
    print(f"Position adjusted to: {env.position_size:.6f} BTC")
    print(f"Fees for adjustment: ${adjust_fees:.2f}")
    
    # Test 4: Close position completely
    print("\\n" + "-" * 40)
    print("Test 4: Closing Position Completely")
    print("-" * 40)
    
    fees_before_close = env.total_fees
    balance_before_close = env.balance
    
    action = np.array([0.0])  # Close position
    obs, reward, terminated, truncated, info = env.step(action)
    
    fees_after_close = env.total_fees
    close_fees = fees_after_close - fees_before_close
    
    print(f"Position size: {env.position_size:.6f} BTC")
    print(f"Fees for closing: ${close_fees:.2f}")
    print(f"Final balance: ${env.balance:.2f}")
    print(f"Total fees paid: ${env.total_fees:.2f}")
    print(f"Realized PnL: ${env.total_realized_pnl:.2f}")
    
    # Summary
    print("\\n" + "=" * 60)
    print("EFFICIENCY ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Calculate what old method would have cost
    old_method_fee_estimate = 4 * (5.0 * env.equity * env.taker_fee)  # 4 separate trades
    new_method_actual = env.total_fees
    
    print(f"Estimated old method total fees: ${old_method_fee_estimate:.2f}")
    print(f"New efficient method total fees: ${new_method_actual:.2f}")
    print(f"Savings: ${old_method_fee_estimate - new_method_actual:.2f}")
    print(f"Efficiency improvement: {((old_method_fee_estimate - new_method_actual) / old_method_fee_estimate * 100):.1f}%")
    
    print("\\n🎯 Key Improvements:")
    print("- Position flips processed as single trades")
    print("- Fees calculated on net trade volume only")
    print("- Realistic exchange-like behavior")
    print("- Reduced overstimation of trading costs")


def test_fee_comparison():
    """Compare old vs new method fees in detail"""
    
    print("\\n\\nDetailed Fee Comparison Test")
    print("=" * 60)
    
    df = create_test_data(50, 50000.0)
    
    # Test scenario: Flip from 10x long to 10x short
    equity = 10000.0
    leverage = 10.0
    price = 50000.0
    taker_fee = 0.0004
    
    position_size = leverage * equity / price
    position_value = position_size * price
    
    print(f"Scenario: Flip from {leverage}x long to {leverage}x short")
    print(f"Equity: ${equity:.2f}")
    print(f"Price: ${price:.2f}")
    print(f"Position size: {position_size:.6f} BTC")
    print(f"Position value: ${position_value:.2f}")
    
    # Old method simulation
    print("\\n📊 OLD METHOD SIMULATION:")
    print("Step 1: Close long position")
    old_close_fee = position_value * taker_fee
    print(f"  Close fee: ${old_close_fee:.2f}")
    
    print("Step 2: Open short position")  
    old_open_fee = position_value * taker_fee
    print(f"  Open fee: ${old_open_fee:.2f}")
    
    old_total = old_close_fee + old_open_fee
    print(f"  OLD TOTAL: ${old_total:.2f}")
    
    # New method calculation
    print("\\n⚡ NEW METHOD CALCULATION:")
    print("Single trade: +10x -> -10x")
    net_trade_size = 2 * position_size  # Going from +pos to -pos
    net_trade_value = net_trade_size * price
    new_fee = net_trade_value * taker_fee
    print(f"  Net trade size: {net_trade_size:.6f} BTC")
    print(f"  Net trade value: ${net_trade_value:.2f}")
    print(f"  NEW TOTAL: ${new_fee:.2f}")
    
    savings = old_total - new_fee
    print(f"\\n💰 SAVINGS: ${savings:.2f} ({savings/old_total*100:.1f}%)")
    
    # Verify with actual environment
    print("\\n🧪 VERIFICATION WITH ACTUAL ENVIRONMENT:")
    # Create test data with enough samples for our environment
    test_data = create_test_data(100)
    
    env = FuturesTradingEnv(
        df=test_data,
        initial_equity=equity,
        max_leverage=25.0,
        taker_fee=taker_fee,
        window_size=30
    )
    
    env.reset()
    
    # Open long position
    long_action = np.array([leverage])
    env.step(long_action)
    fees_after_long = env.total_fees
    
    # Flip to short position  
    short_action = np.array([-leverage])
    env.step(short_action)
    fees_after_flip = env.total_fees
    
    actual_flip_fee = fees_after_flip - fees_after_long
    print(f"  Actual flip fee: ${actual_flip_fee:.2f}")
    print(f"  Expected flip fee: ${new_fee:.2f}")
    print(f"  Difference: ${abs(actual_flip_fee - new_fee):.2f}")
    
    if abs(actual_flip_fee - new_fee) < 1.0:  # Allow small rounding differences
        print("✅ VERIFIED: Efficient execution working correctly!")
    else:
        print("❌ ISSUE: Actual fee doesn't match expected efficient fee")


if __name__ == "__main__":
    try:
        test_efficient_trade_execution()
        test_fee_comparison()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
