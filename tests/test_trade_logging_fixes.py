"""
Quick test script to verify trade logging fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import TradingEnvironment
import pandas as pd
import numpy as np

def create_simple_test_data():
    """Create simple test data for quick verification"""
    # Create simple price data
    timestamps = pd.date_range('2024-01-01', periods=100, freq='15min')
    np.random.seed(42)  # For reproducible results
    
    # Simple price movement: starts at 40000, has some volatility
    base_price = 40000
    price_changes = np.random.normal(0, 50, 100)  # Small random changes
    prices = base_price + np.cumsum(price_changes)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps.astype(np.int64) // 10**9,  # Convert to Unix timestamp
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(100, 1000, 100)
    })
    
    return df

def test_trade_logging_fixes():
    """Test the key trade logging fixes"""
    print("🔧 Testing Trade Logging Fixes...")
    
    # Create test environment
    df = create_simple_test_data()
    env = TradingEnvironment(df)
    
    print(f"📊 Created test environment with {len(df)} data points")
    print(f"💰 Initial equity: ${env.equity:.2f}")
    
    # Test 1: Open a position
    print("\n🚀 Test 1: Opening LONG position...")
    initial_equity = env.equity
    action = [1, 0.1, 0.02]  # BUY action with 10% leverage, 2% risk
    obs, reward, done, info = env.step(action)
    
    print(f"   Position size: {env.position_size:.6f}")
    print(f"   Entry price: ${env.entry_price:.2f}")
    print(f"   Trade entry price: ${getattr(env, 'trade_entry_price', 'NOT SET'):.2f}")
    print(f"   Entry datetime: {getattr(env, 'trade_entry_datetime', 'NOT SET')}")
    print(f"   Entry equity: ${getattr(env, 'entry_equity', 'NOT SET'):.2f}")
    
    # Test 2: Wait a few steps then close
    print("\n⏳ Test 2: Waiting 5 steps then closing...")
    for i in range(5):
        action = [0, 0, 0]  # HOLD
        obs, reward, done, info = env.step(action)
    
    # Close position
    action = [2, 0, 0]  # SELL/CLOSE
    obs, reward, done, info = env.step(action)
    
    print(f"   Final equity: ${env.equity:.2f}")
    print(f"   Net change: ${env.equity - initial_equity:.2f}")
    
    # Check if trade was logged
    if hasattr(env, 'logger') and env.logger:
        trades = env.logger.trades
        print(f"\n📝 Trade Log Summary:")
        print(f"   Total trades logged: {len(trades)}")
        
        if trades:
            # Get the last few trades (should include open and close)
            for i, trade in enumerate(trades[-3:]):
                print(f"   Trade {i+1}: {trade.get('entry_action', 'N/A')} | "
                      f"Entry: ${trade.get('entry_price', 0):.2f} | "
                      f"Close: ${trade.get('close_price', 0):.2f} | "
                      f"PnL: ${trade.get('net_pnl', 0):.2f} | "
                      f"Entry DT: {trade.get('entry_datetime', 'N/A')} | "
                      f"Close DT: {trade.get('close_datetime', 'N/A')}")
    
    print("\n✅ Trade logging test completed!")

if __name__ == "__main__":
    test_trade_logging_fixes()
