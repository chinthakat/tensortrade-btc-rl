#!/usr/bin/env python3
"""
Test script to verify ADJUST position logging fix
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

# Ensure we can import the trading environment
try:
    from trading_environment import FuturesTradingEnv
    print("✅ Successfully imported FuturesTradingEnv")
except ImportError as e:
    print(f"❌ Failed to import FuturesTradingEnv: {e}")
    sys.exit(1)

def test_adjust_position_logging():
    """Test the ADJUST position logging fix"""
    
    print("🧪 Testing ADJUST Position Logging Fix")
    print("=" * 50)
    
    # Create minimal test data
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic BTC price data
    base_price = 40000
    price_data = []
    current_price = base_price
    
    for i in range(n_samples):
        # Random walk with small steps
        change = np.random.normal(0, 0.002) * current_price  # 0.2% average change
        current_price += change
        current_price = max(35000, min(45000, current_price))  # Keep in reasonable range
        
        high = current_price * (1 + abs(np.random.normal(0, 0.001)))
        low = current_price * (1 - abs(np.random.normal(0, 0.001)))
        volume = np.random.uniform(100, 1000)
        
        price_data.append({
            'timestamp': i * 900,  # 15-minute intervals
            'open': current_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume
        })
    
    df = pd.DataFrame(price_data)
    
    # Add required technical indicators (simplified)
    df['returns'] = df['close'].pct_change().fillna(0)
    df['rsi'] = 50 + np.random.normal(0, 15, len(df))  # Random RSI around 50
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['ema_10'] - df['ema_20']
    df['adx'] = 20 + abs(np.random.normal(0, 10, len(df)))  # Random ADX
    df['atr'] = df['close'] * 0.01  # Simple ATR approximation
    df['volume_ratio'] = np.ones(len(df))
    
    print(f"✅ Created test data with {len(df)} samples")
    
    # Create environment (without logger to avoid file complications)
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        window_size=10
    )
    
    # Mock logger to capture trade data
    class MockLogger:
        def __init__(self):
            self.trades = []
        
        def log_trade(self, trade_data):
            self.trades.append(trade_data.copy())
            print(f"📝 Logged: {trade_data['entry_action']} - Position: {trade_data['position_size']:.6f}")
    
    # Attach mock logger
    env.logger = MockLogger()
    
    print("\n🔄 Testing Position Adjustments")
    print("-" * 30)
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial state: Step {env.current_step}, Equity ${env.equity:.2f}")
    
    # Test sequence: OPEN -> ADJUST -> ADJUST -> CLOSE
    test_actions = [
        ("OPEN", 0.01, "Open initial position"),
        ("ADJUST", 0.02, "Increase position size"),
        ("ADJUST", 0.015, "Decrease position size"),
        ("CLOSE", 0.0, "Close position")
    ]
    
    for action_name, target_size, description in test_actions:
        print(f"\n🎯 {action_name}: {description}")
        print(f"   Target position size: {target_size:.6f} BTC")
        
        # Get current price
        current_price = env._safe_get_price_data(env.current_step, 'close')
        print(f"   Current price: ${current_price:.2f}")
        
        # Execute trade
        env._execute_efficient_trade(target_size, current_price)
        
        print(f"   Actual position size after trade: {env.position_size:.6f} BTC")
        print(f"   Position side: {env.position_side}")
        
        # Check if a trade was logged
        if len(env.logger.trades) > 0:
            last_trade = env.logger.trades[-1]
            logged_position = last_trade['position_size']
            expected_position = target_size
            
            print(f"   Logged position size: {logged_position:.6f} BTC")
            
            # Check if the logged position matches expectation
            if abs(logged_position - expected_position) < 0.000001:
                print(f"   ✅ CORRECT: Logged position matches expected")
            else:
                print(f"   ❌ ERROR: Logged position {logged_position:.6f} != expected {expected_position:.6f}")
                print(f"   📊 Trade details: {last_trade}")
        else:
            print(f"   ⚠️  No trade logged (might be duplicate prevention)")
        
        # Move to next step
        if env.current_step < len(env.price_data) - 2:
            env.current_step += 1
    
    print(f"\n📊 Test Summary")
    print("=" * 30)
    print(f"Total trades logged: {len(env.logger.trades)}")
    
    # Analyze logged trades
    adjust_trades = [t for t in env.logger.trades if 'ADJUST' in t['entry_action']]
    print(f"ADJUST trades logged: {len(adjust_trades)}")
    
    for i, trade in enumerate(adjust_trades):
        action = trade['entry_action']
        position = trade['position_size']
        status = trade['status']
        
        print(f"  {i+1}. {action}: Position {position:.6f} BTC, Status: {status}")
        
        # Check for the specific issue: position_size = 0.0 on ADJUST with OPEN status
        if abs(position) < 0.000001 and status == 'OPEN':
            print(f"     ❌ BUG DETECTED: ADJUST action with zero position_size but OPEN status")
        elif status == 'OPEN' and abs(position) > 0.000001:
            print(f"     ✅ CORRECT: ADJUST action with valid position_size and OPEN status")
    
    if len(adjust_trades) == 0:
        print("  ⚠️  No ADJUST trades found - test might need refinement")
    
    print(f"\n🎯 Fix Status:")
    if any(abs(t['position_size']) < 0.000001 and t['status'] == 'OPEN' 
           for t in adjust_trades):
        print("❌ ISSUE STILL PRESENT: Found ADJUST trades with zero position_size")
    else:
        print("✅ FIX SUCCESSFUL: All ADJUST trades show correct position_size")

if __name__ == "__main__":
    # Suppress logging for cleaner output
    logging.getLogger().setLevel(logging.WARNING)
    
    test_adjust_position_logging()
