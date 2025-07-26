#!/usr/bin/env python3
"""
Step by step environment test
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_step_by_step():
    """Test environment creation step by step"""
    try:
        print("1. Testing imports...")
        from trading_environment import FuturesTradingEnv
        from improved_reward_configs import TREND_RIDER_CONFIG
        print("✅ Imports successful")
        
        print("2. Creating test data...")
        # Create simple test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'open': [50000.0] * 100,
            'high': [50100.0] * 100,
            'low': [49900.0] * 100,
            'close': [50000.0] * 100,
            'volume': [1000.0] * 100
        })
        print("✅ Test data created")
        
        print("3. Testing technical indicators manually...")
        import pandas_ta as ta
        
        # Test basic indicators
        sma_result = ta.sma(data['close'], length=10)
        print(f"✅ SMA calculated: {sma_result is not None}")
        
        ema_result = ta.ema(data['close'], length=10)
        print(f"✅ EMA calculated: {ema_result is not None}")
        
        rsi_result = ta.rsi(data['close'], length=14)
        print(f"✅ RSI calculated: {rsi_result is not None}")
        
        atr_result = ta.atr(data['high'], data['low'], data['close'], length=14)
        print(f"✅ ATR calculated: {atr_result is not None}")
        
        adx_result = ta.adx(data['high'], data['low'], data['close'], length=14)
        print(f"✅ ADX calculated: {adx_result is not None}")
        
        print("4. Creating environment...")
        env = FuturesTradingEnv(
            df=data,
            initial_equity=10000,
            max_leverage=10,
            reward_config=TREND_RIDER_CONFIG
        )
        print("✅ Environment created successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_step_by_step()
