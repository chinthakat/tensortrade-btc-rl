#!/usr/bin/env python3
"""
Quick test for the ADX fix
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from trading_environment import FuturesTradingEnv
from improved_reward_configs import TREND_RIDER_CONFIG

def test_adx_fix():
    """Test that the ADX issue is fixed"""
    print("🧪 Testing ADX fix...")
    
    # Create test data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='15min'),
        'open': np.random.randn(200).cumsum() + 50000,
        'high': np.random.randn(200).cumsum() + 50100,
        'low': np.random.randn(200).cumsum() + 49900,
        'close': np.random.randn(200).cumsum() + 50000,
        'volume': np.random.rand(200) * 1000
    })
    
    try:
        print("Creating environment...")
        env = FuturesTradingEnv(
            df=data,
            initial_equity=10000,
            max_leverage=10,
            reward_config=TREND_RIDER_CONFIG
        )
        print("✅ Environment created successfully!")
        
        # Test reset
        obs, info = env.reset()
        print("✅ Environment reset successful!")
        
        # Test step
        action = [0.5]  # Small action
        obs, reward, terminated, truncated, info = env.step(action)
        print("✅ Environment step successful!")
        
        print("🎯 All tests passed! ADX issue is fixed.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_adx_fix()
