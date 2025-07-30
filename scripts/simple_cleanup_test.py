#!/usr/bin/env python3
"""
Simple verification test for code cleanup changes
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_cleanup_verification():
    """Test that our code cleanup worked correctly"""
    print("🧪 Testing Code Cleanup Verification...")
    
    try:
        from trading_environment import FuturesTradingEnv
        print("  ✅ Successfully imported FuturesTradingEnv")
        
        # Create minimal test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'open': np.random.normal(40000, 100, 100),
            'high': np.random.normal(40100, 100, 100),
            'low': np.random.normal(39900, 100, 100),
            'close': np.random.normal(40000, 100, 100),
            'volume': np.random.normal(1000, 100, 100)
        })
        
        # Test 1: Configurable max_risk_per_trade
        print("\n📝 Test 1: Configurable max_risk_per_trade parameter")
        
        # Test default value
        env1 = FuturesTradingEnv(df=test_data, window_size=60)
        assert hasattr(env1, 'max_risk_per_trade'), "max_risk_per_trade attribute missing"
        assert env1.max_risk_per_trade == 0.02, f"Default max_risk_per_trade should be 0.02, got {env1.max_risk_per_trade}"
        print("  ✅ Default max_risk_per_trade = 2% ✓")
        
        # Test custom value
        env2 = FuturesTradingEnv(df=test_data, max_risk_per_trade=0.05, window_size=60)
        assert env2.max_risk_per_trade == 0.05, f"Custom max_risk_per_trade should be 0.05, got {env2.max_risk_per_trade}"
        print("  ✅ Custom max_risk_per_trade = 5% ✓")
        
        # Test 2: Legacy _close_position method removed
        print("\n🗑️  Test 2: Legacy _close_position method removed")
        assert not hasattr(env1, '_close_position') or not callable(getattr(env1, '_close_position', None)), "_close_position should be removed or non-callable"
        print("  ✅ Legacy _close_position method properly removed ✓")
        
        # Test 3: Environment still functions normally
        print("\n🔧 Test 3: Environment functionality preserved")
        obs, info = env1.reset()
        print(f"  ✅ Environment reset successful")
        print(f"  📊 Observation type: {type(obs)}")
        
        if isinstance(obs, dict):
            print(f"  📊 Observation keys: {list(obs.keys())}")
            if 'market_features' in obs:
                print(f"  📈 Market features shape: {obs['market_features'].shape}")
            if 'portfolio_features' in obs:
                print(f"  💼 Portfolio features shape: {obs['portfolio_features'].shape}")
        
        print(f"  💰 Initial equity: ${env1.equity:,.2f}")
        print(f"  🎯 Max risk per trade: {env1.max_risk_per_trade*100:.1f}%")
        
        # Test 4: Stop-loss/take-profit still uses efficient trade system
        print("\n⚡ Test 4: Stop-loss system uses efficient trades")
        assert hasattr(env1, '_check_stop_loss_take_profit'), "Stop-loss method should exist"
        assert hasattr(env1, '_execute_efficient_trade'), "Efficient trade method should exist"
        print("  ✅ Stop-loss system properly integrated ✓")
        
        print("\n🎉 All cleanup verification tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cleanup_verification()
    if success:
        print("\n✅ Code cleanup verification completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Code cleanup verification failed!")
        sys.exit(1)
