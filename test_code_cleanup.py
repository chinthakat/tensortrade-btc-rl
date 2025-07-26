#!/usr/bin/env python3
"""
Test script to verify code cleanup and improvements:
1. Configurable max_risk_per_trade parameter
2. Removed legacy _close_position method
3. Improved FLIP operation logging
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv

def test_configurable_risk_parameter():
    """Test that max_risk_per_trade is configurable"""
    print("🧪 Testing configurable max_risk_per_trade parameter...")
    
    # Create minimal test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
        'open': np.random.normal(40000, 100, 100),
        'high': np.random.normal(40100, 100, 100),
        'low': np.random.normal(39900, 100, 100),
        'close': np.random.normal(40000, 100, 100),
        'volume': np.random.normal(1000, 100, 100)
    })
    
    # Test different risk parameters
    risk_configs = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
    
    for max_risk in risk_configs:
        print(f"  📊 Testing max_risk_per_trade = {max_risk*100:.1f}%")
        
        env = FuturesTradingEnv(
            df=test_data,
            initial_equity=10000.0,
            max_risk_per_trade=max_risk,
            log_file=None  # No logging for test
        )
        
        # Verify the parameter is stored correctly
        assert env.max_risk_per_trade == max_risk, f"Expected {max_risk}, got {env.max_risk_per_trade}"
        print(f"    ✅ Parameter correctly stored: {env.max_risk_per_trade}")
    
    print("✅ All risk parameter tests passed!")

def test_legacy_method_removal():
    """Test that legacy _close_position method is removed"""
    print("\n🧪 Testing legacy method removal...")
    
    # Create minimal test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='15min'),
        'open': np.random.normal(40000, 100, 50),
        'high': np.random.normal(40100, 100, 50),
        'low': np.random.normal(39900, 100, 50),
        'close': np.random.normal(40000, 100, 50),
        'volume': np.random.normal(1000, 100, 50)
    })
    
    env = FuturesTradingEnv(
        df=test_data,
        initial_equity=10000.0,
        log_file=None
    )
    
    # Check that _close_position method is commented out/removed
    import inspect
    source_lines = inspect.getsource(FuturesTradingEnv)
    
    # Should not find the method definition
    assert "def _close_position(self" not in source_lines, "Legacy _close_position method still exists!"
    
    # But should find the comment indicating removal
    assert "REMOVED: _close_position method" in source_lines, "Removal comment not found"
    
    print("  ✅ _close_position method successfully removed")
    print("  ✅ Removal documentation comment found")
    print("✅ Legacy method removal test passed!")

def test_environment_creation():
    """Test that environment can be created with new parameters"""
    print("\n🧪 Testing environment creation with new parameters...")
    
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
        'open': np.random.normal(40000, 100, 100),
        'high': np.random.normal(40100, 100, 100),
        'low': np.random.normal(39900, 100, 100),
        'close': np.random.normal(40000, 100, 100),
        'volume': np.random.normal(1000, 100, 100)
    })
    
    # Test environment creation with custom risk parameter
    env = FuturesTradingEnv(
        df=test_data,
        initial_equity=10000.0,
        max_risk_per_trade=0.03,  # 3% custom risk
        window_size=60,
        use_advanced_action_space=True
    )
    
    # Verify basic functionality
    obs, info = env.reset()
    print(f"  ✅ Environment reset successful")
    
    # Check observation structure
    if isinstance(obs, dict):
        if 'market_features' in obs:
            print(f"  📊 Market features shape: {obs['market_features'].shape}")
        if 'portfolio_features' in obs:
            print(f"  💼 Portfolio features shape: {obs['portfolio_features'].shape}")
        # Legacy support for older observation format
        if 'prices' in obs:
            print(f"  📊 Observation shape: {obs['prices'].shape}")
    else:
        print(f"  📊 Observation shape: {obs.shape}")
    
    print(f"  💰 Initial equity: ${env.equity:,.2f}")
    print(f"  🎯 Max risk per trade: {env.max_risk_per_trade*100:.1f}%")
    
    # Test one step
    if isinstance(env.action_space, dict):
        action = {
            'action_type': np.array([1]),  # BUY
            'leverage': np.array([5.0]),
            'risk_percentage': np.array([0.5])  # 50% of max risk
        }
    else:
        action = np.array([5.0])  # Simple action space
    
    next_obs, reward, done, truncated, info = env.step(action)
    print(f"  ✅ Environment step successful")
    print(f"  🎁 Reward: {reward:.4f}")
    
    print("✅ Environment creation test passed!")

def run_all_tests():
    """Run all cleanup verification tests"""
    print("🚀 Running Code Cleanup Verification Tests")
    print("=" * 50)
    
    try:
        test_configurable_risk_parameter()
        test_legacy_method_removal()
        test_environment_creation()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Code cleanup successful!")
        print("✅ max_risk_per_trade is now configurable")
        print("✅ Legacy _close_position method removed")
        print("✅ Environment functionality preserved")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("Please check the implementation")
        raise

if __name__ == "__main__":
    run_all_tests()
