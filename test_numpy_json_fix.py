"""
Test script to verify JSON serialization fix for NumPy types
"""
import json
import numpy as np
import pandas as pd
from multi_episode_training import NumpyEncoder

def test_numpy_json_serialization():
    """Test that NumPy types can be JSON serialized"""
    print("🧪 Testing NumPy JSON Serialization Fix")
    print("=" * 50)
    
    # Create test data with various NumPy types (similar to backtest results)
    test_data = {
        'total_return_pct': np.float32(15.75),
        'sharpe_ratio': np.float64(1.234567),
        'max_drawdown': np.float32(0.0876),
        'total_trades': np.int32(142),
        'win_rate': np.float32(0.5634),
        'equity_history': np.array([10000.0, 10150.0, 10075.0, 10200.0]),
        'timestamp': pd.Timestamp('2025-07-13 15:15:34'),
        'nan_value': np.nan,
        'regular_float': 123.456,
        'regular_int': 789
    }
    
    print("📊 Test data types:")
    for key, value in test_data.items():
        print(f"  - {key}: {type(value)} = {value}")
    
    try:
        # Test serialization with custom encoder
        json_string = json.dumps(test_data, indent=2, cls=NumpyEncoder)
        print("\n✅ JSON serialization successful!")
        print(f"📝 Serialized length: {len(json_string)} characters")
        
        # Test deserialization
        parsed_data = json.loads(json_string)
        print("✅ JSON deserialization successful!")
        
        print("\n📋 Parsed data:")
        for key, value in parsed_data.items():
            print(f"  - {key}: {type(value)} = {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_numpy_json_serialization()
    if success:
        print("\n🎉 NumPy JSON serialization fix verified!")
        print("Multi-episode training should no longer fail with JSON serialization errors.")
    else:
        print("\n💥 Fix needs more work")
