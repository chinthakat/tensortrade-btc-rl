"""
Test Script for Binance Futures Trading Bot
Run basic tests to ensure everything is working correctly
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔧 Testing imports...")
    
    try:
        # Core libraries
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import gymnasium as gym
        print(f"✅ Gymnasium {gym.__version__}")
        
        from stable_baselines3 import PPO
        print("✅ Stable-Baselines3")
        
        # Data libraries
        import pandas as pd
        import numpy as np
        print("✅ Pandas & NumPy")
        
        # Visualization
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
        
        # CLI interface
        from rich.console import Console
        print("✅ Rich")
        
        # Technical analysis
        try:
            import pandas_ta as ta
            print("✅ Pandas-TA")
            ta_available = True
        except ImportError as e:
            print(f"⚠️  Pandas-TA import failed: {e}")
            print("   Using fallback technical analysis implementation")
            ta_available = False
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing data loading...")
    
    # Create sample data
    sample_data = {
        'open': [40000, 40100, 40050, 40200],
        'high': [40150, 40200, 40180, 40250],
        'low': [39950, 40000, 40000, 40150],
        'close': [40100, 40050, 40200, 40180],
        'volume': [1000, 1200, 800, 1500],
        'timestamp': [1704067200, 1704068100, 1704069000, 1704069900]
    }
    
    df = pd.DataFrame(sample_data)
    
    try:
        from data_utils import validate_data_format, clean_data
        
        # Test validation
        validation = validate_data_format(df)
        print(f"✅ Data validation: {all(validation.values())}")
        
        # Test cleaning
        cleaned_df = clean_data(df)
        print(f"✅ Data cleaning: {len(cleaned_df)} rows")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        return False

def test_environment():
    """Test trading environment"""
    print("\n🏪 Testing trading environment...")
    
    try:
        from trading_environment import FuturesTradingEnv
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        prices = 40000 + np.cumsum(np.random.randn(100) * 10)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.rand(100) * 50,
            'low': prices - np.random.rand(100) * 50,
            'close': prices + np.random.randn(100) * 20,
            'volume': np.random.rand(100) * 1000 + 500,
            'timestamp': [int(d.timestamp()) for d in dates]
        })
        
        # Create environment
        env = FuturesTradingEnv(df=df, initial_equity=10000)
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset - obs shape: {obs['market_features'].shape}")
        
        # Test step
        action = np.array([5.0])  # 5x long position
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step - reward: {reward:.4f}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"❌ Environment test failed: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return False

def test_model_architecture():
    """Test model architectures"""
    print("\n🧠 Testing model architectures...")
    
    try:
        from model_architectures import CNNLSTMFeatureExtractor
        import torch
        import gymnasium as gym
        
        # Create observation space
        obs_space = gym.spaces.Dict({
            'market_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(60, 17), dtype=np.float32),
            'portfolio_features': gym.spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        })
        
        # Create feature extractor
        extractor = CNNLSTMFeatureExtractor(obs_space, features_dim=256)
        
        # Test forward pass
        batch_size = 4
        sample_obs = {
            'market_features': torch.randn(batch_size, 60, 17),
            'portfolio_features': torch.randn(batch_size, 5)
        }
        
        output = extractor(sample_obs)
        print(f"✅ Model forward pass - output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {str(e)}")
        return False

def test_directories():
    """Test and create necessary directories"""
    print("\n📁 Testing directory structure...")
    
    directories = [
        'data', 'models', 'logs', 'configs', 
        'episodes', 'backtest_logs', 'live_trading_logs',
        'tensorboard_logs', 'backtest_plots', 'backtest_reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory: {directory}")
    
    return True

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Binance Futures Trading Bot Tests\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Environment Test", test_environment),
        ("Model Architecture Test", test_model_architecture),
        ("Directory Structure Test", test_directories)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("🎯 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your trading bot is ready to use.")
        print("\nNext steps:")
        print("1. Add data files to the 'data' directory")
        print("2. Run 'python main.py' to start the main interface")
        print("3. Begin with training a model or downloading data")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    run_all_tests()
