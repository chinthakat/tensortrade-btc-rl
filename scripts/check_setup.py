#!/usr/bin/env python3
"""
Quick Environment Setup Script
Run this to activate the conda environment and verify setup
"""

import subprocess
import sys
import os

def check_conda_env():
    """Check if we're in the correct conda environment"""
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"Current conda environment: {env_name}")
    
    if env_name != 'rl_trading_15m':
        print("❌ Not in rl_trading_15m environment")
        print("💡 To activate, run:")
        print("   conda activate rl_trading_15m")
        print("   OR double-click: activate_env.bat")
        return False
    else:
        print("✅ Correct environment active!")
        return True

def check_python_packages():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'torch', 'stable_baselines3', 
        'gymnasium', 'matplotlib', 'scikit-learn'
    ]
    
    print("\n📦 Checking required packages:")
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("💡 To install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    else:
        print("\n🎉 All packages available!")
        return True

def main():
    print("🔧 ENVIRONMENT SETUP CHECK")
    print("=" * 40)
    
    env_ok = check_conda_env()
    packages_ok = check_python_packages()
    
    if env_ok and packages_ok:
        print("\n🚀 SETUP COMPLETE - Ready for training!")
        print("\n🎯 Quick start commands:")
        print("   python train_model.py              # Start training")
        print("   python -m tests.test_silent_penalties  # Test penalty system")
        print("   python backtest.py                 # Run backtesting")
    else:
        print("\n⚠️  Setup incomplete - please fix issues above")

if __name__ == "__main__":
    main()
