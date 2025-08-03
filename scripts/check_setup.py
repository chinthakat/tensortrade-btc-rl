#!/usr/bin/env python3
"""
Quick Environment Setup Script
Run this to activate the conda environment and verify setup
"""

import subprocess
import sys
import os

def check_conda_env():
    """Report which conda environment is active.

    Informational only: a virtualenv works just as well, so this never
    decides whether the setup is usable.
    """
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    print(f"Current conda environment: {env_name}")
    
    if env_name != 'rl_trading_15m':
        print("ℹ  Not in the rl_trading_15m conda environment")
        print("💡 If you use conda, activate it with:")
        print("   conda activate rl_trading_15m")
        print("   OR double-click: activate_env.bat")
        print("   If you installed into a virtualenv instead, ignore this.")
        return False
    else:
        print("✅ Correct environment active!")
        return True

def check_python_packages():
    """Check if required packages are installed"""
    # Distribution name -> module name to import; they differ for scikit-learn.
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'torch',
        'stable-baselines3': 'stable_baselines3',
        'gymnasium': 'gymnasium',
        'matplotlib': 'matplotlib',
        'scikit-learn': 'sklearn',
    }
    
    print("\n📦 Checking required packages:")
    missing = []
    
    for package, module in required_packages.items():
        try:
            __import__(module)
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
    
    check_conda_env()
    packages_ok = check_python_packages()
    
    if packages_ok:
        print("\n🚀 SETUP COMPLETE - Ready for training!")
        print("\n🎯 Quick start commands:")
        print("   python train_model.py              # Start training")
        print("   python -m tests.test_silent_penalties  # Test penalty system")
        print("   python backtest.py                 # Run backtesting")
    else:
        print("\n⚠️  Setup incomplete - install the missing packages listed above")

if __name__ == "__main__":
    main()
