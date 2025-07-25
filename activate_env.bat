@echo off
echo 🚀 Activating rl_trading_15m conda environment...
call conda activate rl_trading_15m
echo ✅ Environment activated! Current environment: %CONDA_DEFAULT_ENV%
echo.
echo 🐍 Current Python: 
python --version
echo.
echo 📦 Key packages installed:
conda list | findstr "pandas\|numpy\|torch\|stable-baselines3\|gymnasium"
echo.
echo 🎯 Available commands:
echo   python train_model.py              - Start training
echo   python test_system.py              - Run system tests  
echo   python backtest.py                 - Run backtesting
echo   python test_silent_penalties.py    - Test penalty system
echo   python test_invalid_state_penalties.py - Test invalid state penalties
echo.
cmd /k
