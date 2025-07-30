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
echo   python -m tests.test_system        - Run system tests  
echo   python backtest.py                 - Run backtesting
echo   python -m tests.test_silent_penalties - Test penalty system
echo   python -m tests.test_invalid_state_penalties - Test invalid state penalties
echo.
cmd /k
