@echo off
echo ====================================
echo  Binance Trading Bot Setup Script
echo ====================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8-3.11 first.
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv trading_venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call trading_venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Testing installation...
python test_system.py

echo.
echo ====================================
echo  Setup Complete!
echo ====================================
echo.
echo To activate the environment in the future, run:
echo   trading_venv\Scripts\activate.bat
echo.
echo To start the trading bot, run:
echo   python main.py
echo.
pause
