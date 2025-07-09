# Binance Trading Bot - Installation Guide

> **📋 Having installation issues?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for comprehensive solutions to common problems including certifi corruption, TensorTrade conflicts, and TA-Lib compilation issues.

## Environment Issues Fix

You're experiencing a conda environment corruption issue. Here are step-by-step solutions:

### Option 1: Clean Install (Recommended)

1. **Remove the corrupted environment:**
```bash
conda deactivate
conda env remove -n rl_trading_15m
```

2. **Create a fresh environment:**
```bash
conda create -n rl_trading_bot python=3.10
conda activate rl_trading_bot
```

3. **Install core dependencies via conda first:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy pandas matplotlib seaborn scikit-learn -c conda-forge
```

4. **Install remaining packages via pip:**
```bash
pip install -r requirements.txt
```

### Option 2: Fix Current Environment

1. **Try cleaning conda cache:**
```bash
conda clean --all
```

2. **Update conda:**
```bash
conda update conda
conda update --all
```

3. **Reinstall problematic packages:**
```bash
conda install --force-reinstall certifi
pip install --force-reinstall certifi
```

4. **Install requirements:**
```bash
pip install -r requirements.txt
```

### Option 3: Use pip-only approach

1. **Create Python virtual environment:**
```bash
python -m venv trading_bot_env
# On Windows:
trading_bot_env\Scripts\activate
# On Linux/Mac:
source trading_bot_env/bin/activate
```

2. **Upgrade pip:**
```bash
python -m pip install --upgrade pip
```

3. **Install PyTorch first:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Install other requirements:**
```bash
pip install -r requirements.txt
```

## If You Still Have Issues

### Manual Installation Order:
```bash
# Core packages first
pip install numpy pandas matplotlib
pip install torch torchvision torchaudio

# ML packages
pip install gymnasium stable-baselines3
pip install scikit-learn

# Trading packages
pip install python-binance ccxt pandas-ta

# UI packages
pip install rich tqdm plotly seaborn

# Optional packages
pip install jupyter ipykernel
```

### Alternative Requirements (Minimal):
If the full requirements still cause issues, try this minimal set:

```bash
pip install torch gymnasium stable-baselines3
pip install pandas numpy matplotlib
pip install python-binance pandas-ta
pip install rich tqdm
```

## Test Installation

After installation, test if everything works:

```bash
python test_system.py
```

## Common Issues and Solutions:

### 1. **certifi error**: 
```bash
pip uninstall certifi
pip install certifi
```

### 2. **torch installation issues**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU version
# OR for GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. **stable-baselines3 issues**:
```bash
pip install stable-baselines3 --no-deps
pip install gymnasium
```

### 4. **pandas-ta issues**:
```bash
pip install pandas-ta --no-deps
```

## Notes:

- **TensorTrade** is commented out as it can cause dependency conflicts
- **TA-Lib** requires C++ compilation, using pandas-ta instead
- **yfinance** is temporarily disabled due to installation issues
- The system will work perfectly without these optional packages

## Next Steps:

Once installation is complete:
1. Run `python test_system.py` to verify everything works
2. Run `python main.py` to start the trading bot
3. Begin with downloading data (option 5) or training a model (option 1)
