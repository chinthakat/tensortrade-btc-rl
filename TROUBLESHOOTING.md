# Troubleshooting Guide

## Common Installation Issues and Solutions

### 1. Certifi Package Corruption (OSError)

If you encounter an error like:
```
OSError: Could not find a suitable TLS CA certificate bundle, invalid path: C:\Users\yourname\anaconda3\lib\site-packages\certifi\cacert.pem
```

**Solution Options:**

#### Option 1: Quick Fix - Reinstall Certifi
```powershell
conda uninstall certifi
conda install certifi
```

#### Option 2: Complete Environment Reset
```powershell
# Create a fresh conda environment
conda create -n trading_env python=3.10 -y
conda activate trading_env

# Install certifi first
conda install certifi -y

# Then install our requirements
pip install -r requirements.txt
```

#### Option 3: Use pip-only installation
```powershell
# Deactivate conda if active
conda deactivate

# Create a virtual environment with pip
python -m venv trading_venv
trading_venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. TensorTrade Installation Issues

TensorTrade is commented out in requirements.txt as it can cause dependency conflicts. The system includes fallback implementations for all TensorTrade features.

### 3. TA-Lib Installation Issues

TA-Lib requires C++ compilation. We use pandas-ta instead, which provides pure Python implementations of technical indicators. If you need TA-Lib specifically:

```powershell
# Download precompiled wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

# Install downloaded wheel
pip install TA_Lib-0.4.25-cp310-cp310-win_amd64.whl
```

### 4. PyTorch Installation Issues

For GPU support:
```powershell
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Pandas-TA Installation Issues

If pandas-ta fails to install:
```powershell
# Install dependencies first
pip install pandas numpy talib-binary

# Then install pandas-ta
pip install pandas-ta
```

The system includes fallback_ta.py with pure pandas/numpy implementations if pandas-ta is unavailable.

### 6. Testing Your Installation

After installation, run the test script:
```powershell
python test_system.py
```

This will verify all components are working correctly.

### 7. Alternative Installation Methods

#### Method 1: Conda-forge priority
```powershell
conda create -n trading_env python=3.10 -c conda-forge -y
conda activate trading_env
conda install -c conda-forge pandas numpy matplotlib seaborn plotly -y
pip install -r requirements.txt
```

#### Method 2: Mamba (faster conda alternative)
```powershell
# Install mamba
conda install mamba -n base -c conda-forge

# Use mamba instead of conda
mamba create -n trading_env python=3.10 -y
mamba activate trading_env
mamba install pandas numpy matplotlib seaborn plotly -y
pip install -r requirements.txt
```

### 8. Environment Variables

If you encounter SSL/TLS issues, you may need to set:
```powershell
# PowerShell
$env:REQUESTS_CA_BUNDLE = ""
$env:CURL_CA_BUNDLE = ""

# Or permanently in system environment variables
```

### 9. Network/Proxy Issues

If behind a corporate firewall:
```powershell
# Configure pip to use corporate certificates
pip config set global.trusted-host pypi.org
pip config set global.trusted-host pypi.python.org
pip config set global.trusted-host files.pythonhosted.org

# Install with --trusted-host flags
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### 10. Minimal Installation

For a minimal working setup, install only core components:
```powershell
pip install torch gymnasium stable-baselines3 pandas numpy scikit-learn matplotlib tqdm rich click
```

Then run `python test_system.py` to check what additional packages are needed.

## Contact and Support

If you continue to experience issues:
1. Check the specific error message
2. Verify Python version compatibility (3.8-3.11 recommended)
3. Try the minimal installation approach
4. Use virtual environments to avoid conflicts
5. Check for Windows-specific installation guides for problematic packages

## Quick Start After Installation

1. **Test the system**: `python test_system.py`
2. **Prepare your data**: Place CSV files in the `data/` directory
3. **Start training**: `python main.py` and select "Train Model"
4. **Run backtests**: `python main.py` and select "Backtest Strategy"
5. **Configure live trading**: Copy `config_template.json` to `config.json` and add your API keys
