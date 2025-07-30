# Binance Trading Bot Setup Script for Windows PowerShell
Write-Host "====================================" -ForegroundColor Cyan
Write-Host " Binance Trading Bot Setup Script" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.8-3.11 first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Option to fix conda environment if exists
Write-Host ""
$condaChoice = Read-Host "Do you have conda installed and want to use it? (y/n)"
if ($condaChoice -eq "y" -or $condaChoice -eq "Y") {
    Write-Host "Setting up conda environment..." -ForegroundColor Yellow
    
    # Check if environment exists and remove if corrupted
    Write-Host "Checking for existing environment..." -ForegroundColor Yellow
    conda env list
    
    $envChoice = Read-Host "Do you want to create a fresh conda environment? (y/n)"
    if ($envChoice -eq "y" -or $envChoice -eq "Y") {
        Write-Host "Creating fresh conda environment..." -ForegroundColor Yellow
        conda create -n trading_bot python=3.10 -y
        conda activate trading_bot
        
        Write-Host "Installing core packages via conda..." -ForegroundColor Yellow
        conda install numpy pandas matplotlib seaborn scikit-learn -c conda-forge -y
        
        Write-Host "Installing requirements via pip..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
} else {
    # Virtual environment setup
    Write-Host ""
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv trading_venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host ""
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "trading_venv\Scripts\Activate.ps1"

    Write-Host ""
    Write-Host "Upgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip

    Write-Host ""
    Write-Host "Installing PyTorch (CPU version)..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    Write-Host ""
    Write-Host "Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "Testing installation..." -ForegroundColor Yellow
python -m tests.test_system

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

if ($condaChoice -ne "y" -and $condaChoice -ne "Y") {
    Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
    Write-Host "  trading_venv\Scripts\Activate.ps1" -ForegroundColor White
} else {
    Write-Host "To activate the conda environment in the future, run:" -ForegroundColor Cyan
    Write-Host "  conda activate trading_bot" -ForegroundColor White
}

Write-Host ""
Write-Host "To start the trading bot, run:" -ForegroundColor Cyan
Write-Host "  python main.py" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"
