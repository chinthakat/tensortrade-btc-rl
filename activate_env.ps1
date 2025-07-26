# PowerShell script to auto-activate conda environment for this project
# This will automatically activate rl_trading_15m when opening terminal in this directory

# Check if we're in the TensorTradeModel directory
$currentPath = Get-Location
if ($currentPath.Path -like "*TensorTradeModel*") {
    # Check if conda environment is not already active
    if (-not $env:CONDA_DEFAULT_ENV -or $env:CONDA_DEFAULT_ENV -ne "rl_trading_15m") {
        Write-Host "🚀 Auto-activating conda environment: rl_trading_15m" -ForegroundColor Green
        conda activate rl_trading_15m
    }
}
