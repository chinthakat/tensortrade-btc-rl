# Logs Directory

This directory contains various log files generated during system operation.

## Directory Structure

```
logs/
├── README.md                    # This file
├── trading.log                  # General trading operations
├── training.log                 # Model training logs
├── backtest.log                # Backtesting results
├── live_trading.log            # Live trading operations
└── error.log                   # Error logs
```

## Log Types

- **Training Logs**: Model training progress and metrics
- **Trading Logs**: Buy/sell decisions and portfolio updates
- **Error Logs**: System errors and exceptions
- **Performance Logs**: System performance metrics

## Log Levels

- `DEBUG`: Detailed debugging information
- `INFO`: General information about system operation
- `WARNING`: Warning messages about potential issues
- `ERROR`: Error messages about problems
- `CRITICAL`: Critical errors that may stop execution

## Note

Log files are excluded from Git tracking to avoid repository bloat. Logs are generated automatically during system operation.
