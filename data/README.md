# Data Directory

This directory contains historical market data for training and backtesting.

## Directory Structure

```
data/
├── README.md                    # This file
├── coinapi/                     # Data from CoinAPI
│   └── *.csv                   # OHLCV data files
└── *.csv                       # Direct data files
```

## Supported Data Formats

The trading bot expects CSV files with the following columns:
- `timestamp` (Unix timestamp)
- `open` (Opening price)
- `high` (Highest price)
- `low` (Lowest price) 
- `close` (Closing price)
- `volume` (Trading volume)

## Data Sources

- **Binance API**: Use the main.py interface to download data
- **CoinAPI**: Historical data from coinapi.io
- **Manual Upload**: Place CSV files here following the format above

## Getting Data

1. Run `python main.py`
2. Select option 5: "Download Data from Binance"
3. Follow the prompts to download historical data

## Note

Actual data files are excluded from Git tracking for performance and storage reasons. The `.gitignore` file ensures that `*.csv` files in this directory are not committed to the repository.
