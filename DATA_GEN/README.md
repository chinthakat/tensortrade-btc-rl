# Bitcoin Synthetic Data Generator

Generates synthetic BTC OHLCV data in the same CSV shape the trading environment
expects, so the pipeline can be exercised without downloading real market data.

## Features

- **Market types**:
  - `UPTREND`: bullish, consistent upward movement
  - `DOWNTREND`: bearish, consistent downward movement, higher volatility
  - `SWING`: sideways, oscillating around a range
  - `MIXED`: alternating segments of the above
  - `CUSTOM_1`, `CUSTOM_UPTREND`: hand-tuned profiles. Their intended
    documentation files were committed empty and have been removed; read
    `MarketConfig` and the `CUSTOM_1` branches in `btc_data_generator.py` for
    what they actually do.
- **Timeframes**: 1m, 5m, 15m
- Open, High, Low, Close, Volume and Unix timestamp per candle
- Configurable starting price, volatility and trend strength

## Installation

Only pandas and numpy are needed:

```bash
pip install -r requirements.txt
```

## Usage

Run from the repository root:

```bash
python DATA_GEN/btc_data_generator.py \
    --start-date 2024-01-01 --end-date 2024-02-01 \
    --interval 15m --market-type UPTREND \
    --output data/uptrend_data.csv
```

### Parameters

- `--start-date`: start date, `YYYY-MM-DD` (required)
- `--end-date`: end date, `YYYY-MM-DD` (required)
- `--interval`: `1m`, `5m` or `15m` (required)
- `--market-type`: `UPTREND`, `DOWNTREND`, `SWING`, `MIXED`, `CUSTOM_1` or
  `CUSTOM_UPTREND` (required)
- `--initial-price`: starting BTC price (default `50000.0`)
- `--output`: output CSV path. If omitted, defaults to
  `data/BTC_SYNTHETIC_<market-type>_<interval>_<start>_to_<end>.csv`, relative
  to the current working directory.

The generator validates the output before it finishes: OHLC relationships, price
range and continuity between candles.

### Programmatic use

```python
from DATA_GEN.btc_data_generator import BTCDataGenerator

generator = BTCDataGenerator(initial_price=45000.0)

data = generator.generate_timeframe_data(
    start_date='2024-01-01',
    end_date='2024-01-31',
    interval='15m',
    market_type='UPTREND',
    output_path='data/my_data.csv'
)
```

## Output format

Matches the Binance-style CSV the environment reads, timestamp in Unix seconds:

```csv
,open,high,low,close,volume,timestamp
0,45000.0,45123.5,44987.2,45098.1,2150.45,1704067200
1,45098.1,45234.7,45067.8,45201.3,1876.23,1704068100
```

A 96-row sample of real generator output is committed as
[`test_small.csv`](test_small.csv). Full generated datasets are not committed —
regenerate them with the command above.

## Data quality

The generator produces:

- Valid OHLC relationships (High ≥ max(Open, Close), Low ≤ min(Open, Close))
- Volume correlated with price movement
- Continuity between candles, with configurable gaps
- Configurable noise and volatility

## Files

```
DATA_GEN/
├── btc_data_generator.py       Generator and CLI
├── test_small.csv              96-row output sample
├── requirements.txt            pandas, numpy
└── README.md                   This file
```
