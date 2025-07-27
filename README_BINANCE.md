# Binance Live Trading Integration

This integration allows your trained model to trade live on Binance Testnet while continuously learning from market data.

## Features

1. **Live Trading with Paper Trading Option**
   - Real Binance Testnet integration
   - Paper trading mode for safe testing
   - Automatic order execution based on model signals

2. **Continuous Learning**
   - Model updates every hour with new market data
   - Preserves learned patterns while adapting to new conditions
   - Automatic model checkpointing

3. **Risk Management**
   - Position sizing based on risk percentage
   - Stop loss and take profit orders
   - Trailing stop loss for profit protection
   - Maximum daily loss limits
   - Maximum position limits

4. **Comprehensive Logging**
   - Market data logging (market_data_YYYYMMDD_HHMMSS.csv)
   - Trade logging with full details (trades_YYYYMMDD_HHMMSS.csv)
   - Action logging for model decisions (actions_YYYYMMDD_HHMMSS.csv)
   - Performance metrics logging (performance_YYYYMMDD_HHMMSS.csv)

## Setup

1. **Install Additional Requirements**
   ```bash
   pip install -r requirements_binance.txt
   ```

2. **Get Binance Testnet API Keys (Optional)**
   - Visit https://testnet.binancefuture.com
   - Create an account and generate API keys
   - Note: You can use paper trading mode without API keys

## Usage

### Quick Start (Paper Trading)
```bash
python launch_live_trading.py
```

### Manual Launch
```bash
# Paper trading mode (recommended for testing)
python binance_integration.py --model models/best_model.zip --paper

# With custom settings
python binance_integration.py --model models/best_model.zip --symbol BTCUSDT --timeframe 15m --balance 10000 --paper

# Live trading (requires API keys)
python binance_integration.py --model models/best_model.zip --api-key YOUR_KEY --api-secret YOUR_SECRET
```

## Data Format

The system expects market data in the following format:
```csv
,open,high,low,close,volume,timestamp
0,50000.0,50007.076,49999.848,50006.660,1803.186,1704027600
```

## Log Files

All logs are saved in the `binance_logs/` directory:

1. **market_data_*.csv**: Raw market data from Binance
2. **trades_*.csv**: Detailed trade execution logs
3. **actions_*.csv**: Model predictions and decisions
4. **performance_*.csv**: Account performance metrics

## Risk Management Settings

Default risk parameters (can be modified in code):
- Max position size: 10% of account balance
- Max open positions: 3
- Max daily loss: 5%
- Stop loss: 2%
- Take profit: 4%
- Trailing stop: 2%

## Continuous Training

The model automatically retrains every hour with new market data:
- Minimum 100 samples required for training
- Uses online learning (doesn't reset previous knowledge)
- Saves updated model with timestamp

## Safety Features

1. **Paper Trading Mode**: Test without real money
2. **Risk Limits**: Automatic trading halt on excessive losses
3. **Position Limits**: Prevents over-leveraging
4. **Error Handling**: Graceful handling of connection issues

## Monitoring

The system displays real-time performance metrics:
- Current balance and PnL
- Open positions
- Win rate and trade statistics
- Maximum drawdown

## Tips

1. Always test with paper trading first
2. Start with small position sizes
3. Monitor the first few trades closely
4. Check log files for detailed analysis
5. Use stop losses to protect capital

## Troubleshooting

1. **Connection Issues**: Check internet connection and API keys
2. **No Trades**: Verify model is generating signals (check actions log)
3. **Data Issues**: Ensure market data is being received (check market_data log)

## Example Output

```
🚀 Starting Live Trading System
Symbol: BTCUSDT
Timeframe: 15m
Initial Balance: $10,000.00

📊 Starting market data stream...
📥 Fetching historical data...
✅ Loaded 100 historical candles
💹 Starting trading loop...
📊 Starting performance monitor...
🧠 Starting continuous training loop...

✅ Opened BUY position: 0.002 @ $50,123.45
📈 Updated trailing stop for TRADE_1234: $49,621.01
💰 Closed BUY position: PnL $45.23 (STOP_LOSS)

Live Trading Performance
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric          ┃ Value       ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Balance         │ $10,045.23  │
│ Open PnL        │ $12.34      │
│ Realized PnL    │ $45.23      │
│ Total Trades    │ 5           │
│ Win Rate        │ 60.0%       │
│ Max Drawdown    │ 1.2%        │
│ Active Positions│ 1           │
└─────────────────┴─────────────┘
```