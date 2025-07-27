# BTCUSDT Perpetual Futures Trading System

A sophisticated live trading system specifically designed for BTCUSDT perpetual futures on Binance, featuring continuous model learning, secure configuration management, and comprehensive risk controls.

## 🚀 Features

- **BTCUSDT Focus**: Specifically optimized for Bitcoin perpetual futures trading
- **Secure Configuration**: API keys and settings managed through config.json (excluded from git)
- **Live Trading**: Real-time trading with Binance testnet and mainnet support
- **Paper Trading**: Safe simulation mode for testing strategies
- **Continuous Learning**: Model retrains with new market data every hour
- **Risk Management**: Comprehensive position sizing, stop losses, and drawdown protection
- **Funding Rate Monitoring**: Tracks BTCUSDT perpetual futures funding rates
- **Performance Monitoring**: Real-time P&L tracking and performance metrics
- **Data Logging**: Comprehensive CSV logging for trades, market data, and performance

## 📋 Prerequisites

1. **Python Dependencies**: Install required packages
   ```bash
   pip install -r requirements_binance.txt
   ```

2. **Trained Model**: Have a trained PPO model (`.zip` file) in the `models/` directory

3. **Configuration File**: Create `config.json` with your settings (see Configuration section)

## ⚙️ Configuration

### 1. Create config.json

The system uses `config.json` for secure configuration management. This file contains:

```json
{
    "binance": {
        "testnet": {
            "api_key": "your_testnet_api_key",
            "api_secret": "your_testnet_api_secret",
            "base_url": "https://testnet.binancefuture.com"
        },
        "mainnet": {
            "api_key": "your_mainnet_api_key", 
            "api_secret": "your_mainnet_api_secret",
            "base_url": "https://fapi.binance.com"
        }
    },
    "trading": {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "initial_balance": 10000.0,
        "use_testnet": true,
        "paper_trading": false
    },
    "risk_management": {
        "max_position_size_pct": 10.0,
        "max_open_positions": 3,
        "max_daily_loss_pct": 5.0,
        "stop_loss_pct": 2.0,
        "take_profit_pct": 4.0,
        "trailing_stop_pct": 2.0,
        "max_leverage": 10
    },
    "training": {
        "training_interval_hours": 1,
        "min_training_samples": 100,
        "model_save_interval_hours": 6,
        "continuous_learning": true
    }
}
```

### 2. API Key Setup

#### For Testnet (Recommended for testing):
1. Go to [Binance Testnet](https://testnet.binancefuture.com/)
2. Create account and generate API keys
3. Add keys to `config.json` under `binance.testnet`
4. Set `"use_testnet": true` in trading configuration

#### For Mainnet (Live trading):
1. Go to [Binance](https://www.binance.com/)
2. Create API keys with futures trading permissions
3. Add keys to `config.json` under `binance.mainnet`
4. Set `"use_testnet": false` in trading configuration

### 3. Security Notes

- `config.json` is automatically excluded from git tracking
- Never commit API keys to version control
- Use testnet for initial testing
- Start with small position sizes

## 🎮 Usage

### Quick Start

```bash
# Auto-detect latest model and use default config
python launch_btcusdt_trading.py

# Specify model and config
python launch_btcusdt_trading.py --model models/best_model.zip --config config.json

# Check system requirements
python launch_btcusdt_trading.py --check

# Show configuration information
python launch_btcusdt_trading.py --info
```

### Direct Usage

```bash
# Run with specific model
python binance_integration.py --model models/best_model.zip --config config.json
```

### Paper Trading Mode

If you want to test without real money:
1. Set `"paper_trading": true` in config.json, OR
2. Don't configure API keys (system will auto-enable paper trading)

## 📊 Risk Management

The system includes comprehensive risk management:

- **Position Sizing**: Maximum percentage of balance per trade
- **Stop Losses**: Automatic stop loss orders based on percentage
- **Take Profits**: Automatic take profit orders
- **Trailing Stops**: Dynamic stop loss adjustment in profit
- **Daily Loss Limits**: Maximum daily loss percentage
- **Leverage Limits**: Maximum allowed leverage
- **Position Limits**: Maximum number of open positions

## 🧠 Continuous Learning

The system continuously improves by:

1. **Data Collection**: Recording all market data and trading actions
2. **Hourly Training**: Retraining model with new data every hour
3. **Model Versioning**: Saving updated models with timestamps
4. **Performance Tracking**: Monitoring model performance over time

## 📈 Monitoring

### Real-time Display
- Current balance and P&L
- Open positions and their status
- Win rate and performance metrics
- Risk metrics and drawdown

### Data Logging
All data is logged to CSV files:
- `market_data_*.csv`: OHLCV data
- `trades_*.csv`: All executed trades
- `actions_*.csv`: Model decisions and actions
- `performance_*.csv`: Performance metrics over time

## 🔧 Troubleshooting

### Common Issues

1. **"No model found"**
   - Train a model first using `train_model.py`
   - Specify model path with `--model`

2. **"API connection failed"**
   - Check API keys in config.json
   - Verify testnet/mainnet settings
   - System will fall back to paper trading

3. **"Insufficient balance"**
   - Check account balance
   - Reduce position sizes in risk management settings

4. **"Import errors"**
   - Install requirements: `pip install -r requirements_binance.txt`
   - Ensure all required files are present

### Safe Testing

1. **Start with Paper Trading**: Set `"paper_trading": true`
2. **Use Testnet**: Set `"use_testnet": true`
3. **Small Positions**: Set low `max_position_size_pct`
4. **Monitor Performance**: Watch logs and performance metrics

## 📁 File Structure

```
TensorTradeModel/
├── binance_integration.py      # Main trading system
├── launch_btcusdt_trading.py   # Launch script
├── config.json                 # Configuration (create this)
├── trading_environment.py      # Trading environment
├── action_space_wrapper.py     # Action space wrapper
├── improved_reward_configs.py  # Reward configurations
├── models/                     # Trained models
├── binance_logs/              # Trading logs
└── data/                      # Market data
```

## 🛡️ Security Best Practices

1. **API Key Security**:
   - Use testnet for development
   - Restrict API key permissions
   - Regular key rotation

2. **Configuration Security**:
   - Never commit config.json
   - Use environment variables for production
   - Regular backups of configuration

3. **Trading Security**:
   - Start with small amounts
   - Monitor positions regularly
   - Set appropriate risk limits

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files in `binance_logs/`
3. Ensure all dependencies are installed
4. Verify configuration settings

## 🔄 Updates

The system automatically:
- Updates models with new market data
- Saves performance metrics
- Logs all trading activity
- Maintains position and risk management

## ⚠️ Disclaimer

This trading system is for educational and research purposes. Cryptocurrency trading involves significant financial risk. Always:
- Test thoroughly with paper trading
- Start with small amounts
- Monitor positions actively
- Use appropriate risk management
- Never trade more than you can afford to lose

The developers are not responsible for any financial losses incurred through the use of this system.
