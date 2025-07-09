# Binance Futures Trading Bot

A sophisticated cryptocurrency trading bot powered by deep reinforcement learning, designed specifically for Binance Futures trading. This project combines advanced machine learning techniques with real-world trading requirements, featuring TensorTrade-inspired architecture, comprehensive risk management, and multi-episode training capabilities.

## 🚀 Features

### Core Features
- **Deep Reinforcement Learning**: Advanced CNN-LSTM hybrid architectures for market analysis
- **Multiple Model Architectures**: CNN-LSTM, Attention CNN-LSTM, ResNet-LSTM
- **Risk Management**: Built-in stop-loss, take-profit, and position sizing
- **Live Trading**: Real-time trading on Binance Futures (testnet and live)
- **Multi-Episode Training**: Walk-forward validation with model persistence
- **Comprehensive Backtesting**: Detailed performance analysis with visualizations
- **Funding Fee Awareness**: Considers swap fees for short positions
- **Customizable Leverage**: Up to 25x leverage with proper risk management

### Technical Features
- **TensorTrade Integration**: Influenced by TensorTrade components and design patterns
- **Stable-Baselines3**: Industry-standard RL algorithms (PPO, A2C, SAC)
- **PyTorch Backend**: High-performance deep learning with GPU support
- **Rich CLI Interface**: Beautiful command-line interface with progress tracking
- **Comprehensive Logging**: Detailed trade logs in CSV format
- **Data Preprocessing**: Built-in technical indicators and feature engineering

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Binance account with Futures trading enabled
- Minimum 8GB RAM
- 5GB+ free disk space

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd GeminiModel
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up data directory:**
```bash
mkdir data
mkdir models
mkdir logs
```

4. **Download sample data** (or use the built-in downloader):
```bash
python main.py
# Select option 5 → Download data from Binance
```

## 🎯 Quick Start

### 1. Launch the Application
```bash
python main.py
```

### 2. First-Time Setup
1. **Download Data**: Use option 5 to download historical data from Binance
2. **Train Model**: Use option 1 to train your first model
3. **Backtest**: Use option 3 to evaluate model performance
4. **Live Trading**: Use option 4 (start with testnet!)

### 3. Training a Model
```bash
# From the main menu, select option 1
python main.py
# Follow the interactive prompts to:
# - Select data file
# - Choose model architecture
# - Configure training parameters
# - Start training process
```

## 📊 Data Format

Your CSV files should contain the following columns:
```csv
,open,high,low,close,volume,timestamp
0,42313.9,42535.0,42289.6,42532.5,3531.295,1704067200
1,42532.4,42603.2,42449.1,42458.5,2245.947,1704068100
```

- **timestamp**: Unix timestamp in seconds
- **open, high, low, close**: Price data in USDT
- **volume**: Trading volume

## 🧠 Model Architectures

### 1. CNN-LSTM (Default)
- 1D Convolutional layers for local pattern recognition
- LSTM layers for temporal sequence modeling
- Batch normalization and dropout for regularization

### 2. Attention CNN-LSTM
- Enhanced with multi-head attention mechanism
- Better long-range dependency modeling
- Improved performance on complex market patterns

### 3. ResNet-LSTM
- ResNet-style skip connections
- Deeper architecture for robust feature extraction
- Adaptive pooling for variable sequence lengths

## ⚡ RL Algorithms

### PPO (Proximal Policy Optimization) - Recommended
- Stable and sample-efficient
- Good balance between exploration and exploitation
- Robust to hyperparameter choices

### A2C (Advantage Actor-Critic)
- Faster training with parallel environments
- Lower sample efficiency than PPO
- Good for quick experiments

### SAC (Soft Actor-Critic)
- Maximum entropy framework
- Excellent for continuous action spaces
- Requires more computational resources

## 🎛️ Configuration Options

### Training Parameters
- **Initial Equity**: Starting capital (default: $10,000)
- **Max Leverage**: Maximum leverage (default: 25x)
- **Window Size**: Lookback period (default: 60)
- **Stop Loss**: Risk management (default: 2%)
- **Take Profit**: Profit target (default: 4%)
- **Training Steps**: Total timesteps (default: 1M)

### Risk Management
- **Position Sizing**: Based on Kelly Criterion
- **Daily Loss Limits**: Automatic trading halt
- **Liquidation Protection**: Conservative margin requirements
- **Funding Fee Consideration**: Especially for short positions

## 📈 Live Trading

### Safety First
1. **Always start with testnet**
2. **Test with small amounts**
3. **Monitor performance closely**
4. **Have exit strategies**

### API Setup
1. Create Binance API keys at: https://www.binance.com/en/my/settings/api-management
2. Enable Futures trading
3. Whitelist IP addresses (recommended)
4. Start with testnet: https://testnet.binancefuture.com/

### Risk Warning
⚠️ **IMPORTANT**: 
- Cryptocurrency trading involves significant financial risk
- You can lose all your invested capital
- Past performance does not guarantee future results
- This is experimental software
- Always test thoroughly before live trading

## 📊 Performance Analysis

### Backtest Metrics
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Trade Duration**: Time in market per trade

### Trade Logging
Every trade is logged with comprehensive details:
```csv
trade_id,training_step,training_iteration,entry_datetime,close_datetime,side,entry_action,entry_price,close_price,net_pnl,close_reward,entry_net_worth,close_net_worth,trade_duration_hours,status,win_loss,position_size,fees_paid,stop_loss_price,take_profit_price,close_reason
```

## 🔄 Multi-Episode Training

### Walk-Forward Validation
- Splits data into multiple training/validation periods
- Simulates real-world model deployment
- Prevents overfitting to historical data
- Tracks performance across different market conditions

### Model Persistence
- Automatically saves best-performing models
- Continues training from previous episodes
- Maintains comprehensive training history
- Enables model comparison and selection

## 🛡️ Risk Management Features

### Built-in Protections
- **Stop Loss Orders**: Automatic loss limiting
- **Take Profit Orders**: Profit securing
- **Position Size Limits**: Capital preservation
- **Daily Loss Limits**: Account protection
- **Liquidation Avoidance**: Conservative margin usage

### Funding Fee Optimization
- Considers 8-hourly funding payments
- Optimizes short position timing
- Accounts for funding rate trends
- Minimizes holding costs

## 📁 Project Structure

```
GeminiModel/
├── main.py                      # Main CLI interface
├── train_model.py              # Single model training
├── multi_episode_training.py   # Multi-episode training system
├── trading_environment.py      # Custom Gym environment
├── model_architectures.py      # Neural network architectures
├── backtest.py                 # Backtesting system
├── live_trading.py             # Live trading interface
├── requirements.txt            # Python dependencies
├── data/                       # Market data files
├── models/                     # Trained model files
├── logs/                       # Training and trade logs
├── configs/                    # Configuration files
├── episodes/                   # Multi-episode training data
└── README.md                   # This file
```

## 🔧 Advanced Usage

### Custom Environment Parameters
```python
env = FuturesTradingEnv(
    df=data,
    initial_equity=10000.0,
    max_leverage=25.0,
    maker_fee=0.0002,
    taker_fee=0.0004,
    funding_rate=0.0001,
    window_size=60,
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)
```

### Custom Model Training
```python
from model_architectures import CNNLSTMFeatureExtractor
from stable_baselines3 import PPO

policy_kwargs = {
    "features_extractor_class": CNNLSTMFeatureExtractor,
    "features_extractor_kwargs": {"features_dim": 256}
}

model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000000)
```

## 📚 Documentation

### Key Concepts
- **State Space**: Market features + Portfolio features
- **Action Space**: Continuous leverage from -25x to +25x
- **Reward Function**: Risk-adjusted profit with penalties
- **Feature Engineering**: 17+ technical indicators
- **Risk Management**: Multiple protection layers

### Technical Indicators Used
- Simple/Exponential Moving Averages
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- Volume-based indicators

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or n_envs
   - Use CPU training if necessary

2. **Data Loading Errors**
   - Check CSV format and column names
   - Ensure timestamp is Unix seconds

3. **API Connection Issues**
   - Verify API keys and permissions
   - Check internet connection
   - Try testnet first

4. **Training Convergence**
   - Adjust learning rate
   - Try different model architecture
   - Increase training timesteps

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚖️ Legal Disclaimer

This software is for educational and research purposes only. The authors and contributors are not responsible for any financial losses incurred through the use of this software. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorTrade**: For architectural inspiration and design patterns
- **Stable-Baselines3**: For robust RL algorithm implementations
- **Binance**: For comprehensive API access
- **PyTorch**: For deep learning framework
- **Rich**: For beautiful CLI interfaces

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review existing issues and discussions

---

**Happy Trading! 🚀💰**

*Remember: Always trade responsibly and never risk more than you can afford to lose.*
