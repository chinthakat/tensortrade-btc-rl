"""
Binance BTCUSDT Perpetual Futures Integration with Live Trading and Continuous Learning
Specifically designed for BTCUSDT perpetual futures trading with secure configuration management
"""

import os
import json
import time
import asyncio
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging
from pathlib import Path

# Binance imports (optional for testing)
try:
    from binance import AsyncClient, BinanceSocketManager
    from binance.client import Client
    from binance.enums import (
        ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL, 
        FUTURE_ORDER_TYPE_STOP_MARKET, FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET
    )
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    print("Warning: python-binance not available. Running in simulation mode.")
    # Mock classes for testing
    class AsyncClient:
        pass
    class BinanceSocketManager:
        pass
    class Client:
        pass
    class BinanceAPIException(Exception):
        pass
    ORDER_TYPE_MARKET = "MARKET"
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    FUTURE_ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
    FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    BINANCE_AVAILABLE = False

# Model imports
from stable_baselines3 import PPO
from trading_environment import FuturesTradingEnv
from dict_to_box_wrapper import DictToBoxObservationWrapper
from action_space_wrapper import wrap_environment_for_algorithm

# Rich console for better output
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigurationLoader:
    """Secure configuration loader for API keys and trading parameters"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            if not os.path.exists(self.config_path):
                console.print(f"[red]❌ Configuration file not found: {self.config_path}[/red]")
                console.print("[yellow]💡 Please create config.json with your API credentials[/yellow]")
                return self.get_default_config()
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            console.print(f"[green]✅ Configuration loaded from {self.config_path}[/green]")
            return config
            
        except Exception as e:
            console.print(f"[red]❌ Error loading configuration: {e}[/red]")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Return default configuration for paper trading"""
        return {
            "binance": {
                "testnet": {
                    "api_key": "",
                    "api_secret": "",
                    "base_url": "https://testnet.binancefuture.com"
                },
                "mainnet": {
                    "api_key": "",
                    "api_secret": "",
                    "base_url": "https://fapi.binance.com"
                }
            },
            "trading": {
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "initial_balance": 10000.0,
                "use_testnet": True,
                "paper_trading": True
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
                "continuous_learning": True
            }
        }
    
    def get_binance_config(self, use_testnet: bool = True) -> Dict:
        """Get Binance API configuration"""
        if use_testnet:
            return self.config["binance"]["testnet"]
        else:
            return self.config["binance"]["mainnet"]
    
    def get_trading_config(self) -> Dict:
        """Get trading configuration"""
        return self.config["trading"]
    
    def get_risk_config(self) -> Dict:
        """Get risk management configuration"""
        return self.config["risk_management"]
    
    def get_training_config(self) -> Dict:
        """Get training configuration"""
        return self.config["training"]

@dataclass
class FundingRateData:
    """Funding rate information for perpetual futures"""
    symbol: str
    funding_rate: float
    funding_time: datetime
    next_funding_time: datetime
    mark_price: float
    index_price: float

@dataclass
class TradeSignal:
    """Trade signal from the model"""
    timestamp: datetime
    action_type: int  # 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
    leverage: float
    risk_percentage: float
    confidence: float
    predicted_return: float

@dataclass
class ActiveTrade:
    """Active trade tracking"""
    trade_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    quantity: float
    leverage: float
    stop_loss: float
    take_profit: float
    status: str  # 'OPEN', 'CLOSED', 'LIQUIDATED'
    pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None

class BinanceBTCUSDTPerpConnector:
    """Binance BTCUSDT Perpetual Futures connector with secure configuration"""
    
    def __init__(self, config_loader: ConfigurationLoader):
        self.config_loader = config_loader
        self.trading_config = config_loader.get_trading_config()
        self.binance_config = config_loader.get_binance_config(self.trading_config.get("use_testnet", True))
        
        self.symbol = "BTCUSDT"  # Specifically for BTCUSDT perpetual futures
        self.paper_trading = self.trading_config.get("paper_trading", True)
        
        # Initialize client
        api_key = self.binance_config.get("api_key")
        api_secret = self.binance_config.get("api_secret")
        
        if not self.paper_trading and api_key and api_secret and BINANCE_AVAILABLE:
            try:
                is_testnet = self.trading_config.get("use_testnet", True)
                self.client = Client(api_key, api_secret, testnet=is_testnet)
                
                # Test the connection
                if is_testnet:
                    # Test testnet connection with futures account info
                    account_info = self.client.futures_account()
                    console.print("[green]✅ Connected to Binance TESTNET BTCUSDT Perpetual Futures[/green]")
                    console.print(f"[cyan]💰 Testnet Account Balance: {account_info.get('totalWalletBalance', 'N/A')} USDT[/cyan]")
                else:
                    # Test mainnet connection
                    account_info = self.client.futures_account()
                    console.print("[green]✅ Connected to Binance MAINNET BTCUSDT Perpetual Futures[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Failed to connect to Binance: {e}[/red]")
                console.print("[yellow]📝 Falling back to paper trading mode[/yellow]")
                self.client = None
                self.paper_trading = True
        else:
            self.client = None
            if not BINANCE_AVAILABLE:
                console.print("[yellow]📝 python-binance not available - Running in paper trading mode[/yellow]")
            else:
                console.print("[yellow]📝 Running in paper trading mode for BTCUSDT[/yellow]")
        
        # Paper trading state
        self.paper_balance = self.trading_config.get("initial_balance", 10000.0)
        self.paper_positions = {}
        self.paper_trades = []
        self.current_btc_price = 50000.0  # Default BTC price for simulation
    
    async def get_funding_rate(self) -> Optional[FundingRateData]:
        """Get current funding rate for BTCUSDT perpetual futures"""
        if self.paper_trading:
            # Simulate funding rate
            return FundingRateData(
                symbol=self.symbol,
                funding_rate=0.0001,  # 0.01% funding rate
                funding_time=datetime.now(),
                next_funding_time=datetime.now() + timedelta(hours=8),
                mark_price=self.current_btc_price,
                index_price=self.current_btc_price * 0.9999
            )
        
        try:
            # Get funding rate from Binance
            funding_rate = self.client.futures_funding_rate(symbol=self.symbol, limit=1)[0]
            mark_price = self.client.futures_mark_price(symbol=self.symbol)
            
            return FundingRateData(
                symbol=self.symbol,
                funding_rate=float(funding_rate['fundingRate']),
                funding_time=datetime.fromtimestamp(funding_rate['fundingTime'] / 1000),
                next_funding_time=datetime.fromtimestamp(funding_rate['fundingTime'] / 1000) + timedelta(hours=8),
                mark_price=float(mark_price['markPrice']),
                index_price=float(mark_price['indexPrice'])
            )
            
        except Exception as e:
            logger.error(f"Error getting funding rate: {e}")
            return None
    async def get_account_balance(self) -> Dict[str, float]:
        """Get USDT balance for BTCUSDT perpetual futures trading"""
        if self.paper_trading:
            return {
                'USDT': self.paper_balance,
                'available': self.paper_balance,
                'total': self.paper_balance
            }
        
        try:
            account = self.client.futures_account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    balance = float(asset['walletBalance'])
                    available = float(asset['availableBalance'])
                    
                    # For testnet, if balance is very low (< $100), use the configured initial balance
                    # This prevents emergency halts due to testnet having minimal funds
                    if balance < 100 and self.trading_config.get("use_testnet", True):
                        console.print(f"[yellow]⚠️ Testnet balance is low (${balance:.2f}), using configured initial balance[/yellow]")
                        return {
                            'USDT': self.trading_config.get("initial_balance", 10000.0),
                            'available': self.trading_config.get("initial_balance", 10000.0),
                            'total': self.trading_config.get("initial_balance", 10000.0)
                        }
                    
                    return {
                        'USDT': balance,
                        'available': available,
                        'total': balance
                    }
            return {'USDT': 0.0, 'available': 0.0, 'total': 0.0}
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    async def place_btcusdt_order(self, side: str, quantity: float, 
                                 leverage: int = 1, stop_loss: Optional[float] = None,
                                 take_profit: Optional[float] = None) -> Dict:
        """Place a BTCUSDT perpetual futures order"""
        if self.paper_trading:
            return await self._paper_place_order(side, quantity, leverage, stop_loss, take_profit)
        
        try:
            # Set leverage for BTCUSDT
            self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
            
            # Place market order
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            # Place stop loss if provided
            if stop_loss:
                self.client.futures_create_order(
                    symbol=self.symbol,
                    side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                    type=FUTURE_ORDER_TYPE_STOP_MARKET,
                    stopPrice=stop_loss,
                    quantity=quantity,
                    closePosition=True
                )
            
            # Place take profit if provided
            if take_profit:
                self.client.futures_create_order(
                    symbol=self.symbol,
                    side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                    type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                    stopPrice=take_profit,
                    quantity=quantity,
                    closePosition=True
                )
            
            return order
            
        except Exception as e:
            logger.error(f"Binance API error: {e}")
            return {'error': str(e)}
    
    async def _paper_place_order(self, side: str, quantity: float,
                                leverage: int, stop_loss: Optional[float],
                                take_profit: Optional[float]) -> Dict:
        """Simulate BTCUSDT order placement for paper trading"""
        # Get current BTC price (simulate or from data)
        current_price = await self.get_current_btc_price()
        
        order_id = f"PAPER_BTC_{int(time.time() * 1000)}"
        order_value = quantity * current_price
        
        # Check if we have enough balance
        required_margin = order_value / leverage
        if required_margin > self.paper_balance:
            return {'error': 'Insufficient USDT balance for BTCUSDT position'}
        
        # Create paper order
        order = {
            'orderId': order_id,
            'symbol': self.symbol,
            'side': side,
            'price': current_price,
            'origQty': quantity,
            'executedQty': quantity,
            'status': 'FILLED',
            'type': 'MARKET',
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        # Update paper balance
        self.paper_balance -= required_margin
        self.paper_positions[self.symbol] = order
        self.paper_trades.append(order)
        
        console.print(f"[green]📝 Paper trade: {side} {quantity} BTCUSDT @ ${current_price:,.2f}[/green]")
        
        return order
    
    async def get_current_btc_price(self) -> float:
        """Get current BTCUSDT market price"""
        if self.paper_trading:
            # Simulate price movement
            volatility = 0.001  # 0.1% volatility per update
            price_change = np.random.normal(0, volatility)
            self.current_btc_price *= (1 + price_change)
            return self.current_btc_price
        
        try:
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            price = float(ticker['price'])
            self.current_btc_price = price  # Update cached price
            return price
        except Exception as e:
            logger.error(f"Error getting BTCUSDT price: {e}")
            return self.current_btc_price
    
    async def close_btcusdt_position(self) -> Dict:
        """Close BTCUSDT perpetual futures position"""
        if self.paper_trading:
            if self.symbol in self.paper_positions:
                position = self.paper_positions[self.symbol]
                current_price = await self.get_current_btc_price()
                
                # Calculate PnL for BTCUSDT
                if position['side'] == 'BUY':
                    pnl = (current_price - position['price']) * position['executedQty']
                else:
                    pnl = (position['price'] - current_price) * position['executedQty']
                
                # Update balance
                self.paper_balance += (position['price'] * position['executedQty'] / position['leverage']) + pnl
                
                # Remove position
                del self.paper_positions[self.symbol]
                
                console.print(f"[green]📝 Paper close: BTCUSDT position closed, PnL: ${pnl:,.2f}[/green]")
                
                return {'status': 'CLOSED', 'pnl': pnl}
            return {'error': 'No BTCUSDT position found'}
        
        try:
            # Get current BTCUSDT position
            positions = self.client.futures_position_information(symbol=self.symbol)
            for position in positions:
                if float(position['positionAmt']) != 0:
                    # Close position
                    side = SIDE_SELL if float(position['positionAmt']) > 0 else SIDE_BUY
                    quantity = abs(float(position['positionAmt']))
                    
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=quantity,
                        reduceOnly=True
                    )
                    return order
            
            return {'error': 'No BTCUSDT position found'}
            
        except Exception as e:
            logger.error(f"Error closing BTCUSDT position: {e}")
            return {'error': str(e)}

class DataLogger:
    """Handles all data logging operations"""
    
    def __init__(self, base_path: str = "binance_logs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Setup log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.market_data_file = self.base_path / f"market_data_{timestamp}.csv"
        self.trades_file = self.base_path / f"trades_{timestamp}.csv"
        self.actions_file = self.base_path / f"actions_{timestamp}.csv"
        self.performance_file = self.base_path / f"performance_{timestamp}.csv"
        
        # Initialize CSV headers
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers"""
        # Market data header
        market_header = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        pd.DataFrame(columns=market_header).to_csv(self.market_data_file, index=False)
        
        # Trades header
        trades_header = ['trade_id', 'timestamp', 'symbol', 'side', 'entry_price', 
                        'quantity', 'leverage', 'stop_loss', 'take_profit', 'status',
                        'exit_price', 'exit_time', 'pnl', 'commission']
        pd.DataFrame(columns=trades_header).to_csv(self.trades_file, index=False)
        
        # Actions header
        actions_header = ['timestamp', 'action_type', 'leverage', 'risk_percentage',
                         'confidence', 'predicted_return', 'market_price', 'position_size',
                         'account_balance', 'open_positions']
        pd.DataFrame(columns=actions_header).to_csv(self.actions_file, index=False)
        
        # Performance header
        perf_header = ['timestamp', 'total_balance', 'available_balance', 'open_pnl',
                      'realized_pnl', 'daily_pnl', 'total_trades', 'winning_trades', 'losing_trades',
                      'win_rate', 'sharpe_ratio', 'max_drawdown', 'consecutive_losses', 'trading_enabled']
        pd.DataFrame(columns=perf_header).to_csv(self.performance_file, index=False)
    
    def log_market_data(self, data: Dict):
        """Log market data to CSV"""
        df = pd.DataFrame([data])
        df.to_csv(self.market_data_file, mode='a', header=False, index=False)
    
    def log_trade(self, trade: ActiveTrade):
        """Log trade details"""
        trade_data = {
            'trade_id': trade.trade_id,
            'timestamp': trade.entry_time,
            'symbol': trade.symbol,
            'side': trade.side,
            'entry_price': trade.entry_price,
            'quantity': trade.quantity,
            'leverage': trade.leverage,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'status': trade.status,
            'exit_price': trade.exit_price,
            'exit_time': trade.exit_time,
            'pnl': trade.pnl,
            'commission': 0.0  # Add commission calculation
        }
        df = pd.DataFrame([trade_data])
        df.to_csv(self.trades_file, mode='a', header=False, index=False)
    
    def log_action(self, signal: TradeSignal, market_price: float, 
                   position_size: float, balance: float, open_positions: int):
        """Log model actions"""
        action_data = {
            'timestamp': signal.timestamp,
            'action_type': signal.action_type,
            'leverage': signal.leverage,
            'risk_percentage': signal.risk_percentage,
            'confidence': signal.confidence,
            'predicted_return': signal.predicted_return,
            'market_price': market_price,
            'position_size': position_size,
            'account_balance': balance,
            'open_positions': open_positions
        }
        df = pd.DataFrame([action_data])
        df.to_csv(self.actions_file, mode='a', header=False, index=False)
    
    def log_performance(self, performance_data: Dict):
        """Log performance metrics"""
        df = pd.DataFrame([performance_data])
        df.to_csv(self.performance_file, mode='a', header=False, index=False)

class LiveBTCUSDTTradingSystem:
    """Live BTCUSDT Perpetual Futures trading system with continuous learning"""
    
    def __init__(self, model_path: str, config_path: str = "config.json", 
                 reward_config: Optional[Dict] = None):
        
        self.model_path = model_path
        
        # Set default reward config if none provided
        if reward_config is None:
            from improved_reward_configs import TREND_RIDER_CONFIG
            self.reward_config = TREND_RIDER_CONFIG
            console.print("[yellow]⚙️ Using default TREND_RIDER_CONFIG for reward configuration[/yellow]")
        else:
            self.reward_config = reward_config
        
        # Load configuration
        self.config_loader = ConfigurationLoader(config_path)
        self.trading_config = self.config_loader.get_trading_config()
        self.risk_config = self.config_loader.get_risk_config()
        self.training_config = self.config_loader.get_training_config()
        
        # Initialize components
        self.connector = BinanceBTCUSDTPerpConnector(self.config_loader)
        self.data_logger = DataLogger()
        
        # Load model
        self.model = PPO.load(model_path)
        console.print(f"[green]✅ Loaded model from {model_path}[/green]")
        
        # Trading state
        self.symbol = "BTCUSDT"
        self.timeframe = self.trading_config.get("timeframe", "15m")
        self.initial_balance = self.trading_config.get("initial_balance", 10000.0)
        
        self.market_data_buffer = deque(maxlen=100)  # Keep last 100 candles
        self.active_trades: Dict[str, ActiveTrade] = {}
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': self.initial_balance
        }
        
        # Risk management parameters from config
        self.max_position_size = self.risk_config.get("max_position_size_pct", 10.0) / 100.0
        self.max_open_positions = self.risk_config.get("max_open_positions", 3)
        self.max_daily_loss = self.risk_config.get("max_daily_loss_pct", 5.0) / 100.0
        self.trailing_stop_pct = self.risk_config.get("trailing_stop_pct", 2.0) / 100.0
        self.stop_loss_pct = self.risk_config.get("stop_loss_pct", 2.0) / 100.0
        self.take_profit_pct = self.risk_config.get("take_profit_pct", 4.0) / 100.0
        
        # Training state
        self.training_data_buffer = []
        self.min_training_samples = self.training_config.get("min_training_samples", 100)
        self.training_interval = self.training_config.get("training_interval_hours", 1) * 3600
        self.last_training_time = time.time()
        
        # Control flags
        self.running = False
        self.trading_enabled = True
        
        # Balance monitoring
        self.last_balance_update = 0
        self.balance_update_interval = 30  # Update balance every 30 seconds
        self.cached_balance = self.initial_balance
        self.balance_history = deque(maxlen=100)  # Keep balance history
        
        # Enhanced risk monitoring
        self.daily_start_balance = self.initial_balance
        self.daily_pnl = 0.0
        self.last_risk_check = 0
        self.risk_check_interval = 10  # Check risk every 10 seconds
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        
        # Net worth protection (50% loss = 24h halt)
        self.net_worth_protection_threshold = 0.5  # 50% loss
        self.emergency_halt_duration = 24 * 3600  # 24 hours in seconds
        self.emergency_halt_start_time = None
        self.is_emergency_halted = False
    
    async def start(self):
        """Start the BTCUSDT live trading system"""
        self.running = True
        
        console.print("[bold green]🚀 Starting BTCUSDT Perpetual Futures Trading System[/bold green]")
        console.print(f"Symbol: {self.symbol}")
        console.print(f"Timeframe: {self.timeframe}")
        console.print(f"Initial Balance: ${self.initial_balance:,.2f}")
        console.print(f"Paper Trading: {self.connector.paper_trading}")
        console.print(f"Risk Management: {self.risk_config}")
        
        # Initialize balance properly from actual account or config
        initial_balance_data = await self.connector.get_account_balance()
        if initial_balance_data and 'USDT' in initial_balance_data:
            actual_balance = initial_balance_data['USDT']
            self.cached_balance = actual_balance
            self.daily_start_balance = actual_balance
            console.print(f"[cyan]💰 Starting Balance: ${actual_balance:,.2f} USDT[/cyan]")
            
            # For testnet with low balance, adjust the initial balance reference
            if actual_balance < 100 and self.trading_config.get("use_testnet", True):
                console.print("[yellow]⚠️ Using testnet with configured initial balance for risk calculations[/yellow]")
                # Keep initial_balance as configured for risk calculations
            else:
                # Update initial balance to match actual balance for mainnet
                self.initial_balance = actual_balance
        
        # Get and display funding rate
        funding_rate = await self.connector.get_funding_rate()
        if funding_rate:
            console.print(f"[cyan]💰 Current Funding Rate: {funding_rate.funding_rate*100:.4f}%[/cyan]")
        
        # Start components
        await asyncio.gather(
            self.market_data_stream(),
            self.trading_loop(),
            self.performance_monitor(),
            self.balance_monitor(),  # New balance monitoring task
            self.risk_monitor(),     # New risk monitoring task
            self.continuous_training_loop() if self.training_config.get("continuous_learning", True) else self.dummy_task()
        )
    
    async def dummy_task(self):
        """Dummy task when continuous learning is disabled"""
        while self.running:
            await asyncio.sleep(3600)  # Sleep for 1 hour
    
    async def balance_monitor(self):
        """Continuously monitor account balance"""
        console.print("[cyan]💰 Starting balance monitor...[/cyan]")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update balance every 30 seconds
                if current_time - self.last_balance_update >= self.balance_update_interval:
                    await self.update_account_balance()
                    self.last_balance_update = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Balance monitor error: {e}")
                await asyncio.sleep(30)
    
    async def update_account_balance(self):
        """Update cached account balance and history"""
        try:
            balance_data = await self.connector.get_account_balance()
            
            if balance_data and 'USDT' in balance_data:
                new_balance = balance_data['USDT']['available'] if isinstance(balance_data['USDT'], dict) else balance_data['USDT']
                
                # Update cached balance
                old_balance = self.cached_balance
                self.cached_balance = new_balance
                
                # Add to balance history
                self.balance_history.append({
                    'timestamp': datetime.now(),
                    'balance': new_balance,
                    'change': new_balance - old_balance if old_balance else 0
                })
                
                # Calculate daily P&L
                if len(self.balance_history) > 0:
                    # Find balance from 24 hours ago
                    day_ago = datetime.now() - timedelta(hours=24)
                    day_start_balance = None
                    
                    for entry in self.balance_history:
                        if entry['timestamp'] >= day_ago:
                            day_start_balance = entry['balance']
                            break
                    
                    if day_start_balance:
                        self.daily_pnl = new_balance - day_start_balance
                    else:
                        self.daily_pnl = new_balance - self.daily_start_balance
                
                # Log significant balance changes
                balance_change = new_balance - old_balance
                if abs(balance_change) > 10:  # Log changes > $10
                    console.print(f"[cyan]💰 Balance update: ${new_balance:,.2f} (${balance_change:+,.2f})[/cyan]")
                
            else:
                logger.warning("Failed to get balance data from connector")
                
        except Exception as e:
            logger.error(f"Error updating account balance: {e}")
    
    async def risk_monitor(self):
        """Continuously monitor risk metrics and enforce limits"""
        console.print("[cyan]⚠️ Starting risk monitor...[/cyan]")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check risk every 10 seconds
                if current_time - self.last_risk_check >= self.risk_check_interval:
                    await self.comprehensive_risk_check()
                    self.last_risk_check = current_time
                
                await asyncio.sleep(2)  # Check every 2 seconds for responsiveness
                
            except Exception as e:
                logger.error(f"Risk monitor error: {e}")
                await asyncio.sleep(10)
    
    async def comprehensive_risk_check(self):
        """Comprehensive risk management checks"""
        try:
            current_balance = self.cached_balance
            
            # Check if we're in emergency halt period
            if self.is_emergency_halted:
                if self.emergency_halt_start_time and (time.time() - self.emergency_halt_start_time) < self.emergency_halt_duration:
                    # Still in halt period
                    remaining_hours = (self.emergency_halt_duration - (time.time() - self.emergency_halt_start_time)) / 3600
                    if int(time.time()) % 300 == 0:  # Log every 5 minutes
                        console.print(f"[red]🚨 EMERGENCY HALT ACTIVE: {remaining_hours:.1f} hours remaining[/red]")
                    return
                else:
                    # Halt period expired
                    console.print("[yellow]⏰ Emergency halt period expired. Evaluating conditions...[/yellow]")
                    await self.evaluate_emergency_halt_lift()
            
            # 0. NET WORTH PROTECTION CHECK (CRITICAL - 50% loss = 24h halt)
            net_worth_loss_pct = (self.initial_balance - current_balance) / self.initial_balance
            if net_worth_loss_pct >= self.net_worth_protection_threshold:
                if not self.is_emergency_halted:
                    console.print(f"[red]🚨🚨 CRITICAL: NET WORTH LOSS {net_worth_loss_pct*100:.1f}% - EMERGENCY 24H TRADING HALT 🚨🚨[/red]")
                    await self.trigger_emergency_halt("NET_WORTH_PROTECTION", net_worth_loss_pct)
                return
            
            # 1. Daily Loss Limit Check
            daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance if self.daily_start_balance > 0 else 0
            if self.daily_pnl < 0 and daily_loss_pct > self.max_daily_loss:
                if self.trading_enabled:
                    console.print(f"[red]🚨 DAILY LOSS LIMIT EXCEEDED: {daily_loss_pct*100:.2f}% (Max: {self.max_daily_loss*100:.1f}%)[/red]")
                    self.trading_enabled = False
                    await self.emergency_close_all_positions("DAILY_LOSS_LIMIT")
            
            # 2. Balance Threshold Check
            balance_loss_pct = (self.initial_balance - current_balance) / self.initial_balance
            if balance_loss_pct > self.max_daily_loss * 1.5:  # 1.5x daily loss limit as emergency stop
                if self.trading_enabled:
                    console.print(f"[red]🚨 EMERGENCY BALANCE PROTECTION: {balance_loss_pct*100:.2f}% loss[/red]")
                    self.trading_enabled = False
                    await self.emergency_close_all_positions("BALANCE_PROTECTION")
            
            # 3. Position Size Check
            total_position_value = 0
            for trade in self.active_trades.values():
                if trade.status == "OPEN":
                    current_price = self.market_data_buffer[-1]['close'] if self.market_data_buffer else trade.entry_price
                    position_value = trade.quantity * current_price / trade.leverage
                    total_position_value += position_value
            
            position_size_pct = total_position_value / current_balance if current_balance > 0 else 0
            max_total_position_pct = self.max_position_size * self.max_open_positions
            
            if position_size_pct > max_total_position_pct * 1.2:  # 20% over limit
                console.print(f"[yellow]⚠️ Total position size high: {position_size_pct*100:.1f}%[/yellow]")
            
            # 4. Consecutive Losses Check
            if self.consecutive_losses >= self.max_consecutive_losses:
                if self.trading_enabled:
                    console.print(f"[red]🚨 MAX CONSECUTIVE LOSSES REACHED: {self.consecutive_losses}[/red]")
                    self.trading_enabled = False
                    # Pause trading for 1 hour
                    asyncio.create_task(self.pause_trading_temporarily(3600))
            
            # 5. Balance Trend Analysis
            if len(self.balance_history) >= 10:
                recent_balances = [entry['balance'] for entry in list(self.balance_history)[-10:]]
                balance_trend = (recent_balances[-1] - recent_balances[0]) / recent_balances[0]
                
                if balance_trend < -0.02:  # 2% decline in recent history
                    console.print(f"[yellow]⚠️ Declining balance trend detected: {balance_trend*100:.2f}%[/yellow]")
            
            # Display current risk status
            self.display_risk_status()
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk check: {e}")
    
    def display_risk_status(self):
        """Display current risk status including emergency halt status"""
        current_balance = self.cached_balance
        daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance * 100 if self.daily_start_balance > 0 else 0
        net_worth_loss_pct = (self.initial_balance - current_balance) / self.initial_balance * 100
        
        # Calculate total exposure
        total_exposure = 0
        for trade in self.active_trades.values():
            if trade.status == "OPEN":
                current_price = self.market_data_buffer[-1]['close'] if self.market_data_buffer else trade.entry_price
                exposure = trade.quantity * current_price
                total_exposure += exposure
        
        exposure_pct = total_exposure / current_balance * 100 if current_balance > 0 else 0
        
        risk_table = Table(title="Risk Management Status", title_style="bold red")
        risk_table.add_column("Risk Metric", style="cyan")
        risk_table.add_column("Current", style="white")
        risk_table.add_column("Limit", style="yellow")
        risk_table.add_column("Status", style="bold")
        
        # Emergency Halt Status (CRITICAL)
        if self.is_emergency_halted:
            remaining_hours = (self.emergency_halt_duration - (time.time() - self.emergency_halt_start_time)) / 3600 if self.emergency_halt_start_time else 0
            risk_table.add_row(
                "🚨 EMERGENCY HALT",
                f"{remaining_hours:.1f}h left",
                "24h halt",
                "🔴 ACTIVE"
            )
        
        # Net Worth Protection (CRITICAL)
        net_worth_status = "🟢 SAFE" if net_worth_loss_pct < 30 else "🟡 WARNING" if net_worth_loss_pct < 45 else "🔴 CRITICAL"
        risk_table.add_row(
            "Net Worth Loss",
            f"{net_worth_loss_pct:.1f}%",
            f"{self.net_worth_protection_threshold*100:.0f}%",
            net_worth_status
        )
        
        # Daily Loss
        daily_status = "🟢 OK" if daily_loss_pct < self.max_daily_loss * 50 else "🟡 WARNING" if daily_loss_pct < self.max_daily_loss * 80 else "🔴 CRITICAL"
        risk_table.add_row(
            "Daily Loss", 
            f"{daily_loss_pct:.2f}%",
            f"{self.max_daily_loss*100:.1f}%",
            daily_status
        )
        
        # Position Count
        pos_status = "🟢 OK" if len(self.active_trades) < self.max_open_positions else "🔴 MAX"
        risk_table.add_row(
            "Open Positions",
            str(len(self.active_trades)),
            str(self.max_open_positions),
            pos_status
        )
        
        # Total Exposure
        exp_status = "🟢 OK" if exposure_pct < 50 else "🟡 MODERATE" if exposure_pct < 100 else "🔴 HIGH"
        risk_table.add_row(
            "Total Exposure",
            f"{exposure_pct:.1f}%",
            "100%",
            exp_status
        )
        
        # Consecutive Losses
        loss_status = "🟢 OK" if self.consecutive_losses < 3 else "🟡 WARNING" if self.consecutive_losses < 5 else "🔴 CRITICAL"
        risk_table.add_row(
            "Consecutive Losses",
            str(self.consecutive_losses),
            str(self.max_consecutive_losses),
            loss_status
        )
        
        # Trading Status
        if self.is_emergency_halted:
            trading_status = "� EMERGENCY HALT"
        elif self.trading_enabled:
            trading_status = "🟢 ENABLED"
        else:
            trading_status = "� PAUSED"
        
        risk_table.add_row(
            "Trading Status",
            trading_status,
            "ENABLED",
            trading_status
        )
        
        console.print(risk_table)
        
        # Show emergency halt warning if active
        if self.is_emergency_halted:
            remaining_time = self.emergency_halt_duration - (time.time() - self.emergency_halt_start_time) if self.emergency_halt_start_time else 0
            console.print(f"[red bold]🚨 EMERGENCY HALT ACTIVE: {remaining_time/3600:.1f} hours remaining 🚨[/red bold]")
    
    async def emergency_close_all_positions(self, reason: str):
        """Emergency close all open positions"""
        console.print(f"[red]🚨 EMERGENCY: Closing all positions - {reason}[/red]")
        
        for trade_id in list(self.active_trades.keys()):
            try:
                await self.close_trade(trade_id, reason=f"EMERGENCY_{reason}")
                await asyncio.sleep(1)  # Small delay between closes
            except Exception as e:
                logger.error(f"Error closing position {trade_id}: {e}")
    
    async def pause_trading_temporarily(self, duration_seconds: int):
        """Temporarily pause trading for a specified duration"""
        console.print(f"[yellow]⏸️ Trading paused for {duration_seconds//60} minutes[/yellow]")
        await asyncio.sleep(duration_seconds)
        
        # Re-enable trading if risk conditions are acceptable
        if await self.check_risk_limits():
            self.trading_enabled = True
            self.consecutive_losses = 0  # Reset consecutive losses
            console.print("[green]▶️ Trading resumed after pause[/green]")
    
    async def trigger_emergency_halt(self, reason: str, loss_percentage: float):
        """Trigger 24-hour emergency trading halt due to severe losses"""
        console.print(f"[red]🚨🚨 EMERGENCY HALT TRIGGERED: {reason} - {loss_percentage*100:.1f}% LOSS 🚨🚨[/red]")
        
        # Close all positions immediately
        await self.emergency_close_all_positions(f"EMERGENCY_HALT_{reason}")
        
        # Set emergency halt flags
        self.is_emergency_halted = True
        self.emergency_halt_start_time = time.time()
        self.trading_enabled = False
        
        # Log the emergency halt
        emergency_data = {
            'timestamp': datetime.now(),
            'reason': reason,
            'loss_percentage': loss_percentage,
            'balance_at_halt': self.cached_balance,
            'initial_balance': self.initial_balance,
            'halt_duration_hours': self.emergency_halt_duration / 3600
        }
        
        # Save emergency halt log
        emergency_log_file = Path("binance_logs") / f"emergency_halt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(emergency_log_file, 'w') as f:
            json.dump(emergency_data, f, indent=2, default=str)
        
        console.print(f"[red]📄 Emergency halt logged to: {emergency_log_file}[/red]")
        
        # Send alert notifications (if configured)
        await self.send_emergency_alert(reason, loss_percentage)
        
        console.print("[red]🛑 TRADING HALTED FOR 24 HOURS - SYSTEM WILL AUTO-EVALUATE AFTER HALT PERIOD[/red]")
    
    async def evaluate_emergency_halt_lift(self):
        """Evaluate whether to lift the emergency halt after 24 hours"""
        console.print("[yellow]🔍 Evaluating emergency halt lift conditions...[/yellow]")
        
        current_balance = self.cached_balance
        current_loss_pct = (self.initial_balance - current_balance) / self.initial_balance
        
        # Conditions for lifting the halt
        conditions_met = []
        
        # 1. Balance must have recovered somewhat (< 45% loss)
        if current_loss_pct < 0.45:  # Less than 45% loss
            conditions_met.append("✅ Balance recovery detected")
        else:
            conditions_met.append("❌ Balance still critical")
        
        # 2. No active positions (all should be closed)
        if len(self.active_trades) == 0:
            conditions_met.append("✅ No active positions")
        else:
            conditions_met.append("❌ Active positions still exist")
        
        # 3. Market conditions stable (check balance history)
        if len(self.balance_history) >= 5:
            recent_balances = [entry['balance'] for entry in list(self.balance_history)[-5:]]
            balance_stability = max(recent_balances) - min(recent_balances)
            if balance_stability < current_balance * 0.05:  # Less than 5% volatility
                conditions_met.append("✅ Balance stability confirmed")
            else:
                conditions_met.append("❌ Balance still volatile")
        else:
            conditions_met.append("⚠️ Insufficient balance history")
        
        console.print("[cyan]Emergency Halt Lift Evaluation:[/cyan]")
        for condition in conditions_met:
            console.print(f"  {condition}")
        
        # Decision
        if all("✅" in condition for condition in conditions_met):
            # Lift the halt with reduced trading parameters
            console.print("[green]✅ Emergency halt lifted - resuming with REDUCED risk parameters[/green]")
            await self.lift_emergency_halt()
        else:
            # Extend the halt for another 12 hours
            console.print("[red]❌ Conditions not met - extending halt for 12 more hours[/red]")
            self.emergency_halt_start_time = time.time()
            self.emergency_halt_duration = 12 * 3600  # 12 more hours
    
    async def lift_emergency_halt(self):
        """Lift emergency halt and resume trading with reduced risk parameters"""
        # Reset halt flags
        self.is_emergency_halted = False
        self.emergency_halt_start_time = None
        
        # Reduce risk parameters temporarily
        original_max_position_size = self.max_position_size
        original_max_open_positions = self.max_open_positions
        
        # Reduce to 50% of original risk parameters
        self.max_position_size = original_max_position_size * 0.5
        self.max_open_positions = max(1, original_max_open_positions // 2)
        
        console.print(f"[yellow]⚠️ Reduced risk parameters:[/yellow]")
        console.print(f"  Max position size: {self.max_position_size*100:.1f}% (was {original_max_position_size*100:.1f}%)")
        console.print(f"  Max open positions: {self.max_open_positions} (was {original_max_open_positions})")
        
        # Re-enable trading with caution
        self.trading_enabled = True
        self.consecutive_losses = 0
        
        # Schedule risk parameter restoration in 24 hours
        asyncio.create_task(self.restore_risk_parameters_after_delay(24 * 3600, original_max_position_size, original_max_open_positions))
        
        console.print("[green]▶️ Trading resumed with REDUCED risk parameters[/green]")
    
    async def restore_risk_parameters_after_delay(self, delay_seconds: int, original_max_position_size: float, original_max_open_positions: int):
        """Restore original risk parameters after a delay"""
        await asyncio.sleep(delay_seconds)
        
        # Check if conditions are still stable
        current_balance = self.cached_balance
        current_loss_pct = (self.initial_balance - current_balance) / self.initial_balance
        
        if current_loss_pct < 0.3 and self.consecutive_losses < 3:  # Less than 30% loss and no recent consecutive losses
            self.max_position_size = original_max_position_size
            self.max_open_positions = original_max_open_positions
            console.print("[green]✅ Risk parameters restored to original values[/green]")
        else:
            console.print("[yellow]⚠️ Risk parameters remain reduced due to ongoing concerns[/yellow]")
    
    async def send_emergency_alert(self, reason: str, loss_percentage: float):
        """Send emergency alerts (placeholder for future implementation)"""
        try:
            # Log the alert
            alert_message = f"EMERGENCY HALT: {reason} - {loss_percentage*100:.1f}% loss at {datetime.now()}"
            logger.critical(alert_message)
            
            # Future: Send email, SMS, Discord/Slack notifications
            # This is a placeholder for future notification implementations
            
        except Exception as e:
            logger.error(f"Error sending emergency alert: {e}")
    
    async def market_data_stream(self):
        """Stream market data from Binance"""
        console.print("[cyan]📊 Starting market data stream...[/cyan]")
        
        while self.running:
            try:
                # Get historical data first
                if len(self.market_data_buffer) == 0:
                    await self.fetch_historical_data()
                
                # Then stream real-time data
                if self.connector.paper_trading:
                    # Simulate real-time data for paper trading
                    await self.simulate_market_data()
                else:
                    # Connect to Binance websocket
                    await self.stream_binance_data()
                
            except Exception as e:
                logger.error(f"Market data error: {e}")
                await asyncio.sleep(5)
    
    async def fetch_historical_data(self):
        """Fetch historical BTCUSDT klines data"""
        console.print("[cyan]📥 Fetching historical BTCUSDT data...[/cyan]")
        
        if self.connector.paper_trading:
            # Load from CSV file for paper trading
            try:
                df = pd.read_csv("data/btc_usdt_15m.csv")
                df = df.tail(100)  # Get last 100 candles
                
                for _, row in df.iterrows():
                    candle = {
                        'timestamp': int(row['timestamp']),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume'])
                    }
                    self.market_data_buffer.append(candle)
                    self.data_logger.log_market_data(candle)
                
                # Update current BTC price
                if len(self.market_data_buffer) > 0:
                    self.connector.current_btc_price = self.market_data_buffer[-1]['close']
                
                console.print(f"[green]✅ Loaded {len(self.market_data_buffer)} BTCUSDT historical candles[/green]")
                
            except Exception as e:
                logger.error(f"Error loading historical BTCUSDT data: {e}")
                # Generate synthetic data
                await self.generate_synthetic_btc_data()
        else:
            # Fetch from Binance API
            try:
                klines = self.connector.client.futures_klines(
                    symbol=self.symbol,
                    interval=self.timeframe,
                    limit=100
                )
                
                for kline in klines:
                    candle = {
                        'timestamp': kline[0],
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5])
                    }
                    self.market_data_buffer.append(candle)
                    self.data_logger.log_market_data(candle)
                
                console.print(f"[green]✅ Loaded {len(self.market_data_buffer)} BTCUSDT candles from Binance[/green]")
                
            except Exception as e:
                logger.error(f"Error fetching Binance BTCUSDT data: {e}")
                await self.generate_synthetic_btc_data()
    
    async def generate_synthetic_btc_data(self):
        """Generate synthetic BTCUSDT data for testing"""
        console.print("[yellow]🔧 Generating synthetic BTCUSDT data...[/yellow]")
        
        base_price = 50000.0
        for i in range(100):
            # Simple random walk
            if i == 0:
                price = base_price
            else:
                last_price = self.market_data_buffer[-1]['close']
                change = np.random.normal(0, 0.005)  # 0.5% volatility
                price = last_price * (1 + change)
            
            # Generate OHLC
            high = price * (1 + abs(np.random.normal(0, 0.002)))
            low = price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = price * (1 + np.random.normal(0, 0.001))
            
            candle = {
                'timestamp': int((time.time() - (100-i) * 900) * 1000),  # 15min intervals
                'open': open_price,
                'high': max(open_price, price, high),
                'low': min(open_price, price, low),
                'close': price,
                'volume': np.random.uniform(100, 1000)
            }
            
            self.market_data_buffer.append(candle)
            self.data_logger.log_market_data(candle)
        
        # Update current BTC price
        self.connector.current_btc_price = self.market_data_buffer[-1]['close']
        
        console.print(f"[green]✅ Generated {len(self.market_data_buffer)} synthetic BTCUSDT candles[/green]")
    
    async def simulate_market_data(self):
        """Simulate real-time market data for paper trading"""
        # Use historical data and add some random walk
        last_candle = self.market_data_buffer[-1] if self.market_data_buffer else None
        
        if last_candle:
            while self.running:
                # Generate new candle every 15 minutes (or faster for testing)
                await asyncio.sleep(5)  # 5 seconds for testing
                
                # Random walk from last price
                last_price = last_candle['close']
                volatility = 0.001  # 0.1% volatility
                
                open_price = last_price
                close_price = last_price * (1 + np.random.normal(0, volatility))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/2)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/2)))
                volume = last_candle['volume'] * (1 + np.random.normal(0, 0.1))
                
                new_candle = {
                    'timestamp': int(time.time() * 1000),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                }
                
                self.market_data_buffer.append(new_candle)
                self.data_logger.log_market_data(new_candle)
                last_candle = new_candle
    
    async def stream_binance_data(self):
        """Stream real-time market data from Binance WebSocket"""
        try:
            if not self.connector.client:
                console.print("[yellow]⚠️ No Binance client available, falling back to simulation[/yellow]")
                await self.simulate_market_data()
                return
            
            # For now, use periodic API calls instead of WebSocket
            # This is more reliable for testnet
            while self.running:
                try:
                    # Get latest kline data
                    klines = self.connector.client.futures_klines(
                        symbol=self.symbol,
                        interval=self.timeframe,
                        limit=1
                    )
                    
                    if klines:
                        kline = klines[0]
                        candle = {
                            'timestamp': kline[0],
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        }
                        
                        # Only add if it's a new candle
                        if not self.market_data_buffer or candle['timestamp'] > self.market_data_buffer[-1]['timestamp']:
                            self.market_data_buffer.append(candle)
                            self.data_logger.log_market_data(candle)
                            console.print(f"[cyan]📊 New BTCUSDT candle: ${candle['close']:,.2f}[/cyan]")
                    
                    await asyncio.sleep(15)  # Check every 15 seconds
                    
                except Exception as e:
                    logger.error(f"Error streaming Binance data: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"Binance streaming error: {e}")
            # Fall back to simulation
            await self.simulate_market_data()
    
    async def trading_loop(self):
        """Main trading loop"""
        console.print("[cyan]💹 Starting trading loop...[/cyan]")
        
        while self.running:
            try:
                if len(self.market_data_buffer) < 20:  # Need minimum data
                    await asyncio.sleep(1)
                    continue
                
                # Get current market state
                current_data = self.prepare_model_input()
                
                if current_data is not None and self.trading_enabled:
                    # Get model prediction
                    signal = await self.get_model_signal(current_data)
                    
                    # Execute trade based on signal
                    if signal:
                        await self.execute_trade_signal(signal)
                    
                    # Check and update existing positions
                    await self.manage_positions()
                
                # Wait before next iteration
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(5)
    
    def prepare_model_input(self) -> Optional[pd.DataFrame]:
        """Prepare input data for the model"""
        # Need at least 30 data points for technical indicators
        if len(self.market_data_buffer) < 30:
            logger.debug(f"Not enough market data: {len(self.market_data_buffer)} < 30")
            return None
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(list(self.market_data_buffer))
            
            # Validate data
            if df.empty:
                logger.warning("Market data buffer is empty")
                return None
            
            # Ensure all required columns are present and not None
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return None
                if df[col].isnull().any():
                    logger.warning(f"Column {col} contains null values, filling...")
                    df[col] = df[col].ffill().fillna(0)  # Forward fill, then 0 fallback
            
            # Add required columns matching training data format
            df['timestamp'] = df['timestamp'] / 1000  # Convert to seconds
            
            # Ensure timestamp is not None
            if df['timestamp'].isnull().any():
                logger.error("Timestamp column contains null values after conversion")
                return None
            
            # Return last 30 rows (minimum required for technical indicators)
            result = df.tail(30).copy()
            
            # Final validation
            if result.empty or len(result) < 30:
                logger.warning(f"Prepared data insufficient: {len(result)} rows")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparing model input: {e}")
            return None
    
    async def get_model_signal(self, market_data: pd.DataFrame) -> Optional[TradeSignal]:
        """Get trading signal from the model"""
        try:
            # Validate input data
            if market_data is None or len(market_data) < 30:
                logger.warning(f"Insufficient market data for prediction: {len(market_data) if market_data is not None else 'None'}")
                return None
            
            # Ensure data doesn't have NaN values
            if market_data.isnull().any().any():
                logger.warning("Market data contains NaN values")
                market_data = market_data.ffill().fillna(0)
            
            # Create temporary environment for prediction
            env = FuturesTradingEnv(
                df=market_data,
                window_size=30,  # Match the data length we're providing
                initial_equity=self.initial_balance,
                reward_config=self.reward_config
            )
            
            # Wrap environment to convert Dict observations to Box (flattened) format
            # This is needed because the trained model expects Box observations but
            # the environment returns Dict observations
            env = DictToBoxObservationWrapper(env)
            env = wrap_environment_for_algorithm(env, "PPO")
            
            # Get model prediction
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
                
            if obs is None:
                logger.warning("Environment returned None observation")
                return None
                
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Validate action
            if action is None:
                logger.warning("Model returned None action")
                return None
            
            # Parse action
            try:
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'parse_action'):
                    action_dict = env.unwrapped.parse_action(action)
                else:
                    # Fallback parsing
                    if len(action) < 3:
                        logger.warning(f"Action array too short: {len(action)}, expected 3")
                        return None
                    action_dict = {
                        'action_type': int(action[0]) if action[0] is not None else 0,
                        'leverage': float(action[1]) if action[1] is not None else 1.0,
                        'risk_percentage': float(action[2]) if action[2] is not None else 0.01
                    }
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Error parsing action {action}: {e}")
                return None
            
            # Validate action_dict
            if not action_dict or 'action_type' not in action_dict:
                logger.error("Failed to parse action properly")
                return None
            
            # Create signal
            signal = TradeSignal(
                timestamp=datetime.now(),
                action_type=action_dict['action_type'],
                leverage=action_dict['leverage'],
                risk_percentage=action_dict['risk_percentage'],
                confidence=0.8,  # You can calculate this from model
                predicted_return=0.0  # You can calculate expected return
            )
            
            # Log action
            current_price = market_data['close'].iloc[-1]
            balance = await self.get_account_balance()
            
            # Ensure balance is not None
            if balance is None:
                balance = self.cached_balance or self.initial_balance
            
            self.data_logger.log_action(
                signal, current_price, 0.0, balance, len(self.active_trades)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error getting model signal: {e}")
            return None
    
    async def execute_trade_signal(self, signal: TradeSignal):
        """Execute BTCUSDT trade based on model signal"""
        # Risk management checks
        if not await self.check_risk_limits():
            console.print("[yellow]⚠️ Risk limits exceeded, skipping BTCUSDT trade[/yellow]")
            return
        
        current_price = self.market_data_buffer[-1]['close']
        
        # Handle different action types
        if signal.action_type == 0:  # HOLD
            return
        
        elif signal.action_type == 3:  # CLOSE
            # Close all BTCUSDT positions
            for trade_id in list(self.active_trades.keys()):
                await self.close_trade(trade_id)
        
        elif signal.action_type in [1, 2]:  # LONG or SHORT
            # Check if we already have a position
            if len(self.active_trades) >= self.max_open_positions:
                console.print("[yellow]⚠️ Max positions reached[/yellow]")
                return
            
            # Calculate position size
            balance = await self.get_account_balance()
            position_size = await self.calculate_position_size(
                balance, signal.risk_percentage, signal.leverage
            )
            
            if position_size <= 0:
                return
            
            # Calculate stop loss and take profit
            side = "BUY" if signal.action_type == 1 else "SELL"
            stop_loss = self.calculate_stop_loss(current_price, side)
            take_profit = self.calculate_take_profit(current_price, side)
            
            # Ensure leverage is within limits
            max_leverage = self.risk_config.get("max_leverage", 10)
            leverage = min(int(signal.leverage), max_leverage)
            
            # Place BTCUSDT order
            order = await self.connector.place_btcusdt_order(
                side=side,
                quantity=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if 'error' not in order:
                # Create trade record
                trade = ActiveTrade(
                    trade_id=str(order.get('orderId', f"BTCUSDT_{int(time.time())}")),
                    symbol=self.symbol,
                    side=side,
                    entry_price=current_price,
                    entry_time=datetime.now(),
                    quantity=position_size,
                    leverage=leverage,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    status="OPEN"
                )
                
                self.active_trades[trade.trade_id] = trade
                self.data_logger.log_trade(trade)
                
                console.print(f"[green]✅ Opened BTCUSDT {side} position: {position_size} @ ${current_price:,.2f} (Leverage: {leverage}x)[/green]")
            else:
                console.print(f"[red]❌ Failed to place BTCUSDT order: {order['error']}[/red]")
    
    async def manage_positions(self):
        """Manage open positions (trailing stops, etc.)"""
        current_price = self.market_data_buffer[-1]['close']
        
        for trade_id, trade in list(self.active_trades.items()):
            if trade.status != "OPEN":
                continue
            
            # Calculate current PnL
            if trade.side == "BUY":
                pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            else:
                pnl_pct = (trade.entry_price - current_price) / trade.entry_price
            
            # Update trailing stop if in profit
            if pnl_pct > self.trailing_stop_pct:
                new_stop = current_price * (1 - self.trailing_stop_pct) if trade.side == "BUY" else current_price * (1 + self.trailing_stop_pct)
                
                if trade.side == "BUY" and new_stop > trade.stop_loss:
                    trade.stop_loss = new_stop
                    console.print(f"[cyan]📈 Updated trailing stop for {trade_id}: ${new_stop:,.2f}[/cyan]")
                elif trade.side == "SELL" and new_stop < trade.stop_loss:
                    trade.stop_loss = new_stop
                    console.print(f"[cyan]📉 Updated trailing stop for {trade_id}: ${new_stop:,.2f}[/cyan]")
            
            # Check stop loss
            if (trade.side == "BUY" and current_price <= trade.stop_loss) or \
               (trade.side == "SELL" and current_price >= trade.stop_loss):
                await self.close_trade(trade_id, reason="STOP_LOSS")
            
            # Check take profit
            elif (trade.side == "BUY" and current_price >= trade.take_profit) or \
                 (trade.side == "SELL" and current_price <= trade.take_profit):
                await self.close_trade(trade_id, reason="TAKE_PROFIT")
    
    async def close_trade(self, trade_id: str, reason: str = "MANUAL"):
        """Close a specific BTCUSDT trade"""
        if trade_id not in self.active_trades:
            return
        
        trade = self.active_trades[trade_id]
        current_price = self.market_data_buffer[-1]['close']
        
        # Close BTCUSDT position on exchange
        result = await self.connector.close_btcusdt_position()
        
        if 'error' not in result:
            # Update trade record
            trade.exit_price = current_price
            trade.exit_time = datetime.now()
            trade.status = "CLOSED"
            
            # Calculate PnL for BTCUSDT
            if trade.side == "BUY":
                trade.pnl = (current_price - trade.entry_price) * trade.quantity
            else:
                trade.pnl = (trade.entry_price - current_price) * trade.quantity
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            if trade.pnl > 0:
                self.performance_metrics['winning_trades'] += 1
                self.consecutive_losses = 0  # Reset consecutive losses on win
            else:
                self.performance_metrics['losing_trades'] += 1
                self.consecutive_losses += 1  # Increment consecutive losses
            self.performance_metrics['total_pnl'] += trade.pnl
            
            # Log trade
            self.data_logger.log_trade(trade)
            
            # Remove from active trades
            del self.active_trades[trade_id]
            
            console.print(f"[{'green' if trade.pnl > 0 else 'red'}]💰 Closed BTCUSDT {trade.side} position: PnL ${trade.pnl:,.2f} ({reason})[/{'green' if trade.pnl > 0 else 'red'}]")
        else:
            console.print(f"[red]❌ Failed to close BTCUSDT position: {result.get('error', 'Unknown error')}[/red]")
    
    async def check_risk_limits(self) -> bool:
        """Enhanced risk limits check before opening new positions"""
        # Emergency halt check (highest priority)
        if self.is_emergency_halted:
            return False
            
        if not self.trading_enabled:
            return False
        
        try:
            # Ensure we have fresh balance data
            current_balance = await self.get_account_balance()
            
            # 0. Critical Net Worth Protection Check
            net_worth_loss_pct = (self.initial_balance - current_balance) / self.initial_balance
            if net_worth_loss_pct >= self.net_worth_protection_threshold:
                console.print(f"[red]🚨 CRITICAL: Net worth loss {net_worth_loss_pct*100:.1f}% - blocking new positions[/red]")
                return False
            
            # 1. Daily loss limit check
            daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance if self.daily_start_balance > 0 else 0
            if self.daily_pnl < 0 and daily_loss_pct > self.max_daily_loss:
                console.print(f"[red]❌ Daily loss limit exceeded: {daily_loss_pct*100:.2f}%[/red]")
                return False
            
            # 2. Maximum open positions check
            if len(self.active_trades) >= self.max_open_positions:
                console.print(f"[yellow]⚠️ Maximum open positions reached: {len(self.active_trades)}/{self.max_open_positions}[/yellow]")
                return False
            
            # 3. Minimum balance check
            if current_balance < self.initial_balance * 0.1:  # Minimum 10% of initial balance
                console.print(f"[red]❌ Balance too low: ${current_balance:,.2f}[/red]")
                return False
            
            # 4. Consecutive losses check
            if self.consecutive_losses >= self.max_consecutive_losses:
                console.print(f"[red]❌ Too many consecutive losses: {self.consecutive_losses}[/red]")
                return False
            
            # 5. Total exposure check
            total_exposure = 0
            for trade in self.active_trades.values():
                if trade.status == "OPEN":
                    current_price = self.market_data_buffer[-1]['close'] if self.market_data_buffer else trade.entry_price
                    exposure = trade.quantity * current_price
                    total_exposure += exposure
            
            exposure_pct = total_exposure / current_balance if current_balance > 0 else 0
            if exposure_pct > 1.0:  # 100% exposure limit
                console.print(f"[red]❌ Total exposure too high: {exposure_pct*100:.1f}%[/red]")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk limits check: {e}")
            return False
    
    async def get_account_balance(self) -> float:
        """Get current account balance (uses cached value for performance)"""
        current_time = time.time()
        
        # If cached balance is recent (less than 60 seconds old), use it
        if current_time - self.last_balance_update < 60:
            return self.cached_balance
        
        # Otherwise, force update
        await self.update_account_balance()
        return self.cached_balance
    
    async def calculate_position_size(self, balance: float, risk_pct: float, leverage: float) -> float:
        """Calculate position size based on risk management"""
        # Maximum position size based on balance
        max_position_value = balance * self.max_position_size
        
        # Position size based on risk percentage
        risk_amount = balance * risk_pct
        current_price = self.market_data_buffer[-1]['close']
        
        # Calculate quantity
        position_value = min(max_position_value, risk_amount * leverage)
        quantity = position_value / current_price
        
        # Round to appropriate decimals
        return round(quantity, 3)
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price based on configuration"""
        stop_distance = entry_price * self.stop_loss_pct
        
        if side == "BUY":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price based on configuration"""
        tp_distance = entry_price * self.take_profit_pct
        
        if side == "BUY":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance
    
    async def performance_monitor(self):
        """Monitor and log performance metrics with enhanced balance tracking"""
        console.print("[cyan]📊 Starting performance monitor...[/cyan]")
        
        while self.running:
            try:
                # Use cached balance for performance
                balance = self.cached_balance
                
                # Calculate open PnL
                open_pnl = 0.0
                current_price = self.market_data_buffer[-1]['close'] if self.market_data_buffer else 0
                
                for trade in self.active_trades.values():
                    if trade.status == "OPEN":
                        if trade.side == "BUY":
                            open_pnl += (current_price - trade.entry_price) * trade.quantity
                        else:
                            open_pnl += (trade.entry_price - current_price) * trade.quantity
                
                # Calculate metrics
                total_trades = self.performance_metrics['total_trades']
                win_rate = (self.performance_metrics['winning_trades'] / total_trades * 100) if total_trades > 0 else 0
                
                # Update max drawdown
                current_equity = balance + open_pnl
                if current_equity > self.performance_metrics['peak_balance']:
                    self.performance_metrics['peak_balance'] = current_equity
                
                drawdown = (self.performance_metrics['peak_balance'] - current_equity) / self.performance_metrics['peak_balance']
                if drawdown > self.performance_metrics['max_drawdown']:
                    self.performance_metrics['max_drawdown'] = drawdown
                
                # Log performance with enhanced data
                perf_data = {
                    'timestamp': datetime.now(),
                    'total_balance': balance,
                    'available_balance': balance,
                    'open_pnl': open_pnl,
                    'realized_pnl': self.performance_metrics['total_pnl'],
                    'daily_pnl': self.daily_pnl,
                    'total_trades': total_trades,
                    'winning_trades': self.performance_metrics['winning_trades'],
                    'losing_trades': self.performance_metrics['losing_trades'],
                    'win_rate': win_rate,
                    'sharpe_ratio': 0.0,  # Calculate if needed
                    'max_drawdown': self.performance_metrics['max_drawdown'],
                    'consecutive_losses': self.consecutive_losses,
                    'trading_enabled': self.trading_enabled
                }
                
                self.data_logger.log_performance(perf_data)
                
                # Display performance every 60 seconds, risk status every 30 seconds
                current_time = time.time()
                if int(current_time) % 60 == 0:  # Every minute
                    self.display_performance(perf_data)
                elif int(current_time) % 30 == 0:  # Every 30 seconds
                    self.display_risk_status()
                
                # Wait 10 seconds before next update (faster for better monitoring)
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(30)
    
    def display_performance(self, perf_data: Dict):
        """Display performance metrics with enhanced balance and risk info"""
        table = Table(title="BTCUSDT Live Trading Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Basic performance metrics
        table.add_row("Balance", f"${perf_data['total_balance']:,.2f}")
        table.add_row("Open PnL", f"${perf_data['open_pnl']:,.2f}")
        table.add_row("Realized PnL", f"${perf_data['realized_pnl']:,.2f}")
        table.add_row("Daily PnL", f"${self.daily_pnl:,.2f}")
        table.add_row("Total Trades", str(perf_data['total_trades']))
        table.add_row("Win Rate", f"{perf_data['win_rate']:.1f}%")
        table.add_row("Max Drawdown", f"{perf_data['max_drawdown']*100:.1f}%")
        table.add_row("Active Positions", str(len(self.active_trades)))
        
        # Risk metrics
        daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance * 100 if self.daily_start_balance > 0 else 0
        table.add_row("Daily Loss %", f"{daily_loss_pct:.2f}%")
        table.add_row("Consecutive Losses", str(self.consecutive_losses))
        
        # Trading status
        status_color = "green" if self.trading_enabled else "red"
        status_text = "ENABLED" if self.trading_enabled else "DISABLED"
        table.add_row("Trading Status", f"[{status_color}]{status_text}[/{status_color}]")
        
        # Balance update info
        time_since_update = int(time.time() - self.last_balance_update)
        table.add_row("Balance Age", f"{time_since_update}s ago")
        
        console.print(table)
    
    async def continuous_training_loop(self):
        """Continuously train the model with new data"""
        console.print("[cyan]🧠 Starting continuous training loop...[/cyan]")
        
        while self.running:
            try:
                # Check if it's time to train
                if time.time() - self.last_training_time < self.training_interval:
                    await asyncio.sleep(60)
                    continue
                
                # Check if we have enough new data
                if len(self.training_data_buffer) < self.min_training_samples:
                    await asyncio.sleep(60)
                    continue
                
                console.print("[yellow]🔄 Starting model training update...[/yellow]")
                
                # Prepare training data
                training_df = pd.DataFrame(self.training_data_buffer)
                
                # Create training environment
                env = FuturesTradingEnv(
                    df=training_df,
                    window_size=20,
                    initial_equity=self.initial_balance,
                    reward_config=self.reward_config
                )
                env = wrap_environment_for_algorithm(env, "PPO")
                
                # Train model (small number of steps for online learning)
                self.model.learn(total_timesteps=10000, reset_num_timesteps=False)
                
                # Save updated model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"models/live_updated_{timestamp}.zip"
                self.model.save(model_path)
                
                console.print(f"[green]✅ Model updated and saved to {model_path}[/green]")
                
                # Clear training buffer
                self.training_data_buffer = []
                self.last_training_time = time.time()
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

def main():
    """Main entry point for BTCUSDT Perpetual Futures Trading System"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BTCUSDT Perpetual Futures Live Trading System")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        console.print(f"[red]❌ Model file not found: {args.model}[/red]")
        return
    
    # Load reward config if available
    try:
        from improved_reward_configs import TREND_RIDER_CONFIG
        reward_config = TREND_RIDER_CONFIG
        console.print("[green]✅ Loaded TREND_RIDER reward configuration[/green]")
    except:
        reward_config = None
        console.print("[yellow]⚠️ Using default reward configuration[/yellow]")
    
    # Create BTCUSDT trading system
    trading_system = LiveBTCUSDTTradingSystem(
        model_path=args.model,
        config_path=args.config,
        reward_config=reward_config
    )
    
    console.print("[cyan]🔧 System Configuration:[/cyan]")
    console.print(f"  - Trading Config: {trading_system.trading_config}")
    console.print(f"  - Risk Config: {trading_system.risk_config}")
    console.print(f"  - Training Config: {trading_system.training_config}")
    
    # Run async event loop
    try:
        asyncio.run(trading_system.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ BTCUSDT Trading system stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ BTCUSDT Trading system error: {e}[/red]")
        logger.exception("Trading system error")

if __name__ == "__main__":
    main()