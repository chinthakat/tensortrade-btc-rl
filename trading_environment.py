"""
Binance Futures Trading Environment with TensorTrade Integration
A custom gymnasium environment for cryptocurrency futures trading using deep reinforcement learning.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List
import logging
from datetime import datetime, timezone
import csv
import os
from collections import deque
import warnings

# Required dependencies - fail fast if not available
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("scikit-learn is required but not installed. Please install with: pip install scikit-learn")

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("pandas_ta is required but not installed. Please install with: pip install pandas_ta")

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingMetrics:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        if len(returns) < 2:
            return 0.0
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(returns) > target_return else 0.0
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        if downside_deviation == 0:
            return 0.0
        return (np.mean(returns) - target_return) / downside_deviation
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)

class TradeLogger:
    """Comprehensive trade logging system"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.trades = []
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize CSV log file with headers"""
        headers = [
            'trade_id', 'training_step', 'training_iteration', 'entry_datetime', 
            'close_datetime', 'side', 'entry_action', 'entry_price', 'close_price',
            'net_pnl', 'close_reward', 'entry_net_worth', 'close_net_worth',
            'trade_duration_hours', 'status', 'win_loss', 'position_size',
            'fees_paid', 'stop_loss_price', 'take_profit_price', 'close_reason'
        ]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a completed trade"""
        self.trades.append(trade_data)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_data.get('trade_id', ''),
                trade_data.get('training_step', ''),
                trade_data.get('training_iteration', ''),
                trade_data.get('entry_datetime', ''),
                trade_data.get('close_datetime', ''),
                trade_data.get('side', ''),
                trade_data.get('entry_action', ''),
                trade_data.get('entry_price', ''),
                trade_data.get('close_price', ''),
                trade_data.get('net_pnl', ''),
                trade_data.get('close_reward', ''),
                trade_data.get('entry_net_worth', ''),
                trade_data.get('close_net_worth', ''),
                trade_data.get('trade_duration_hours', ''),
                trade_data.get('status', ''),
                trade_data.get('win_loss', ''),
                trade_data.get('position_size', ''),
                trade_data.get('fees_paid', ''),
                trade_data.get('stop_loss_price', ''),
                trade_data.get('take_profit_price', ''),
                trade_data.get('close_reason', '')
            ])

class FuturesTradingEnv(gym.Env):
    """
    Custom Binance Futures Trading Environment with TensorTrade influence
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_equity: float = 10000.0,
        max_leverage: float = 25.0,
        maker_fee: float = 0.0002,  # 0.02%
        taker_fee: float = 0.0004,  # 0.04%
        funding_rate: float = 0.0001,  # 0.01% per 8 hours
        window_size: int = 60,
        stop_loss_pct: float = 0.02,  # 2% - fallback for fixed mode
        take_profit_pct: float = 0.04,  # 4% - fallback for fixed mode
        # Dynamic Stop-Loss and Take-Profit Configuration
        use_dynamic_stops: bool = True,  # Enable ATR-based dynamic stops
        atr_stop_loss_multiplier: float = 2.0,  # Stop-loss = ATR * multiplier
        atr_take_profit_multiplier: float = 3.0,  # Take-profit = ATR * multiplier
        min_stop_loss_pct: float = 0.005,  # 0.5% minimum stop-loss
        max_stop_loss_pct: float = 0.08,  # 8% maximum stop-loss
        min_take_profit_pct: float = 0.01,  # 1% minimum take-profit
        max_take_profit_pct: float = 0.15,  # 15% maximum take-profit
        log_file: str = None,
        training_iteration: int = 0,
        training_split_ratio: float = 0.7,  # Use 70% of data for scaler fitting
        training_end_idx: Optional[int] = None,  # Explicit training end index for scaling
        # Liquidation parameters (based on Binance Futures)
        maintenance_margin_rate: float = 0.004,  # 0.4% for most symbols at moderate leverage
        liquidation_fee_rate: float = 0.005,  # 0.5% liquidation fee
        # Enhanced action space control
        use_advanced_action_space: bool = False,  # Toggle between simple and advanced action space
        # Configurable Reward Function Parameters
        reward_config: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.df = df.copy()
        self.initial_equity = initial_equity
        self.max_leverage = max_leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.funding_rate = funding_rate
        self.window_size = window_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Dynamic stop-loss and take-profit configuration
        self.use_dynamic_stops = use_dynamic_stops
        self.atr_stop_loss_multiplier = atr_stop_loss_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.min_stop_loss_pct = min_stop_loss_pct
        self.max_stop_loss_pct = max_stop_loss_pct
        self.min_take_profit_pct = min_take_profit_pct
        self.max_take_profit_pct = max_take_profit_pct
        self.training_iteration = training_iteration
        self.training_split_ratio = training_split_ratio
        self.training_end_idx = training_end_idx
        
        # Liquidation parameters
        self.maintenance_margin_rate = maintenance_margin_rate
        self.liquidation_fee_rate = liquidation_fee_rate
        
        # Enhanced action space control
        self.use_advanced_action_space = use_advanced_action_space
        
        # Configure reward function parameters
        self._setup_reward_config(reward_config)
        
        # Initialize logger
        if log_file:
            self.logger = TradeLogger(log_file)
        else:
            self.logger = None
        
        # Prepare technical indicators with proper scaling
        self._prepare_features(training_end_idx)
        
        # Trading state
        self.reset()
        
        # Define action space based on configuration
        if self.use_advanced_action_space:
            # Phase 1: Advanced Dict action space (leverage + risk percentage)
            self.action_space = spaces.Dict({
                'leverage': spaces.Box(
                    low=-self.max_leverage, 
                    high=self.max_leverage, 
                    shape=(1,), 
                    dtype=np.float32
                ),
                'risk_percentage': spaces.Box(
                    low=0.01,  # Minimum 1% of equity at risk
                    high=1.0,  # Maximum 100% of equity at risk
                    shape=(1,), 
                    dtype=np.float32
                )
            })
        else:
            # Legacy simple action space: continuous leverage only
            self.action_space = spaces.Box(
                low=-self.max_leverage, 
                high=self.max_leverage, 
                shape=(1,), 
                dtype=np.float32
            )
        
        # Define observation space with enhanced portfolio features
        n_features = self.feature_columns.shape[1]
        self.observation_space = spaces.Dict({
            'market_features': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size, n_features), 
                dtype=np.float32
            ),
            'portfolio_features': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(12,),  # Enhanced portfolio state
                dtype=np.float32
            )
        })
        
        # Enhanced risk management parameters
        self.consecutive_losses = 0
        self.consecutive_loss_threshold = 5  # Trigger enhanced penalties after 5 consecutive losses
        self.severe_loss_threshold = 0.05  # 5% of initial equity - terminate training
        self.moderate_loss_threshold = 0.30  # 30% of initial equity - increase penalties
        self.balance_history = deque(maxlen=20)  # Track balance trend
        self.loss_penalty_multiplier = 1.0  # Dynamic penalty multiplier
    
    def _prepare_features(self, training_end_idx: Optional[int] = None):
        """
        Prepare technical indicators and features using pandas_ta
        
        Args:
            training_end_idx: Index to split training data for proper scaling.
                            If None, uses the first 70% of data for fitting scaler.
        """
        
        df = self.df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Check if we have enough data for technical indicators
        min_data_required = 30  # Minimum data points needed for most indicators
        if len(df) < min_data_required:
            raise ValueError(f"Insufficient data for technical indicators. Need at least {min_data_required} data points, got {len(df)}")
        
        # Technical indicators using pandas_ta
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['ema_10'] = ta.ema(df['close'], length=10)
        df['ema_20'] = ta.ema(df['close'], length=20)
        
        # Momentum indicators
        rsi_result = ta.rsi(df['close'], length=14)
        if rsi_result is None:
            raise ValueError("Failed to calculate RSI. Check if data is valid.")
        df['rsi'] = rsi_result
        
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is None or not isinstance(stoch, pd.DataFrame) or 'STOCHk_14_3_3' not in stoch.columns:
            raise ValueError("Failed to calculate Stochastic oscillator. Check if data is valid.")
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        
        # Volatility indicators
        bb = ta.bbands(df['close'], length=20)
        if bb is None or not isinstance(bb, pd.DataFrame):
            raise ValueError("Failed to calculate Bollinger Bands. Check if data is valid.")
        df['bb_upper'] = bb['BBU_20_2.0']
        df['bb_lower'] = bb['BBL_20_2.0'] 
        df['bb_middle'] = bb['BBM_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
        if atr_result is None:
            raise ValueError("Failed to calculate ATR. Check if data is valid.")
        df['atr'] = atr_result
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is None or not isinstance(macd, pd.DataFrame):
            raise ValueError("Failed to calculate MACD. Check if data is valid.")
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_histogram'] = macd['MACDh_12_26_9']
        
        # Volume indicators
        volume_sma = ta.sma(df['volume'], length=20)
        if volume_sma is None:
            raise ValueError("Failed to calculate volume SMA. Check if data is valid.")
        df['volume_sma'] = volume_sma
        
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position indicators
        df['price_position'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Drop NaN values
        df.dropna(inplace=True)
        logging.info(f"Data shape after dropna: {df.shape}")
        
        # Select feature columns
        feature_cols = [
            'returns', 'log_returns', 'high_low_pct', 'close_open_pct',
            'sma_10', 'sma_20', 'ema_10', 'ema_20', 'rsi', 'stoch_k',
            'bb_width', 'atr', 'macd', 'macd_signal', 'macd_histogram',
            'volume_ratio', 'price_position'
        ]
        
        # Check which feature columns are available
        available_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            logging.warning(f"Missing feature columns: {missing_cols}")
        if not available_cols:
            raise ValueError(f"No feature columns available. Available columns: {list(df.columns)}")
            
        logging.info(f"Using {len(available_cols)} feature columns: {available_cols}")
        
        self.feature_columns = df[available_cols].copy()
        self.price_data = df[['open', 'high', 'low', 'close', 'volume', 'timestamp']].copy()
        
        # Feature scaling using scikit-learn StandardScaler
        self.scaler = StandardScaler()
        
        # Determine training split for scaler fitting
        if training_end_idx is None:
            # Use first 70% of data for fitting scaler (avoid lookahead bias)
            if len(self.feature_columns) == 0:
                raise ValueError("No feature columns available for training. Check if technical indicators were calculated successfully.")
            training_end_idx = int(len(self.feature_columns) * 0.7)
            logging.info(f"Using first {training_end_idx} samples ({training_end_idx/len(self.feature_columns):.1%}) for scaler fitting")
        
        # Fit scaler ONLY on training data (prevents data leakage)
        training_features = self.feature_columns.iloc[:training_end_idx]
        self.scaler.fit(training_features)
        
        # Transform ALL data using the fitted scaler (no refitting)
        self.feature_columns_scaled = pd.DataFrame(
            self.scaler.transform(self.feature_columns),
            columns=self.feature_columns.columns,
            index=self.feature_columns.index
        )
        
        logging.info(f"Feature scaling completed - fitted on {len(training_features)} training samples, "
                    f"applied to {len(self.feature_columns)} total samples")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Trading state
        self.current_step = self.window_size
        self.equity = self.initial_equity
        self.balance = self.initial_equity
        self.position_size = 0.0  # In base currency (BTC)
        self.position_side = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = 0.0
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.leverage = 0.0
        
        # Risk management
        self.stop_loss_price = None
        self.take_profit_price = None
        self.liquidation_price = None  # Real-time liquidation price
        
        # Trade tracking
        self.trade_id = 0
        self.trade_start_step = None
        self.entry_equity = 0.0
        self.total_fees = 0.0
        self.total_funding_costs = 0.0
        
        # Performance tracking
        self.equity_history = deque(maxlen=1000)
        self.returns_history = deque(maxlen=252)  # 1 year of daily returns
        self.max_equity = self.initial_equity
        self.liquidated = False
        
        # Enhanced balance and risk tracking
        self.balance_history = deque(maxlen=20)
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.loss_penalty_multiplier = 1.0
        self.last_trade_pnl = 0.0
        self.total_realized_pnl = 0.0
        self.balance_trend_slope = 0.0  # Track if balance is declining
        
        # Risk management flags
        self.severe_drawdown_triggered = False
        self.moderate_drawdown_triggered = False
        
        # Episode tracking
        self.episode_trades = 0
        self.episode_profit = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Parse action based on action space type
        if self.use_advanced_action_space:
            # Advanced action space: Dict with leverage and risk_percentage
            if isinstance(action, dict):
                leverage = action['leverage'][0] if isinstance(action['leverage'], np.ndarray) else action['leverage']
                risk_percentage = action['risk_percentage'][0] if isinstance(action['risk_percentage'], np.ndarray) else action['risk_percentage']
            else:
                # If action is from wrapper, it's a numpy array [leverage, risk_percentage]
                leverage = action[0]
                risk_percentage = action[1]
            
            # Clip values to valid ranges
            leverage = float(np.clip(leverage, -self.max_leverage, self.max_leverage))
            risk_percentage = float(np.clip(risk_percentage, 0.01, 1.0))
        else:
            # Legacy action space: single leverage value
            if isinstance(action, dict):
                # This shouldn't happen in legacy mode, but handle gracefully
                leverage = action.get('leverage', 0.0)
                if isinstance(leverage, (list, np.ndarray)):
                    leverage = leverage[0]
            else:
                leverage = action[0] if isinstance(action, (list, np.ndarray)) else action
            leverage = float(np.clip(leverage, -self.max_leverage, self.max_leverage))
            risk_percentage = 1.0  # Use full equity (legacy behavior)
        
        # Store previous state
        prev_equity = self.equity
        
        # Get current price
        current_price = self.price_data.iloc[self.current_step]['close']
        current_high = self.price_data.iloc[self.current_step]['high']
        current_low = self.price_data.iloc[self.current_step]['low']
        
        # Update unrealized PnL
        if self.position_size != 0:
            if self.position_side == 1:  # Long
                self.unrealized_pnl = self.position_size * (current_price - self.entry_price)
            else:  # Short
                self.unrealized_pnl = self.position_size * (self.entry_price - current_price)
        else:
            self.unrealized_pnl = 0.0
        
        # Update equity
        self.equity = self.balance + self.unrealized_pnl
        
        # Check for liquidation
        liquidation_triggered = self._check_liquidation(current_low, current_high)
        
        # Check for stop loss / take profit
        sl_tp_triggered = self._check_stop_loss_take_profit(current_low, current_high)
        
        # Execute new action if not liquidated
        if not liquidation_triggered and not sl_tp_triggered:
            self._execute_action(leverage, risk_percentage, current_price)
        
        # Update tracking
        self.equity_history.append(self.equity)
        self.balance_history.append(self.balance)
        
        if len(self.equity_history) > 1:
            equity_return = (self.equity - self.equity_history[-2]) / self.equity_history[-2]
            self.returns_history.append(equity_return)
        
        # Calculate balance trend
        if len(self.balance_history) >= 5:
            # Simple linear regression slope to detect declining balance
            x = np.arange(len(self.balance_history))
            y = np.array(self.balance_history)
            slope = np.polyfit(x, y, 1)[0]
            self.balance_trend_slope = slope / self.initial_equity  # Normalized slope
        
        self.max_equity = max(self.max_equity, self.equity)
        
        # Check for critical loss thresholds
        equity_ratio = self.equity / self.initial_equity
        terminated_early = False
        
        if equity_ratio <= self.severe_loss_threshold:
            # Terminate training if balance drops below 5%
            terminated_early = True
            self.severe_drawdown_triggered = True
            logging.warning(f"SEVERE LOSS: Equity dropped to {equity_ratio:.1%} of initial. Terminating training.")
        elif equity_ratio <= self.moderate_loss_threshold and not self.moderate_drawdown_triggered:
            # Trigger enhanced penalties if balance drops below 30%
            self.moderate_drawdown_triggered = True
            self.loss_penalty_multiplier = 2.0
            logging.warning(f"MODERATE LOSS: Equity dropped to {equity_ratio:.1%} of initial. Increasing penalties.")
        
        # Calculate reward with enhanced risk management
        reward = self._calculate_enhanced_reward(prev_equity)
        
        # Check terminal conditions
        terminated = self.equity <= 0 or self.liquidated or terminated_early
        truncated = self.current_step >= len(self.price_data) - 1
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, target_leverage: float, risk_percentage: float, current_price: float):
        """Execute trading action based on target leverage and risk percentage"""
        # Enhanced risk controls
        max_risk_per_trade = 0.02  # Maximum 2% risk per trade
        risk_percentage = min(risk_percentage, max_risk_per_trade)
        
        # Debug: Log action execution every 50 steps
        if hasattr(self, 'logger') and self.logger and self.current_step % 50 == 0:
            debug_action = {
                'trade_id': f"ACTION_{self.current_step}",
                'training_step': self.current_step,
                'training_iteration': getattr(self, 'training_iteration', 0),
                'entry_datetime': self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}",
                'close_datetime': '',
                'side': f"ACTION_CALLED",
                'entry_action': f"leverage: {target_leverage:.4f}, risk: {risk_percentage:.4f}",
                'entry_price': current_price,
                'close_price': '',
                'net_pnl': 0,
                'close_reward': 0,
                'entry_net_worth': self.equity,
                'close_net_worth': self.equity,
                'trade_duration_hours': 0,
                'status': f"equity: {self.equity:.2f}",
                'win_loss': 'ACTION_DEBUG',
                'position_size': self.position_size,
                'fees_paid': 0,
                'stop_loss_price': '',
                'take_profit_price': '',
                'close_reason': 'ACTION_ENTRY'
            }
            self.logger.log_trade(debug_action)
        
        # Limit leverage based on current equity ratio
        equity_ratio = self.equity / self.initial_equity
        if equity_ratio < 0.5:  # If down 50%, reduce max leverage
            max_allowed_leverage = self.max_leverage * 0.5
        elif equity_ratio < 0.2:  # If down 80%, reduce leverage drastically
            max_allowed_leverage = self.max_leverage * 0.2
        else:
            max_allowed_leverage = self.max_leverage
        
        target_leverage = np.clip(target_leverage, -max_allowed_leverage, max_allowed_leverage)
        
        # Calculate position size based on risk percentage and leverage
        risk_equity = self.equity * risk_percentage  # Amount of equity to risk
        target_position_value = target_leverage * risk_equity
        target_position_size = target_position_value / current_price if current_price > 0 else 0
        
        # Additional safety: limit position size to reasonable fraction of equity
        max_position_value = self.equity * abs(target_leverage) * 0.8  # 80% safety margin
        if abs(target_position_value) > max_position_value:
            target_position_value = np.sign(target_position_value) * max_position_value
            target_position_size = target_position_value / current_price if current_price > 0 else 0
        
        # Calculate trade size needed
        trade_size = target_position_size - self.position_size
        
        # Debug logging to understand trading behavior
        if hasattr(self, 'logger') and self.logger and self.current_step % 100 == 0:  # Log every 100 steps
            debug_data = {
                'trade_id': f"DEBUG_{self.current_step}",
                'training_step': self.current_step,
                'training_iteration': getattr(self, 'training_iteration', 0),
                'entry_datetime': self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}",
                'close_datetime': '',
                'side': f"target_leverage: {target_leverage:.4f}, risk_pct: {risk_percentage:.4f}",
                'entry_action': f"target_pos: {target_position_size:.6f}, current_pos: {self.position_size:.6f}",
                'entry_price': current_price,
                'close_price': '',
                'net_pnl': 0,
                'close_reward': 0,
                'entry_net_worth': self.equity,
                'close_net_worth': self.equity,
                'trade_duration_hours': 0,
                'status': f"trade_size: {trade_size:.6f}",
                'win_loss': 'SKIP' if abs(trade_size) <= 0.001 else 'EXECUTE',
                'position_size': self.position_size,
                'fees_paid': 0,
                'stop_loss_price': '',
                'take_profit_price': '',
                'close_reason': 'DEBUG_INFO'
            }
            self.logger.log_trade(debug_data)
        
        if abs(trade_size) > 0.001:  # Only trade if significant change
            # Efficient trade execution - single order instead of close + open
            self._execute_efficient_trade(target_position_size, current_price)
    
    def _execute_efficient_trade(self, target_position_size: float, current_price: float):
        """
        Execute trade efficiently by calculating net position change.
        This simulates real exchange behavior where position flips are handled as single orders.
        """
        trade_size = target_position_size - self.position_size
        
        if abs(trade_size) < 0.001:
            return  # No significant trade needed
        
        # Calculate realized PnL if we have an existing position being modified
        realized_pnl = 0.0
        
        if self.position_size != 0:
            # We're modifying an existing position
            if np.sign(target_position_size) != np.sign(self.position_size):
                # Position flip: calculate PnL on the closed portion
                if self.position_side == 1:  # Closing long
                    realized_pnl = self.position_size * (current_price - self.entry_price)
                else:  # Closing short
                    realized_pnl = self.position_size * (self.entry_price - current_price)
                
                # Update realized PnL tracking
                self.total_realized_pnl += realized_pnl
                self.last_trade_pnl = realized_pnl
                
                # Update consecutive tracking
                if realized_pnl > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
            
            elif abs(self.position_size) > abs(target_position_size):
                # Partial close: calculate PnL on the reduced portion
                position_reduction = self.position_size - target_position_size
                if self.position_side == 1:  # Reducing long
                    realized_pnl = position_reduction * (current_price - self.entry_price)
                else:  # Reducing short
                    realized_pnl = position_reduction * (self.entry_price - current_price)
                
                self.total_realized_pnl += realized_pnl
                self.last_trade_pnl = realized_pnl
        
        # Calculate trading fees on the net trade volume (efficient!)
        trade_value = abs(trade_size * current_price)
        trading_fee = trade_value * self.taker_fee
        
        # Update balance with realized PnL and fees
        self.balance += realized_pnl - trading_fee
        self.total_fees += trading_fee
        
        # Update position
        old_position_size = self.position_size
        self.position_size = target_position_size
        
        if abs(self.position_size) > 0.001:
            # We have a position
            self.position_side = 1 if self.position_size > 0 else -1
            
            # Update entry price for new or flipped positions
            if old_position_size == 0 or np.sign(old_position_size) != np.sign(self.position_size):
                # New position or position flip
                self.entry_price = current_price
                self.trade_id += 1
                self.trade_start_step = self.current_step
                self.entry_equity = self.equity
            
            # Update margin and risk management
            self.margin_used = abs(self.position_size * current_price) / self.leverage if self.leverage > 0 else 0
            self._calculate_liquidation_price()
            self._update_stop_loss_take_profit(current_price)
        else:
            # No position
            self.position_side = 0
            self.entry_price = 0.0
            self.margin_used = 0.0
            self.liquidation_price = None
            self.stop_loss_price = None
            self.take_profit_price = None
        
        # Log the efficient trade
        if hasattr(self, 'logger') and self.logger:
            action_type = "FLIP" if old_position_size != 0 and np.sign(old_position_size) != np.sign(self.position_size) else "ADJUST"
            if old_position_size == 0:
                action_type = "OPEN"
            elif abs(self.position_size) < 0.001:
                action_type = "CLOSE"
            
            # Create trade data dictionary for logging
            trade_data = {
                'trade_id': self.trade_id,
                'training_step': self.current_step,
                'training_iteration': getattr(self, 'training_iteration', 0),
                'entry_datetime': self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}",
                'close_datetime': '',  # Will be filled when trade closes
                'side': 'LONG' if self.position_size > 0 else 'SHORT' if self.position_size < 0 else 'FLAT',
                'entry_action': action_type,
                'entry_price': current_price,
                'close_price': '',  # Will be filled when trade closes
                'net_pnl': realized_pnl,
                'close_reward': 0,  # Will be filled when trade closes
                'entry_net_worth': self.equity,
                'close_net_worth': self.equity,
                'trade_duration_hours': 0,
                'status': 'OPEN' if abs(self.position_size) > 0.001 else 'CLOSED',
                'win_loss': 'WIN' if realized_pnl > 0 else 'LOSS' if realized_pnl < 0 else 'NEUTRAL',
                'position_size': self.position_size,
                'fees_paid': trading_fee,
                'stop_loss_price': getattr(self, 'stop_loss_price', ''),
                'take_profit_price': getattr(self, 'take_profit_price', ''),
                'close_reason': action_type
            }
            
            self.logger.log_trade(trade_data)
    
    def _update_stop_loss_take_profit(self, current_price: float):
        """Update stop-loss and take-profit prices with dynamic ATR-based calculation"""
        if abs(self.position_size) < 0.001:
            self.stop_loss_price = None
            self.take_profit_price = None
            return
        
        if self.use_dynamic_stops:
            # Get current ATR value for dynamic calculation
            current_atr = self._get_current_atr(current_price)
            
            # Calculate dynamic stop-loss and take-profit percentages
            dynamic_stop_pct, dynamic_tp_pct = self._calculate_dynamic_stops(current_atr, current_price)
            
            if self.position_side == 1:  # Long
                self.stop_loss_price = current_price * (1 - dynamic_stop_pct)
                self.take_profit_price = current_price * (1 + dynamic_tp_pct)
            else:  # Short
                self.stop_loss_price = current_price * (1 + dynamic_stop_pct)
                self.take_profit_price = current_price * (1 - dynamic_tp_pct)
        else:
            # Fallback to fixed percentages
            if self.position_side == 1:  # Long
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                self.take_profit_price = current_price * (1 + self.take_profit_pct)
            else:  # Short
                self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
                self.take_profit_price = current_price * (1 - self.take_profit_pct)
    
    def _get_current_atr(self, current_price: float) -> float:
        """Get current ATR value from the feature data (before scaling)"""
        try:
            # ATR is in the original feature columns, not price_data
            current_atr = self.feature_columns.iloc[self.current_step]['atr']
            
            # Handle NaN or invalid ATR values
            if pd.isna(current_atr) or current_atr <= 0:
                # Fallback: use recent ATR or calculate a simple estimate
                recent_atr_values = self.feature_columns.iloc[max(0, self.current_step-10):self.current_step+1]['atr'].dropna()
                if len(recent_atr_values) > 0:
                    current_atr = recent_atr_values.iloc[-1]
                else:
                    # Final fallback: estimate ATR as 1% of current price
                    current_atr = current_price * 0.01
                    
            return float(current_atr)
        except (KeyError, IndexError):
            # Fallback if ATR column doesn't exist or index error
            return current_price * 0.01
    
    def _calculate_dynamic_stops(self, atr: float, current_price: float) -> tuple[float, float]:
        """
        Calculate dynamic stop-loss and take-profit percentages based on ATR
        
        Args:
            atr: Current Average True Range value
            current_price: Current market price
            
        Returns:
            Tuple of (stop_loss_percentage, take_profit_percentage)
        """
        # Convert ATR to percentage of current price
        atr_percentage = atr / current_price
        
        # Calculate dynamic stop-loss and take-profit
        dynamic_stop_pct = atr_percentage * self.atr_stop_loss_multiplier
        dynamic_tp_pct = atr_percentage * self.atr_take_profit_multiplier
        
        # Apply minimum and maximum bounds for risk management
        dynamic_stop_pct = max(self.min_stop_loss_pct, 
                              min(self.max_stop_loss_pct, dynamic_stop_pct))
        
        dynamic_tp_pct = max(self.min_take_profit_pct, 
                            min(self.max_take_profit_pct, dynamic_tp_pct))
        
        return dynamic_stop_pct, dynamic_tp_pct
    
    def get_dynamic_stops_info(self) -> Dict[str, float]:
        """Get information about current dynamic stop-loss and take-profit settings"""
        if not self.use_dynamic_stops:
            return {
                'mode': 'fixed',
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }
        
        try:
            current_price = self.price_data.iloc[self.current_step]['close']
            current_atr = self._get_current_atr(current_price)
            atr_percentage = current_atr / current_price
            
            dynamic_stop_pct, dynamic_tp_pct = self._calculate_dynamic_stops(current_atr, current_price)
            
            return {
                'mode': 'dynamic',
                'current_atr': current_atr,
                'atr_percentage': atr_percentage,
                'dynamic_stop_loss_pct': dynamic_stop_pct,
                'dynamic_take_profit_pct': dynamic_tp_pct,
                'atr_stop_multiplier': self.atr_stop_loss_multiplier,
                'atr_tp_multiplier': self.atr_take_profit_multiplier,
                'stop_loss_price': self.stop_loss_price,
                'take_profit_price': self.take_profit_price
            }
        except Exception as e:
            return {'mode': 'dynamic', 'error': str(e)}
    
    def _open_or_adjust_position(self, target_position_size: float, current_price: float):
        """
        DEPRECATED: This method is kept for compatibility but is inefficient.
        Use _execute_efficient_trade instead for normal position changes.
        """
        logging.warning("Using deprecated _open_or_adjust_position. Consider using _execute_efficient_trade instead.")
        
        # For compatibility, just call the efficient method
        self._execute_efficient_trade(target_position_size, current_price)
    
    def _close_position(self, current_price: float, reason: str):
        """Close current position"""
        if self.position_size == 0:
            return
        
        # Calculate PnL
        if self.position_side == 1:  # Long
            pnl = self.position_size * (current_price - self.entry_price)
        else:  # Short
            pnl = self.position_size * (self.entry_price - current_price)
        
        # Calculate fees
        trade_value = abs(self.position_size * current_price)
        exit_fee = trade_value * self.taker_fee
        
        # Update balance
        self.balance += pnl - exit_fee
        self.total_fees += exit_fee
        self.total_realized_pnl += pnl
        self.last_trade_pnl = pnl
        
        # Update consecutive loss/win tracking
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # Increase penalty multiplier for consecutive losses
            if self.consecutive_losses >= self.consecutive_loss_threshold:
                self.loss_penalty_multiplier = min(3.0, 1.0 + (self.consecutive_losses - self.consecutive_loss_threshold) * 0.2)
        
        # Calculate funding costs (simplified)
        if self.trade_start_step:
            hours_held = (self.current_step - self.trade_start_step) * 0.25  # 15min intervals
            funding_periods = int(hours_held / 8)  # Funding every 8 hours
            funding_cost = abs(self.position_size * self.entry_price) * self.funding_rate * funding_periods
            if self.position_side == -1:  # Short positions typically pay funding
                self.balance -= funding_cost
                self.total_funding_costs += funding_cost
        
        # Log trade
        if self.logger:
            self._log_trade(current_price, pnl, reason)
        
        # Reset position
        self.position_size = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.leverage = 0.0
        self.stop_loss_price = None
        self.take_profit_price = None
        self.liquidation_price = None
        
        # Update episode stats
        self.episode_trades += 1
        self.episode_profit += pnl
    
    def _log_trade(self, exit_price: float, pnl: float, reason: str):
        """Log completed trade"""
        if not self.trade_start_step:
            return
        
        duration_steps = self.current_step - self.trade_start_step
        duration_hours = duration_steps * 0.25  # 15min intervals
        
        entry_datetime = pd.to_datetime(
            self.price_data.iloc[self.trade_start_step]['timestamp'], 
            unit='s'
        ).strftime('%d/%m/%Y %H:%M')
        
        close_datetime = pd.to_datetime(
            self.price_data.iloc[self.current_step]['timestamp'], 
            unit='s'
        ).strftime('%d/%m/%Y %H:%M')
        
        trade_data = {
            'trade_id': f"TRADE_{self.trade_id:05d}",
            'training_step': self.current_step,
            'training_iteration': self.training_iteration,
            'entry_datetime': entry_datetime,
            'close_datetime': close_datetime,
            'side': 'LONG' if self.position_side == 1 else 'SHORT',
            'entry_action': 'BUY' if self.position_side == 1 else 'SELL',
            'entry_price': self.entry_price,
            'close_price': exit_price,
            'net_pnl': pnl,
            'close_reward': 0.0,  # Will be updated by reward function
            'entry_net_worth': self.entry_equity,
            'close_net_worth': self.equity,
            'trade_duration_hours': duration_hours,
            'status': 'LIQUIDATED' if self.liquidated else 'CLOSED',
            'win_loss': 'WIN' if pnl > 0 else 'LOSS',
            'position_size': abs(self.position_size),
            'fees_paid': self.total_fees,
            'stop_loss_price': self.stop_loss_price or 0.0,
            'take_profit_price': self.take_profit_price or 0.0,
            'close_reason': reason
        }
        
        self.logger.log_trade(trade_data)
        self.trade_id += 1
    
    def _calculate_liquidation_price(self) -> Optional[float]:
        """
        Calculate realistic liquidation price based on maintenance margin requirements.
        
        Liquidation Price Formula for Long Position:
        LP = Entry Price * (1 - Initial Margin Rate + Maintenance Margin Rate - Liquidation Fee Rate)
        
        Liquidation Price Formula for Short Position:
        LP = Entry Price * (1 + Initial Margin Rate - Maintenance Margin Rate + Liquidation Fee Rate)
        
        Returns:
            Liquidation price or None if no position
        """
        if self.position_size == 0 or self.leverage == 0:
            return None
        
        # Calculate initial margin rate from leverage
        initial_margin_rate = 1.0 / self.leverage
        
        if self.position_side == 1:  # Long position
            # For long: liquidation occurs when price falls to maintenance margin level
            liquidation_price = self.entry_price * (
                1 - initial_margin_rate + self.maintenance_margin_rate - self.liquidation_fee_rate
            )
        else:  # Short position
            # For short: liquidation occurs when price rises to maintenance margin level
            liquidation_price = self.entry_price * (
                1 + initial_margin_rate - self.maintenance_margin_rate + self.liquidation_fee_rate
            )
        
        return max(liquidation_price, 0.01)  # Ensure price is positive
    
    def _calculate_maintenance_margin_required(self, mark_price: float) -> float:
        """
        Calculate the maintenance margin required for current position at given mark price.
        
        Args:
            mark_price: Current market price to evaluate margin at
            
        Returns:
            Required maintenance margin in quote currency
        """
        if self.position_size == 0:
            return 0.0
        
        position_value = abs(self.position_size * mark_price)
        return position_value * self.maintenance_margin_rate
    
    def _calculate_margin_balance(self, mark_price: float) -> float:
        """
        Calculate current margin balance (wallet balance + unrealized PnL).
        
        Args:
            mark_price: Current market price
            
        Returns:
            Current margin balance
        """
        # Calculate unrealized PnL at mark price
        if self.position_size == 0:
            unrealized_pnl = 0.0
        elif self.position_side == 1:  # Long
            unrealized_pnl = self.position_size * (mark_price - self.entry_price)
        else:  # Short
            unrealized_pnl = self.position_size * (self.entry_price - mark_price)
        
        return self.balance + unrealized_pnl
    
    def position_side_str(self) -> str:
        """Get human-readable position side"""
        if self.position_side == 1:
            return "LONG"
        elif self.position_side == -1:
            return "SHORT"
        else:
            return "FLAT"
    
    def get_liquidation_info(self) -> Dict[str, Any]:
        """
        Get detailed liquidation information for current position.
        
        Returns:
            Dict with liquidation price, margin ratios, and risk metrics
        """
        if self.position_size == 0:
            return {
                "position_status": "No position",
                "liquidation_price": None,
                "margin_ratio": None,
                "liquidation_distance": None
            }
        
        current_price = self.price_data.iloc[self.current_step]['close']
        margin_balance = self._calculate_margin_balance(current_price)
        maintenance_margin_required = self._calculate_maintenance_margin_required(current_price)
        
        # Calculate margin ratio (margin balance / maintenance margin required)
        margin_ratio = margin_balance / maintenance_margin_required if maintenance_margin_required > 0 else float('inf')
        
        # Calculate distance to liquidation price
        if self.liquidation_price:
            liquidation_distance = abs(current_price - self.liquidation_price) / current_price
        else:
            liquidation_distance = None
        
        return {
            "position_status": f"{self.position_side_str()} {abs(self.position_size):.4f}",
            "entry_price": self.entry_price,
            "current_price": current_price,
            "liquidation_price": self.liquidation_price,
            "margin_balance": margin_balance,
            "maintenance_margin_required": maintenance_margin_required,
            "margin_ratio": margin_ratio,
            "liquidation_distance_pct": f"{liquidation_distance:.2%}" if liquidation_distance else None,
            "leverage": self.leverage,
            "position_value": abs(self.position_size * current_price),
            "unrealized_pnl": self.unrealized_pnl,
            "liquidation_risk": "HIGH" if margin_ratio < 1.5 else "MEDIUM" if margin_ratio < 3.0 else "LOW"
        }
    
    def _check_liquidation(self, current_low: float, current_high: float) -> bool:
        """
        Check if position should be liquidated based on realistic maintenance margin requirements.
        
        This implements Binance-style liquidation:
        1. Calculate liquidation price based on maintenance margin
        2. Check if mark price touched liquidation price
        3. Execute liquidation with liquidation fee
        
        Args:
            current_low: Lowest price in current candle
            current_high: Highest price in current candle
            
        Returns:
            True if liquidation was triggered, False otherwise
        """
        if self.position_size == 0 or self.liquidation_price is None:
            return False
        
        liquidation_triggered = False
        liquidation_price_hit = None
        
        if self.position_side == 1:  # Long position
            # Long liquidation: check if low price hit liquidation price
            if current_low <= self.liquidation_price:
                liquidation_triggered = True
                liquidation_price_hit = self.liquidation_price
        else:  # Short position
            # Short liquidation: check if high price hit liquidation price
            if current_high >= self.liquidation_price:
                liquidation_triggered = True
                liquidation_price_hit = self.liquidation_price
        
        if liquidation_triggered:
            # Calculate liquidation with realistic fees and slippage
            position_value = abs(self.position_size * liquidation_price_hit)
            liquidation_fee = position_value * self.liquidation_fee_rate
            
            # Calculate PnL at liquidation price
            if self.position_side == 1:  # Long
                pnl = self.position_size * (liquidation_price_hit - self.entry_price)
            else:  # Short
                pnl = self.position_size * (self.entry_price - liquidation_price_hit)
            
            # Apply liquidation fee (taken from remaining balance)
            pnl -= liquidation_fee
            
            # Update balance with liquidation result
            self.balance += pnl
            self.total_realized_pnl += pnl
            self.last_trade_pnl = pnl
            self.liquidated = True
            
            logging.warning(
                f"LIQUIDATION: {self.position_side_str()} position liquidated at {liquidation_price_hit:.2f}, "
                f"PnL: {pnl:.2f}, Fee: {liquidation_fee:.2f}"
            )
            
            # Log liquidation trade
            if self.logger:
                self._log_trade(liquidation_price_hit, pnl, "LIQUIDATION")
            
            # Reset position
            self.position_size = 0.0
            self.position_side = 0
            self.entry_price = 0.0
            self.margin_used = 0.0
            self.unrealized_pnl = 0.0
            self.leverage = 0.0
            self.stop_loss_price = None
            self.take_profit_price = None
            
            # Update episode stats
            self.episode_trades += 1
            self.episode_profit += pnl
            
            return True
        
        return False
    
    def _check_stop_loss_take_profit(self, current_low: float, current_high: float) -> bool:
        """Check if stop loss or take profit should be triggered"""
        if self.position_size == 0 or not self.stop_loss_price or not self.take_profit_price:
            return False
        
        if self.position_side == 1:  # Long position
            if current_low <= self.stop_loss_price:
                self._execute_efficient_trade(0.0, self.stop_loss_price)  # Close to zero position
                return True
            elif current_high >= self.take_profit_price:
                self._execute_efficient_trade(0.0, self.take_profit_price)  # Close to zero position
                return True
        else:  # Short position
            if current_high >= self.stop_loss_price:
                self._execute_efficient_trade(0.0, self.stop_loss_price)  # Close to zero position
                return True
            elif current_low <= self.take_profit_price:
                self._execute_efficient_trade(0.0, self.take_profit_price)  # Close to zero position
                return True
        
        return False
    
    def _calculate_enhanced_reward(self, prev_equity: float) -> float:
        """
        Calculate reward with enhanced risk management and capped segments
        """
        # === BASE PROFIT/LOSS COMPONENT ===
        if prev_equity > 0:
            equity_change = (self.equity - prev_equity) / prev_equity
        else:
            equity_change = 0.0
        
        # Scale and cap base reward using configurable parameters
        base_reward = equity_change * self.reward_config['base_reward_scale']
        base_reward = np.clip(base_reward, 
                            self.reward_config['base_reward_cap_negative'], 
                            self.reward_config['base_reward_cap_positive'])
        
        # === RISK-ADJUSTED COMPONENTS ===
        risk_penalty = 0.0
        balance_penalty = 0.0
        trend_penalty = 0.0
        
        # 1. Drawdown penalties (progressive) - using configurable parameters
        if len(self.equity_history) > 1:
            drawdown = (self.max_equity - self.equity) / self.max_equity
            
            if drawdown > self.reward_config['severe_drawdown_threshold']:  # >50% drawdown (configurable)
                risk_penalty += self.reward_config['severe_drawdown_penalty']
            elif drawdown > self.reward_config['major_drawdown_threshold']:  # >30% drawdown (configurable)
                risk_penalty += self.reward_config['major_drawdown_penalty']
            elif drawdown > self.reward_config['moderate_drawdown_threshold']:  # >10% drawdown (configurable)
                risk_penalty += self.reward_config['moderate_drawdown_penalty']
            else:
                risk_penalty += drawdown * self.reward_config['linear_drawdown_multiplier']  # Linear penalty for smaller drawdowns
        
        # 2. Balance decline penalties (progressive) - using configurable parameters
        equity_ratio = self.equity / self.initial_equity
        
        if equity_ratio <= self.reward_config['critical_equity_threshold']:  # ≤5% remaining - CRITICAL
            balance_penalty = self.reward_config['critical_equity_penalty']
        elif equity_ratio <= self.reward_config['severe_equity_threshold']:  # ≤10% remaining - SEVERE
            balance_penalty = self.reward_config['severe_equity_penalty']
        elif equity_ratio <= self.reward_config['major_equity_threshold']:  # ≤20% remaining - MAJOR
            balance_penalty = self.reward_config['major_equity_penalty']
        elif equity_ratio <= self.reward_config['moderate_equity_threshold']:  # ≤30% remaining - MODERATE
            balance_penalty = self.reward_config['moderate_equity_penalty']
        elif equity_ratio <= self.reward_config['minor_equity_threshold']:  # ≤50% remaining - MINOR
            balance_penalty = self.reward_config['minor_equity_penalty']
        
        # 3. Consecutive loss penalties - using configurable parameters
        consecutive_loss_penalty = 0.0
        if self.consecutive_losses > 0:
            # Exponential penalty for consecutive losses
            consecutive_loss_penalty = min(
                self.reward_config['consecutive_loss_cap'], 
                self.consecutive_losses ** self.reward_config['consecutive_loss_exponent']
            )
        
        # 4. Balance trend penalty (declining balance over time) - using configurable parameters
        if self.balance_trend_slope < 0:  # Declining balance
            trend_penalty = abs(self.balance_trend_slope) * self.reward_config['trend_penalty_multiplier']  # Scale the slope
            trend_penalty = min(trend_penalty, self.reward_config['trend_penalty_cap'])  # Cap trend penalty
        
        # 5. Volatility penalty - using configurable parameters
        volatility_penalty = 0.0
        if len(self.returns_history) > self.reward_config['volatility_history_threshold']:
            volatility = np.std(list(self.returns_history))
            volatility_penalty = min(volatility * self.reward_config['volatility_multiplier'], 
                                   self.reward_config['volatility_penalty_cap'])  # Cap volatility penalty
        
        # 6. Trading cost penalty - using configurable parameters
        cost_penalty = 0.0
        if hasattr(self, '_last_fees') and self._last_fees > 0:
            cost_penalty = min(self._last_fees * self.reward_config['cost_penalty_multiplier'], 
                             self.reward_config['cost_penalty_cap'])  # Cap cost penalty
        
        # 7. Special penalties - using configurable parameters
        special_penalty = 0.0
        
        # Liquidation penalty
        if self.liquidated:
            special_penalty += self.reward_config['liquidation_penalty']
        
        # Excessive leverage penalty
        if self.leverage > self.reward_config['excessive_leverage_threshold']:
            special_penalty += (self.leverage - self.reward_config['excessive_leverage_threshold']) * self.reward_config['excessive_leverage_multiplier']
        
        # === POSITIVE REWARDS === - using configurable parameters
        positive_bonus = 0.0
        
        # Position holding bonus (encourage longer-term thinking)
        if self.position_size != 0 and self.trade_start_step:
            hold_duration = self.current_step - self.trade_start_step
            if (self.reward_config['optimal_hold_min'] <= hold_duration <= 
                self.reward_config['optimal_hold_max']):  # Optimal hold duration
                positive_bonus += self.reward_config['position_hold_bonus']
            elif hold_duration > self.reward_config['excessive_hold_threshold']:  # Penalize too long holds
                positive_bonus -= self.reward_config['position_hold_penalty']
        
        # Consecutive wins bonus
        if self.consecutive_wins > 0:
            positive_bonus += min(self.consecutive_wins * self.reward_config['consecutive_wins_multiplier'], 
                                self.reward_config['consecutive_wins_cap'])  # Cap wins bonus
        
        # Recovery bonus (recovering from drawdown)
        if len(self.equity_history) > 5:
            recent_improvement = (self.equity - min(list(self.equity_history)[-5:])) / self.initial_equity
            if recent_improvement > self.reward_config['recovery_threshold']:  # Configurable% improvement
                positive_bonus += min(recent_improvement * self.reward_config['recovery_multiplier'], 
                                    self.reward_config['recovery_bonus_cap'])
        
        # === COMBINE ALL COMPONENTS ===
        total_penalty = (
            risk_penalty + 
            balance_penalty + 
            consecutive_loss_penalty + 
            trend_penalty + 
            volatility_penalty + 
            cost_penalty + 
            special_penalty
        )
        
        # Apply dynamic loss multiplier
        total_penalty *= self.loss_penalty_multiplier
        
        # Final reward calculation
        final_reward = base_reward + positive_bonus - total_penalty
        
        # === SEGMENT-BASED CAPPING === - using configurable parameters
        # Cap final reward in different segments for stable learning
        if final_reward > 0:
            final_reward = min(final_reward, self.reward_config['final_reward_positive_cap'])  # Cap positive rewards
        else:
            final_reward = max(final_reward, self.reward_config['final_reward_negative_cap'])  # Cap negative rewards
        
        # Additional severe loss capping
        if equity_ratio <= self.reward_config['severe_equity_threshold']:  # Very low equity (configurable)
            final_reward = max(final_reward, self.reward_config['severe_loss_reward_cap'])  # Allow larger negative rewards for severe losses
        
        return float(final_reward)
    
    def _calculate_reward(self, prev_equity: float) -> float:
        """Legacy reward function - kept for compatibility"""
        return self._calculate_enhanced_reward(prev_equity)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation with enhanced portfolio features"""
        # Market features (windowed historical data)
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        market_features = self.feature_columns_scaled.iloc[start_idx:end_idx].values
        
        # Pad if necessary
        if market_features.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - market_features.shape[0], market_features.shape[1]))
            market_features = np.vstack([padding, market_features])
        
        # Enhanced Portfolio features (12 features total)
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        equity_ratio = self.equity / self.initial_equity if self.initial_equity > 0 else 0
        balance_ratio = self.balance / self.initial_equity if self.initial_equity > 0 else 0
        
        # Calculate recent balance trend
        balance_trend = 0.0
        if len(self.balance_history) >= 3:
            recent_change = (self.balance_history[-1] - self.balance_history[-3]) / self.initial_equity
            balance_trend = np.clip(recent_change, -1.0, 1.0)
        
        portfolio_features = np.array([
            # Core metrics
            equity_ratio,  # Current equity / initial equity
            balance_ratio,  # Current balance / initial equity  
            self.leverage / self.max_leverage if self.max_leverage > 0 else 0,  # Normalized leverage
            
            # PnL and position info
            self.unrealized_pnl / self.initial_equity if self.initial_equity > 0 else 0,  # Normalized unrealized PnL
            self.total_realized_pnl / self.initial_equity if self.initial_equity > 0 else 0,  # Normalized total realized PnL
            self.margin_used / self.initial_equity if self.initial_equity > 0 else 0,  # Normalized margin used
            
            # Risk metrics  
            drawdown,  # Current drawdown from peak
            self.balance_trend_slope,  # Balance trend (normalized slope)
            balance_trend,  # Recent balance change
            
            # Trading behavior metrics
            min(self.consecutive_losses / 10.0, 1.0),  # Normalized consecutive losses (cap at 10)
            min(self.consecutive_wins / 10.0, 1.0),   # Normalized consecutive wins (cap at 10)
            min(self.loss_penalty_multiplier / 3.0, 1.0),  # Normalized penalty multiplier
        ], dtype=np.float32)
        
        # Ensure no NaN or inf values
        portfolio_features = np.nan_to_num(portfolio_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return {
            'market_features': market_features.astype(np.float32),
            'portfolio_features': portfolio_features
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info with enhanced risk metrics"""
        equity_ratio = self.equity / self.initial_equity if self.initial_equity > 0 else 0
        
        return {
            # Core trading info
            'equity': self.equity,
            'balance': self.balance,
            'equity_ratio': equity_ratio,
            'position_size': self.position_size,
            'position_side': self.position_side,
            'leverage': self.leverage,
            'unrealized_pnl': self.unrealized_pnl,
            'total_realized_pnl': self.total_realized_pnl,
            
            # Liquidation info
            'liquidation_price': self.liquidation_price,
            'liquidation_info': self.get_liquidation_info(),
            
            # Risk management info
            'consecutive_losses': self.consecutive_losses,
            'consecutive_wins': self.consecutive_wins,
            'loss_penalty_multiplier': self.loss_penalty_multiplier,
            'balance_trend_slope': self.balance_trend_slope,
            'severe_drawdown_triggered': self.severe_drawdown_triggered,
            'moderate_drawdown_triggered': self.moderate_drawdown_triggered,
            
            # Episode info
            'episode_trades': self.episode_trades,
            'episode_profit': self.episode_profit,
            'liquidated': self.liquidated,
            'current_step': self.current_step,
            'max_steps': len(self.price_data) - 1,
            
            # Performance metrics
            'max_equity': self.max_equity,
            'drawdown': (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0,
            'total_fees': self.total_fees,
            'last_trade_pnl': self.last_trade_pnl
        }
    
    def render(self, mode='human'):
        """Render environment state with enhanced risk information"""
        if mode == 'human':
            current_price = self.price_data.iloc[self.current_step]['close']
            equity_ratio = self.equity / self.initial_equity
            drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
            
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Equity: {self.equity:.2f} ({equity_ratio:.1%} of initial)")
            print(f"Balance: {self.balance:.2f}")
            print(f"Position: {self.position_size:.4f}")
            print(f"Leverage: {self.leverage:.2f}x")
            print(f"Unrealized PnL: {self.unrealized_pnl:.2f}")
            print(f"Total Realized PnL: {self.total_realized_pnl:.2f}")
            print(f"Drawdown: {drawdown:.1%}")
            
            # Liquidation information
            if self.liquidation_price:
                liquidation_distance = abs(current_price - self.liquidation_price) / current_price
                print(f"Liquidation Price: {self.liquidation_price:.2f} ({liquidation_distance:.2%} away)")
                
                liquidation_info = self.get_liquidation_info()
                if liquidation_info.get('margin_ratio'):
                    print(f"Margin Ratio: {liquidation_info['margin_ratio']:.2f} ({liquidation_info['liquidation_risk']} risk)")
            
            print(f"Consecutive Losses: {self.consecutive_losses}")
            print(f"Loss Penalty Multiplier: {self.loss_penalty_multiplier:.2f}x")
            print(f"Balance Trend: {self.balance_trend_slope:.6f}")
            
            # Risk warnings
            if self.severe_drawdown_triggered:
                print("⚠️  SEVERE DRAWDOWN - Training termination triggered!")
            elif self.moderate_drawdown_triggered:
                print("⚠️  MODERATE DRAWDOWN - Enhanced penalties active!")
            elif equity_ratio <= 0.50:
                print("⚠️  Significant losses detected!")
            
            print("-" * 50)
    
    def get_risk_management_summary(self) -> Dict[str, Any]:
        """Get a summary of current risk management settings and status"""
        equity_ratio = self.equity / self.initial_equity if self.initial_equity > 0 else 0
        
        return {
            'risk_thresholds': {
                'severe_loss_threshold': f"{self.severe_loss_threshold:.1%}",
                'moderate_loss_threshold': f"{self.moderate_loss_threshold:.1%}",
                'consecutive_loss_threshold': self.consecutive_loss_threshold
            },
            'current_status': {
                'equity_ratio': f"{equity_ratio:.1%}",
                'consecutive_losses': self.consecutive_losses,
                'consecutive_wins': self.consecutive_wins,
                'loss_penalty_multiplier': f"{self.loss_penalty_multiplier:.2f}x",
                'balance_trend': 'Declining' if self.balance_trend_slope < 0 else 'Stable/Rising',
                'severe_drawdown_triggered': self.severe_drawdown_triggered,
                'moderate_drawdown_triggered': self.moderate_drawdown_triggered
            },
            'warnings': self._get_risk_warnings()
        }
    
    def _get_risk_warnings(self) -> List[str]:
        """Get current risk warnings"""
        warnings = []
        equity_ratio = self.equity / self.initial_equity if self.initial_equity > 0 else 0
        
        if self.severe_drawdown_triggered:
            warnings.append("CRITICAL: Severe drawdown triggered - training will terminate")
        elif equity_ratio <= 0.10:
            warnings.append("WARNING: Equity below 10% of initial")
        elif self.moderate_drawdown_triggered:
            warnings.append("WARNING: Moderate drawdown triggered - enhanced penalties active")
        elif equity_ratio <= 0.50:
            warnings.append("CAUTION: Significant losses detected")
        
        if self.consecutive_losses >= self.consecutive_loss_threshold:
            warnings.append(f"WARNING: {self.consecutive_losses} consecutive losses")
        
        if self.balance_trend_slope < -0.01:  # Significant declining trend
            warnings.append("WARNING: Balance showing declining trend")
        
        if self.loss_penalty_multiplier > 2.0:
            warnings.append("INFO: High penalty multiplier active")
        
        return warnings
    
    def update_scaler_with_new_data(self, new_features: pd.DataFrame):
        """
        Update feature scaling for new incoming data (for live trading)
        
        This method applies the existing fitted scaler to new data without refitting,
        maintaining consistency with the training data scaling.
        
        Args:
            new_features: New feature data to be scaled
            
        Returns:
            pd.DataFrame: Scaled new features
        """
        if not hasattr(self, 'scaler') or self.scaler is None:
            raise ValueError("Scaler not initialized. Call _prepare_features first.")
        
        # Transform new data using existing scaler (NO refitting)
        scaled_features = pd.DataFrame(
            self.scaler.transform(new_features),
            columns=new_features.columns,
            index=new_features.index
        )
        
        return scaled_features
    
    def get_scaler_params(self) -> Dict[str, Any]:
        """
        Get scaler parameters for debugging and validation
        
        Returns:
            Dict containing scaler mean, scale, and training data info
        """
        if not hasattr(self, 'scaler') or self.scaler is None:
            return {"error": "Scaler not initialized"}
        
        return {
            "feature_means": dict(zip(self.feature_columns.columns, self.scaler.mean_)),
            "feature_scales": dict(zip(self.feature_columns.columns, self.scaler.scale_)),
            "n_features": self.scaler.n_features_in_,
            "n_samples_seen": self.scaler.n_samples_seen_
        }
    
    def validate_no_data_leakage(self, validation_start_idx: int) -> Dict[str, Any]:
        """
        Validate that scaling was done properly without data leakage
        
        Args:
            validation_start_idx: Index where validation data starts
            
        Returns:
            Dict with validation results
        """
        if not hasattr(self, 'scaler'):
            return {"error": "Scaler not initialized"}
        
        # Check if scaler was fitted on training data only
        training_samples = self.scaler.n_samples_seen_
        validation_samples = len(self.feature_columns) - validation_start_idx
        
        results = {
            "total_samples": len(self.feature_columns),
            "training_samples_used_for_scaling": training_samples,
            "validation_samples": validation_samples,
            "scaler_fitted_on_training_only": training_samples <= validation_start_idx,
            "data_leakage_detected": training_samples > validation_start_idx
        }
        
        if results["data_leakage_detected"]:
            results["warning"] = "POTENTIAL DATA LEAKAGE: Scaler was fitted on validation data"
        else:
            results["status"] = "OK: No data leakage detected in feature scaling"
        
        return results
    
    @classmethod
    def create_train_val_environments(
        cls,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.3,
        **kwargs
    ) -> Tuple['FuturesTradingEnv', 'FuturesTradingEnv']:
        """
        Create separate training and validation environments with proper data splitting
        to prevent data leakage in feature scaling.
        
        Args:
            df: Full dataset
            train_ratio: Ratio of data to use for training (default 0.7)
            val_ratio: Ratio of data to use for validation (default 0.3)
            **kwargs: Additional arguments passed to environment constructor
            
        Returns:
            Tuple of (train_env, val_env)
        """
        if train_ratio + val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio cannot exceed 1.0")
        
        total_samples = len(df)
        train_end_idx = int(total_samples * train_ratio)
        val_start_idx = train_end_idx
        val_end_idx = int(total_samples * (train_ratio + val_ratio))
        
        # Create training environment
        train_df = df.iloc[:train_end_idx].copy()
        train_env = cls(
            df=train_df,
            training_end_idx=len(train_df),  # Use all training data for scaler fitting
            **kwargs
        )
        
        # Create validation environment using the same scaler from training
        val_df = df.iloc[val_start_idx:val_end_idx].copy()
        val_env = cls(
            df=val_df,
            training_end_idx=train_end_idx,  # Use training data size for validation
            **kwargs
        )
        
        # Copy the fitted scaler from training to validation environment
        if hasattr(train_env, 'scaler') and train_env.scaler is not None:
            val_env.scaler = train_env.scaler
            
            # Re-transform validation features using the training scaler
            val_env.feature_columns_scaled = pd.DataFrame(
                val_env.scaler.transform(val_env.feature_columns),
                columns=val_env.feature_columns.columns,
                index=val_env.feature_columns.index
            )
        
        logging.info(f"Created train/val environments:")
        logging.info(f"  Training: {len(train_df)} samples (0 to {train_end_idx-1})")
        logging.info(f"  Validation: {len(val_df)} samples ({val_start_idx} to {val_end_idx-1})")
        logging.info(f"  Scaler fitted on training data only")
        
        return train_env, val_env
    
    @classmethod 
    def create_walk_forward_environments(
        cls,
        df: pd.DataFrame,
        train_window: int = 10000,  # Number of samples for training
        val_window: int = 2000,    # Number of samples for validation
        step_size: int = 1000,     # Step size for walk-forward
        **kwargs
    ) -> List[Tuple['FuturesTradingEnv', 'FuturesTradingEnv']]:
        """
        Create multiple train/validation environment pairs using walk-forward analysis
        to prevent data leakage across time periods.
        
        Args:
            df: Full dataset
            train_window: Number of samples to use for training
            val_window: Number of samples to use for validation  
            step_size: Number of samples to step forward each iteration
            **kwargs: Additional arguments passed to environment constructor
            
        Returns:
            List of (train_env, val_env) tuples
        """
        environments = []
        total_samples = len(df)
        
        start_idx = 0
        while start_idx + train_window + val_window <= total_samples:
            train_end_idx = start_idx + train_window
            val_start_idx = train_end_idx
            val_end_idx = val_start_idx + val_window
            
            # Create training environment
            train_df = df.iloc[start_idx:train_end_idx].copy()
            train_env = cls(
                df=train_df,
                training_end_idx=len(train_df),  # Use all training data for scaler
                **kwargs
            )
            
            # Create validation environment
            val_df = df.iloc[val_start_idx:val_end_idx].copy()
            val_env = cls(
                df=val_df,
                training_end_idx=train_window,  # Reference to training window size
                **kwargs
            )
            
            # Copy fitted scaler from training to validation
            if hasattr(train_env, 'scaler') and train_env.scaler is not None:
                val_env.scaler = train_env.scaler
                val_env.feature_columns_scaled = pd.DataFrame(
                    val_env.scaler.transform(val_env.feature_columns),
                    columns=val_env.feature_columns.columns,
                    index=val_env.feature_columns.index
                )
            
            environments.append((train_env, val_env))
            start_idx += step_size
        
        logging.info(f"Created {len(environments)} walk-forward environment pairs")
        logging.info(f"  Train window: {train_window} samples")
        logging.info(f"  Val window: {val_window} samples") 
        logging.info(f"  Step size: {step_size} samples")
        
        return environments
    
    def _setup_reward_config(self, reward_config: Optional[Dict[str, float]] = None) -> None:
        """
        Setup reward configuration with default values or user-provided overrides.
        
        Args:
            reward_config: Optional dictionary of reward parameters to override defaults
        """
        # Default reward configuration - all the "magic numbers" made configurable
        default_config = {
            # Base reward scaling
            'base_reward_scale': 100,
            'base_reward_cap_positive': 10.0,
            'base_reward_cap_negative': -10.0,
            
            # Drawdown penalties
            'severe_drawdown_threshold': 0.5,  # 50%
            'severe_drawdown_penalty': 20.0,
            'major_drawdown_threshold': 0.3,   # 30%
            'major_drawdown_penalty': 10.0,
            'moderate_drawdown_threshold': 0.1, # 10%
            'moderate_drawdown_penalty': 5.0,
            'linear_drawdown_multiplier': 25,
            
            # Balance ratio penalties
            'critical_equity_threshold': 0.05,  # 5%
            'critical_equity_penalty': 50.0,
            'severe_equity_threshold': 0.10,   # 10%
            'severe_equity_penalty': 30.0,
            'major_equity_threshold': 0.20,    # 20%
            'major_equity_penalty': 20.0,
            'moderate_equity_threshold': 0.30, # 30%
            'moderate_equity_penalty': 10.0,
            'minor_equity_threshold': 0.50,    # 50%
            'minor_equity_penalty': 5.0,
            
            # Consecutive loss penalties
            'consecutive_loss_exponent': 1.5,
            'consecutive_loss_cap': 15.0,
            
            # Trend penalties
            'trend_penalty_multiplier': 1000,
            'trend_penalty_cap': 8.0,
            
            # Volatility penalties
            'volatility_multiplier': 15,
            'volatility_penalty_cap': 5.0,
            'volatility_history_threshold': 10,
            
            # Trading cost penalties
            'cost_penalty_multiplier': 500,
            'cost_penalty_cap': 2.0,
            
            # Special penalties
            'liquidation_penalty': 25.0,
            'excessive_leverage_threshold': 20,
            'excessive_leverage_multiplier': 0.5,
            
            # Positive bonuses
            'position_hold_bonus': 0.5,
            'position_hold_penalty': 0.3,
            'optimal_hold_min': 4,
            'optimal_hold_max': 24,
            'excessive_hold_threshold': 24,
            'consecutive_wins_multiplier': 0.2,
            'consecutive_wins_cap': 2.0,
            'recovery_threshold': 0.05,        # 5%
            'recovery_multiplier': 20,
            'recovery_bonus_cap': 3.0,
            
            # Final reward caps
            'final_reward_positive_cap': 15.0,
            'final_reward_negative_cap': -25.0,
            'severe_loss_reward_cap': -50.0,
        }
        
        # Start with defaults
        self.reward_config = default_config.copy()
        
        # Override with user-provided values if any
        if reward_config:
            for key, value in reward_config.items():
                if key in self.reward_config:
                    self.reward_config[key] = float(value)
                else:
                    print(f"Warning: Unknown reward config parameter '{key}' ignored")
        
        # Log the configuration being used
        print(f"Reward configuration loaded with {len(self.reward_config)} parameters")
        if reward_config:
            print(f"User overrides applied: {list(reward_config.keys())}")
