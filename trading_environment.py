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
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None
    logging.warning("scikit-learn not available. Feature scaling will be skipped.")
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
        stop_loss_pct: float = 0.02,  # 2%
        take_profit_pct: float = 0.04,  # 4%
        log_file: str = None,
        training_iteration: int = 0,
        training_split_ratio: float = 0.7,  # Use 70% of data for scaler fitting
        training_end_idx: Optional[int] = None  # Explicit training end index for scaling
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
        self.training_iteration = training_iteration
        self.training_split_ratio = training_split_ratio
        self.training_end_idx = training_end_idx
        
        # Initialize logger
        if log_file:
            self.logger = TradeLogger(log_file)
        else:
            self.logger = None
        
        # Prepare technical indicators with proper scaling
        self._prepare_features(training_end_idx)
        
        # Trading state
        self.reset()
        
        # Define action space: continuous leverage from -max_leverage to +max_leverage
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
        Prepare technical indicators and features using TensorTrade-style processing
        
        Args:
            training_end_idx: Index to split training data for proper scaling.
                            If None, uses the first 70% of data for fitting scaler.
        """
        try:
            import pandas_ta as ta
            use_pandas_ta = True
        except ImportError:
            # Use fallback implementation
            from fallback_ta import FallbackTA as ta
            use_pandas_ta = False
        
        df = self.df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Technical indicators
        if use_pandas_ta:
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['ema_10'] = ta.ema(df['close'], length=10)
            df['ema_20'] = ta.ema(df['close'], length=20)
            
            # Momentum indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch['STOCHk_14_3_3'] if isinstance(stoch, pd.DataFrame) else stoch
            
            # Volatility indicators
            bb = ta.bbands(df['close'], length=20)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_histogram'] = macd['MACDh_12_26_9']
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=20)
        else:
            # Use fallback implementation
            df['sma_10'] = ta.sma(df['close'], 10)
            df['sma_20'] = ta.sma(df['close'], 20)
            df['ema_10'] = ta.ema(df['close'], 10)
            df['ema_20'] = ta.ema(df['close'], 20)
            
            # Momentum indicators
            df['rsi'] = ta.rsi(df['close'], 14)
            df['stoch_k'] = ta.stochastic_k(df['high'], df['low'], df['close'], 14)
            
            # Volatility indicators
            bb_upper, bb_middle, bb_lower = ta.bollinger_bands(df['close'], 20, 2)
            df['bb_upper'] = bb_upper
            df['bb_lower'] = bb_lower
            df['bb_middle'] = bb_middle
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], 14)
            
            # MACD
            macd_line, macd_signal, macd_histogram = ta.macd(df['close'], 12, 26, 9)
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_histogram
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], 20)
        
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position indicators
        df['price_position'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Select feature columns
        feature_cols = [
            'returns', 'log_returns', 'high_low_pct', 'close_open_pct',
            'sma_10', 'sma_20', 'ema_10', 'ema_20', 'rsi', 'stoch_k',
            'bb_width', 'atr', 'macd', 'macd_signal', 'macd_histogram',
            'volume_ratio', 'price_position'
        ]
        
        self.feature_columns = df[feature_cols].copy()
        self.price_data = df[['open', 'high', 'low', 'close', 'volume', 'timestamp']].copy()
        
        # FIX FOR DATA LEAKAGE: Proper feature scaling without lookahead bias
        if StandardScaler is None:
            logging.warning("StandardScaler not available. Using SimpleStandardScaler fallback.")
            self.scaler = SimpleStandardScaler()
        else:
            self.scaler = StandardScaler()
        
        # Determine training split for scaler fitting
        if training_end_idx is None:
            # Use first 70% of data for fitting scaler (avoid lookahead bias)
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
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        action = action[0]  # Extract scalar from array
        action = np.clip(action, -self.max_leverage, self.max_leverage)
        
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
            self._execute_action(action, current_price)
        
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
    
    def _execute_action(self, target_leverage: float, current_price: float):
        """Execute trading action based on target leverage"""
        target_position_value = target_leverage * self.equity
        target_position_size = target_position_value / current_price if current_price > 0 else 0
        
        # Calculate trade size needed
        trade_size = target_position_size - self.position_size
        
        if abs(trade_size) > 0.001:  # Only trade if significant change
            # Close existing position if changing direction
            if self.position_size != 0 and np.sign(target_position_size) != np.sign(self.position_size):
                self._close_position(current_price, "DIRECTION_CHANGE")
            
            # Open new position or adjust existing
            if abs(target_position_size) > 0.001:
                self._open_or_adjust_position(target_position_size, current_price)
    
    def _open_or_adjust_position(self, target_position_size: float, current_price: float):
        """Open new position or adjust existing position"""
        # Calculate trade details
        trade_value = abs(target_position_size * current_price)
        fee = trade_value * self.taker_fee
        
        # Check if we have enough equity
        required_margin = trade_value / self.max_leverage
        if required_margin + fee > self.equity:
            return  # Not enough equity
        
        # Close existing position if any
        if self.position_size != 0:
            self._close_position(current_price, "ADJUSTMENT")
        
        # Open new position
        self.position_size = target_position_size
        self.position_side = 1 if target_position_size > 0 else -1
        self.entry_price = current_price
        self.leverage = abs(target_position_size * current_price / self.equity)
        self.margin_used = trade_value / self.leverage if self.leverage > 0 else 0
        
        # Deduct fees
        self.balance -= fee
        self.total_fees += fee
        
        # Set stop loss and take profit
        if self.position_side == 1:  # Long
            self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
            self.take_profit_price = current_price * (1 + self.take_profit_pct)
        else:  # Short
            self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
            self.take_profit_price = current_price * (1 - self.take_profit_pct)
        
        # Track trade start
        self.trade_start_step = self.current_step
        self.entry_equity = self.equity
    
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
    
    def _check_liquidation(self, current_low: float, current_high: float) -> bool:
        """Check if position should be liquidated"""
        if self.position_size == 0 or self.leverage < 5:  # Only check for high leverage
            return False
        
        # Simplified liquidation check
        # Real Binance liquidation is more complex with maintenance margin
        liquidation_threshold = 0.9  # 90% of equity lost
        
        if self.equity <= self.initial_equity * (1 - liquidation_threshold):
            self.liquidated = True
            self._close_position((current_high + current_low) / 2, "LIQUIDATION")
            return True
        
        return False
    
    def _check_stop_loss_take_profit(self, current_low: float, current_high: float) -> bool:
        """Check if stop loss or take profit should be triggered"""
        if self.position_size == 0 or not self.stop_loss_price or not self.take_profit_price:
            return False
        
        if self.position_side == 1:  # Long position
            if current_low <= self.stop_loss_price:
                self._close_position(self.stop_loss_price, "STOP_LOSS")
                return True
            elif current_high >= self.take_profit_price:
                self._close_position(self.take_profit_price, "TAKE_PROFIT")
                return True
        else:  # Short position
            if current_high >= self.stop_loss_price:
                self._close_position(self.stop_loss_price, "STOP_LOSS")
                return True
            elif current_low <= self.take_profit_price:
                self._close_position(self.take_profit_price, "TAKE_PROFIT")
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
        
        # Scale and cap base reward
        base_reward = equity_change * 100
        base_reward = np.clip(base_reward, -10.0, 10.0)  # Cap base reward
        
        # === RISK-ADJUSTED COMPONENTS ===
        risk_penalty = 0.0
        balance_penalty = 0.0
        trend_penalty = 0.0
        
        # 1. Drawdown penalties (progressive)
        if len(self.equity_history) > 1:
            drawdown = (self.max_equity - self.equity) / self.max_equity
            
            if drawdown > 0.5:  # >50% drawdown
                risk_penalty += 20.0
            elif drawdown > 0.3:  # >30% drawdown
                risk_penalty += 10.0
            elif drawdown > 0.1:  # >10% drawdown
                risk_penalty += 5.0
            else:
                risk_penalty += drawdown * 25  # Linear penalty for smaller drawdowns
        
        # 2. Balance decline penalties (progressive)
        equity_ratio = self.equity / self.initial_equity
        
        if equity_ratio <= 0.05:  # ≤5% remaining - SEVERE
            balance_penalty = 50.0
        elif equity_ratio <= 0.10:  # ≤10% remaining - CRITICAL
            balance_penalty = 30.0
        elif equity_ratio <= 0.20:  # ≤20% remaining - MAJOR
            balance_penalty = 20.0
        elif equity_ratio <= 0.30:  # ≤30% remaining - MODERATE
            balance_penalty = 10.0
        elif equity_ratio <= 0.50:  # ≤50% remaining - MINOR
            balance_penalty = 5.0
        
        # 3. Consecutive loss penalties
        consecutive_loss_penalty = 0.0
        if self.consecutive_losses > 0:
            # Exponential penalty for consecutive losses
            consecutive_loss_penalty = min(15.0, self.consecutive_losses ** 1.5)
        
        # 4. Balance trend penalty (declining balance over time)
        if self.balance_trend_slope < 0:  # Declining balance
            trend_penalty = abs(self.balance_trend_slope) * 1000  # Scale the slope
            trend_penalty = min(trend_penalty, 8.0)  # Cap trend penalty
        
        # 5. Volatility penalty
        volatility_penalty = 0.0
        if len(self.returns_history) > 10:
            volatility = np.std(list(self.returns_history))
            volatility_penalty = min(volatility * 15, 5.0)  # Cap volatility penalty
        
        # 6. Trading cost penalty
        cost_penalty = 0.0
        if hasattr(self, '_last_fees') and self._last_fees > 0:
            cost_penalty = min(self._last_fees * 500, 2.0)  # Cap cost penalty
        
        # 7. Special penalties
        special_penalty = 0.0
        
        # Liquidation penalty
        if self.liquidated:
            special_penalty += 25.0
        
        # Excessive leverage penalty
        if self.leverage > 20:
            special_penalty += (self.leverage - 20) * 0.5
        
        # === POSITIVE REWARDS ===
        positive_bonus = 0.0
        
        # Position holding bonus (encourage longer-term thinking)
        if self.position_size != 0 and self.trade_start_step:
            hold_duration = self.current_step - self.trade_start_step
            if 4 <= hold_duration <= 24:  # 1-6 hours optimal
                positive_bonus += 0.5
            elif hold_duration > 24:  # Penalize too long holds
                positive_bonus -= 0.3
        
        # Consecutive wins bonus
        if self.consecutive_wins > 0:
            positive_bonus += min(self.consecutive_wins * 0.2, 2.0)  # Cap wins bonus
        
        # Recovery bonus (recovering from drawdown)
        if len(self.equity_history) > 5:
            recent_improvement = (self.equity - min(list(self.equity_history)[-5:])) / self.initial_equity
            if recent_improvement > 0.05:  # 5% improvement
                positive_bonus += min(recent_improvement * 20, 3.0)
        
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
        
        # === SEGMENT-BASED CAPPING ===
        # Cap final reward in different segments for stable learning
        if final_reward > 0:
            final_reward = min(final_reward, 15.0)  # Cap positive rewards
        else:
            final_reward = max(final_reward, -25.0)  # Cap negative rewards
        
        # Additional severe loss capping
        if equity_ratio <= 0.10:  # Very low equity
            final_reward = max(final_reward, -50.0)  # Allow larger negative rewards for severe losses
        
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
    

class SimpleStandardScaler:
    """
    Simple fallback implementation of StandardScaler when scikit-learn is not available
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = None
    
    def fit(self, X):
        """Fit scaler to training data"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        return (X_values - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit scaler and transform data"""
        return self.fit(X).transform(X)
