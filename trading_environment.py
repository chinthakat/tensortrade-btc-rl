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
warnings.filterwarnings("ignore")

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
        training_iteration: int = 0
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
        
        # Initialize logger
        if log_file:
            self.logger = TradeLogger(log_file)
        else:
            self.logger = None
        
        # Prepare technical indicators
        self._prepare_features()
        
        # Trading state
        self.reset()
        
        # Define action space: continuous leverage from -max_leverage to +max_leverage
        self.action_space = spaces.Box(
            low=-self.max_leverage, 
            high=self.max_leverage, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Define observation space
        n_features = self.feature_columns.shape[1]
        self.observation_space = spaces.Dict({
            'market_features': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.window_size, n_features), 
                dtype=np.float32
            ),
            'portfolio_features': spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(5,),  # equity, position, unrealized_pnl, margin_used, drawdown
                dtype=np.float32
            )
        })
    
    def _prepare_features(self):
        """Prepare technical indicators and features using TensorTrade-style processing"""
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
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.feature_columns_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.feature_columns),
            columns=self.feature_columns.columns,
            index=self.feature_columns.index
        )
    
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
        if len(self.equity_history) > 1:
            equity_return = (self.equity - self.equity_history[-2]) / self.equity_history[-2]
            self.returns_history.append(equity_return)
        
        self.max_equity = max(self.max_equity, self.equity)
        
        # Calculate reward
        reward = self._calculate_reward(prev_equity)
        
        # Check terminal conditions
        terminated = self.equity <= 0 or self.liquidated
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
    
    def _calculate_reward(self, prev_equity: float) -> float:
        """Calculate reward using composite risk-adjusted approach"""
        # Base profit/loss component
        if prev_equity > 0:
            equity_change = (self.equity - prev_equity) / prev_equity
        else:
            equity_change = 0.0
        
        reward = equity_change * 100  # Scale for better learning
        
        # Risk penalty based on drawdown
        if len(self.equity_history) > 1:
            drawdown = (self.max_equity - self.equity) / self.max_equity
            reward -= drawdown * 50  # Penalize drawdowns
        
        # Volatility penalty
        if len(self.returns_history) > 10:
            volatility = np.std(list(self.returns_history))
            reward -= volatility * 20
        
        # Cost penalty
        if hasattr(self, '_last_fees'):
            reward -= self._last_fees * 1000  # Penalize trading costs
        
        # Liquidation penalty
        if self.liquidated:
            reward -= 100
        
        # Position holding bonus (encourage longer-term thinking)
        if self.position_size != 0 and self.trade_start_step:
            hold_duration = self.current_step - self.trade_start_step
            if hold_duration > 4:  # More than 1 hour
                reward += 0.1
        
        return float(reward)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        # Market features (windowed historical data)
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        market_features = self.feature_columns_scaled.iloc[start_idx:end_idx].values
        
        # Pad if necessary
        if market_features.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - market_features.shape[0], market_features.shape[1]))
            market_features = np.vstack([padding, market_features])
        
        # Portfolio features
        drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0
        
        portfolio_features = np.array([
            self.equity / self.initial_equity,  # Normalized equity
            self.leverage / self.max_leverage,  # Normalized leverage
            self.unrealized_pnl / self.initial_equity if self.initial_equity > 0 else 0,  # Normalized PnL
            self.margin_used / self.initial_equity if self.initial_equity > 0 else 0,  # Normalized margin
            drawdown  # Current drawdown
        ], dtype=np.float32)
        
        return {
            'market_features': market_features.astype(np.float32),
            'portfolio_features': portfolio_features
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info"""
        return {
            'equity': self.equity,
            'position_size': self.position_size,
            'position_side': self.position_side,
            'leverage': self.leverage,
            'unrealized_pnl': self.unrealized_pnl,
            'episode_trades': self.episode_trades,
            'episode_profit': self.episode_profit,
            'liquidated': self.liquidated,
            'current_step': self.current_step,
            'max_steps': len(self.price_data) - 1
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            current_price = self.price_data.iloc[self.current_step]['close']
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Equity: {self.equity:.2f}")
            print(f"Position: {self.position_size:.4f}")
            print(f"Leverage: {self.leverage:.2f}x")
            print(f"Unrealized PnL: {self.unrealized_pnl:.2f}")
            print("-" * 40)
