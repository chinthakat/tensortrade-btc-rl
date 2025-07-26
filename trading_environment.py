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
    try:
        # Try installing compatible version if import fails
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta==0.3.14b0"])
        import pandas_ta as ta
    except:
        raise ImportError("pandas_ta is required but not installed. Please install with: pip install pandas_ta==0.3.14b0")
except Exception as e:
    # Handle numpy compatibility issues
    import warnings
    warnings.warn(f"pandas_ta import warning: {e}. Using fallback indicators.")
    # Use fallback technical analysis if pandas_ta fails
    ta = None

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create separate logger for penalty errors - logs to file only
penalty_logger = logging.getLogger('penalty_errors')
penalty_logger.setLevel(logging.ERROR)
penalty_logger.propagate = False  # Don't propagate to root logger (terminal)

# Create file handler for penalty errors
os.makedirs('logs', exist_ok=True)
penalty_file_handler = logging.FileHandler('logs/penalty_errors.log', mode='a')
penalty_file_handler.setLevel(logging.ERROR)
penalty_file_formatter = logging.Formatter('%(asctime)s - PENALTY - %(message)s')
penalty_file_handler.setFormatter(penalty_file_formatter)
penalty_logger.addHandler(penalty_file_handler)

# Create separate logger for episode maintenance - logs to file only
episode_logger = logging.getLogger('episode_maintenance')
episode_logger.setLevel(logging.INFO)
episode_logger.propagate = False  # Don't propagate to root logger (terminal)

# Create file handler for episode maintenance
episode_file_handler = logging.FileHandler('logs/episode_maintenance.log', mode='a')
episode_file_handler.setLevel(logging.INFO)
episode_file_formatter = logging.Formatter('%(asctime)s - EPISODE - %(message)s')
episode_file_handler.setFormatter(episode_file_formatter)
episode_logger.addHandler(episode_file_handler)

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


class PriceValidator:
    """Real-time price validation system for trading environment"""
    
    def __init__(self, tolerance_pct: float = 5.0, enable_logging: bool = True):
        """
        Initialize price validator
        
        Args:
            tolerance_pct: Maximum acceptable percentage difference
            enable_logging: Whether to log validation results
        """
        self.tolerance_pct = tolerance_pct
        self.enable_logging = enable_logging
        self.validation_log = []
        
    def validate_price(
        self, 
        trade_price: float, 
        market_price: float, 
        timestamp: Any,
        step: int,
        context: str = "UNKNOWN"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate trade price against market price
        
        Args:
            trade_price: Price used in trade execution
            market_price: Actual market price at the time
            timestamp: Timestamp of the trade
            step: Current trading step
            context: Context of the validation (ENTRY, CLOSE, etc.)
            
        Returns:
            Tuple of (is_valid, validation_details)
        """
        # Handle zero or invalid prices
        if trade_price <= 0:
            validation_result = {
                'valid': False,
                'reason': 'ZERO_TRADE_PRICE',
                'trade_price': trade_price,
                'market_price': market_price,
                'difference_pct': 100.0,
                'timestamp': timestamp,
                'step': step,
                'context': context
            }
            
            if self.enable_logging:
                logging.warning(f"PRICE_VALIDATION_FAILED: {context} - Zero trade price at step {step}")
                self.validation_log.append(validation_result)
            
            return False, validation_result
        
        if market_price <= 0:
            validation_result = {
                'valid': False,
                'reason': 'ZERO_MARKET_PRICE',
                'trade_price': trade_price,
                'market_price': market_price,
                'difference_pct': 0.0,
                'timestamp': timestamp,
                'step': step,
                'context': context
            }
            
            if self.enable_logging:
                logging.warning(f"PRICE_VALIDATION_FAILED: {context} - Zero market price at step {step}")
                self.validation_log.append(validation_result)
            
            return False, validation_result
        
        # Calculate percentage difference
        price_diff_pct = abs(trade_price - market_price) / market_price * 100
        
        is_valid = price_diff_pct <= self.tolerance_pct
        
        validation_result = {
            'valid': is_valid,
            'reason': 'WITHIN_TOLERANCE' if is_valid else 'EXCEEDS_TOLERANCE',
            'trade_price': trade_price,
            'market_price': market_price,
            'difference_pct': price_diff_pct,
            'timestamp': timestamp,
            'step': step,
            'context': context
        }
        
        if self.enable_logging:
            if not is_valid:
                logging.warning(
                    f"PRICE_VALIDATION_FAILED: {context} - Step {step}, "
                    f"Trade: ${trade_price:.2f}, Market: ${market_price:.2f}, "
                    f"Diff: {price_diff_pct:.2f}% (>{self.tolerance_pct}%)"
                )
            self.validation_log.append(validation_result)
        
        return is_valid, validation_result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed"""
        if not self.validation_log:
            return {'total_validations': 0}
        
        total = len(self.validation_log)
        failed = sum(1 for v in self.validation_log if not v['valid'])
        
        reasons = {}
        for v in self.validation_log:
            reason = v['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        
        return {
            'total_validations': total,
            'failed_validations': failed,
            'success_rate': (total - failed) / total * 100 if total > 0 else 0,
            'failure_reasons': reasons
        }


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
        window_size: int = 60,  # IMPROVED: Increased from 20 to 60 to align with indicator buffer requirements
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
        # Risk Management Configuration
        max_risk_per_trade: float = 0.02,  # CONFIGURABLE: Maximum 2% risk per trade (was hardcoded)
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
        
        # Risk management configuration
        self.max_risk_per_trade = max_risk_per_trade  # Store configurable risk parameter
        
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
        
        # Initialize price validator
        self.price_validator = PriceValidator(
            tolerance_pct=5.0,  # 5% tolerance for price validation
            enable_logging=True
        )
        
        # Trading state
        self.reset()
        
        # Define action space based on configuration
        if self.use_advanced_action_space:
            # Phase 1: Enhanced action space with explicit HOLD/CANCEL actions
            self.action_space = spaces.Dict({
                'action_type': spaces.Discrete(4),  # 0=HOLD, 1=BUY, 2=SELL, 3=CANCEL
                'leverage': spaces.Box(
                    low=0.1,  # Minimum leverage when trading
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
            # Legacy simple action space: continuous leverage only with threshold
            self.action_space = spaces.Box(
                low=-self.max_leverage, 
                high=self.max_leverage, 
                shape=(1,), 
                dtype=np.float32
            )
            
        # Set trading threshold for legacy mode
        self.trading_threshold = 0.1  # Minimum leverage to trigger trade
        
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
                shape=(9,),  # Enhanced to 9 features (added PnL trend)
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
        
        # Action tracking for enhanced rewards
        self.last_action_type = "HOLD"  # Track last action for reward calculation
        self.hold_streak = 0  # Count consecutive holds
        self.trade_streak = 0  # Count consecutive trades
        
        # Action type statistics
        self.action_type_counts = {"HOLD": 0, "BUY": 0, "SELL": 0, "CANCEL": 0}
        self.action_log_interval = 1000  # Log action stats every N steps
    
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
        
        # Additional Directional Indicators
        
        # 1. ADX (Average Directional Index) - Trend Strength
        adx_result = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_result is not None and isinstance(adx_result, pd.DataFrame):
            df['adx'] = adx_result['ADX_14']  # Trend strength (>25 = strong trend)
            df['di_plus'] = adx_result['DMP_14']  # Positive directional indicator
            df['di_minus'] = adx_result['DMN_14']  # Negative directional indicator
            # Directional bias: +1 for bullish, -1 for bearish, 0 for neutral
            df['directional_bias'] = np.where(df['di_plus'] > df['di_minus'], 1, 
                                            np.where(df['di_minus'] > df['di_plus'], -1, 0))
        
        # 2. Parabolic SAR - Trend Direction and Stop Loss
        psar_result = ta.psar(df['high'], df['low'], df['close'])
        if psar_result is not None and isinstance(psar_result, pd.DataFrame):
            # PSAR columns vary, check available columns
            psar_cols = [col for col in psar_result.columns if 'PSAR' in col]
            if psar_cols:
                df['psar'] = psar_result[psar_cols[0]]
                # Trend direction: 1 if price above PSAR (bullish), -1 if below (bearish)
                df['psar_trend'] = np.where(df['close'] > df['psar'], 1, -1)
        
        # 3. Williams %R - Momentum Oscillator
        willr_result = ta.willr(df['high'], df['low'], df['close'], length=14)
        if willr_result is not None:
            df['williams_r'] = willr_result
            # Directional signals: > -20 overbought (bearish), < -80 oversold (bullish)
            df['williams_signal'] = np.where(df['williams_r'] > -20, -1,
                                           np.where(df['williams_r'] < -80, 1, 0))
        
        # 4. CCI (Commodity Channel Index) - Momentum
        cci_result = ta.cci(df['high'], df['low'], df['close'], length=20)
        if cci_result is not None:
            df['cci'] = cci_result
            # CCI signals: > 100 bullish, < -100 bearish
            df['cci_signal'] = np.where(df['cci'] > 100, 1,
                                      np.where(df['cci'] < -100, -1, 0))
        
        # 5. Moving Average Cross Signals
        df['ma_cross_signal'] = np.where(df['ema_10'] > df['ema_20'], 1,
                                       np.where(df['ema_10'] < df['ema_20'], -1, 0))
        
        # 6. MACD Signal Line Cross
        df['macd_cross_signal'] = np.where(df['macd'] > df['macd_signal'], 1,
                                         np.where(df['macd'] < df['macd_signal'], -1, 0))
        
        # 7. Multi-timeframe trend strength
        # Short-term trend (5-period)
        df['trend_5'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
        # Medium-term trend (10-period) 
        df['trend_10'] = np.where(df['close'] > df['close'].shift(10), 1, -1)
        # Long-term trend (20-period)
        df['trend_20'] = np.where(df['close'] > df['close'].shift(20), 1, -1)
        
        # 8. Composite Directional Score (-3 to +3)
        df['composite_direction'] = (
            df.get('directional_bias', 0) +
            df.get('psar_trend', 0) + 
            df.get('ma_cross_signal', 0) +
            df.get('macd_cross_signal', 0) +
            np.where(df['rsi'] > 70, -1, np.where(df['rsi'] < 30, 1, 0))
        ).fillna(0)
        
        # Volume indicators
        volume_sma = ta.sma(df['volume'], length=20)
        if volume_sma is None:
            raise ValueError("Failed to calculate volume SMA. Check if data is valid.")
        df['volume_sma'] = volume_sma
        
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price position indicators
        df['price_position'] = (df['close'] - df['sma_20']) / df['sma_20']
        
        # Instead of dropping NaN rows, we'll fill them to preserve alignment with original CSV
        # Store original data shape for reference
        original_shape = df.shape
        
        # Fill NaN values with appropriate strategies to preserve data alignment
        # For price-based features, use forward fill then backward fill
        price_features = ['sma_10', 'sma_20', 'ema_10', 'ema_20', 'bb_upper', 'bb_lower', 'bb_width']
        for col in price_features:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # For percentage-based features, fill with 0
        pct_features = ['returns', 'log_returns', 'high_low_pct', 'close_open_pct', 'price_position']
        for col in pct_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # For technical indicators, use more sophisticated filling
        if 'rsi' in df.columns:
            df['rsi'] = df['rsi'].fillna(50)  # Neutral RSI
        
        if 'stoch_k' in df.columns:
            df['stoch_k'] = df['stoch_k'].fillna(50)  # Neutral Stochastic
            
        if 'atr' in df.columns:
            # Fill ATR with a percentage of current price
            df['atr'] = df['atr'].fillna(df['close'] * 0.02)  # 2% of price as default ATR
            
        # For MACD, fill with 0 (neutral)
        macd_cols = ['macd', 'macd_signal', 'macd_histogram']
        for col in macd_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # For directional indicators, fill with neutral values
        directional_cols = {
            'adx': 0,  # No trend
            'di_plus': 0, 'di_minus': 0,
            'directional_bias': 0,  # Neutral
            'psar': df['close'],  # Use current price if missing
            'psar_trend': 0,  # Neutral
            'williams_r': -50,  # Neutral Williams %R
            'williams_signal': 0,  # No signal
            'cci': 0,  # Neutral CCI
            'cci_signal': 0,  # No signal
            'ma_cross_signal': 0,  # No cross
            'macd_cross_signal': 0,  # No cross
            'trend_5': 0, 'trend_10': 0, 'trend_20': 0,  # Neutral trends
            'composite_direction': 0  # Neutral composite
        }
        
        for col, fill_value in directional_cols.items():
            if col in df.columns:
                if col == 'psar':
                    df[col] = df[col].fillna(df['close'])
                else:
                    df[col] = df[col].fillna(fill_value)
        
        # For volume ratio, fill with 1 (average volume)
        if 'volume_ratio' in df.columns:
            df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # Final check - if any NaN values remain, fill with 0
        df = df.fillna(0)
        
        logging.info(f"Data shape after NaN handling: {df.shape} (preserved {original_shape[0]} rows)")
        
        # Verify no data loss occurred
        if df.shape[0] != original_shape[0]:
            logging.warning(f"Unexpected data loss: {original_shape[0] - df.shape[0]} rows lost during processing")
        else:
            logging.info("SUCCESS: Data alignment preserved - no rows lost during preprocessing")
        
        # Select feature columns (SIMPLIFIED 8-CORE MINIMAL SET)
        # This reduces features from 27 to 8 (70% reduction) to solve overtrading
        feature_cols = [
            # Core price momentum (most important)
            'returns',          # Price percentage change
            
            # Essential momentum indicators  
            'rsi',             # 14-period RSI (overbought/oversold)
            
            # Essential trend indicators
            'ema_10',          # Short-term trend (10-period EMA)
            'ema_20',          # Medium-term trend (20-period EMA)
            
            # Essential momentum oscillator
            'macd',            # MACD line (momentum)
            
            # Essential trend strength
            'adx',             # Average Directional Index (trend strength)
            
            # Essential volatility
            'atr',             # Average True Range (volatility measure)
            
            # Essential volume context
            'volume_ratio'     # Volume relative to average
        ]
        
        logging.info("TARGET: USING SIMPLIFIED 8-CORE INDICATOR SET")
        logging.info("REDUCED: Reduced from 27 to 8 indicators (70% reduction)")
        logging.info("BOOST: This should eliminate overtrading behavior")
        
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
        
        # Calculate safe starting point - ensure technical indicators are stable
        # IMPROVED: Use window_size consistently to avoid padded data at episode start
        # Technical indicators need time to stabilize, so we ensure minimum buffer equals window_size
        min_buffer_for_indicators = max(self.window_size, 60)  # Use the larger of window_size or 60 for stability
        
        # CONSISTENCY FIX: If window_size < 60, we should increase it rather than start with padded data
        if self.window_size < min_buffer_for_indicators:
            episode_logger.warning(f"Window size ({self.window_size}) is smaller than minimum buffer ({min_buffer_for_indicators}). "
                                 f"This may cause padded observations at episode start.")
        
        # Trading state
        self.current_step = min_buffer_for_indicators
        self.equity = self.initial_equity
        self.balance = self.initial_equity
        self.position_size = 0.0  # In base currency (BTC)
        self.position_side = 0  # 1 for long, -1 for short, 0 for flat
        self.entry_price = 0.0
        self.trade_entry_price = 0.0  # Store original entry price for trade logging consistency
        self.entry_equity = 0.0  # Store equity at trade entry for consistent net worth tracking
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.leverage = 0.0
        
        # Prevent duplicate trade logging
        self._efficient_trade_logged = False
        
        # Risk management
        self.stop_loss_price = None
        self.take_profit_price = None
        self.liquidation_price = None  # Real-time liquidation price
        
        # Trade tracking
        self.trade_id = 0
        self.episode_id = getattr(self, 'episode_id', 0) + 1  # Increment episode ID on each reset
        self.trade_start_step = None
        self.trade_entry_datetime = None  # Store original entry datetime for consistent logging
        self.entry_equity = 0.0
        self.total_fees = 0.0
        self.total_funding_costs = 0.0
        
        # Reward tracking for trades
        self.current_trade_reward = 0.0
        
        # PnL-aware reward system
        self.unrealized_pnl_history = []  # Track PnL trend for intelligent rewards
        
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
        
        # Drawdown tracking flags
        self.severe_drawdown_triggered = False
        self.moderate_drawdown_triggered = False
        
        # Action tracking for enhanced rewards
        self.last_action_type = "HOLD"
        self.hold_streak = 0
        self.trade_streak = 0
        
        # Action type statistics
        self.action_type_counts = {"HOLD": 0, "BUY": 0, "SELL": 0, "CANCEL": 0}
        
        # Reset duplicate logging prevention flag
        self._efficient_trade_logged = False
        
        # Episode statistics
        self.episode_trades = 0
        self.episode_profit = 0.0
        self.episode_total_fees = 0.0  # Track total fees per episode
        
        # Initialize penalty tracking variables
        self.dust_position_penalty = 0.0
        self.phantom_trade_penalty = 0.0
        self.invalid_entry_penalty = 0.0
        self.zero_pnl_detection_penalty = 0.0
        self.zero_pnl_close_penalty = 0.0
        self.excessive_modification_penalty = 0.0
        self.fee_cap_penalty = 0.0
        self.entry_price_fallback_penalty = 0.0
        self.critical_trade_block_penalty = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_unique_trade_id(self):
        """Generate a unique trade ID that includes episode information"""
        return f"EP{self.episode_id:03d}_TRADE_{self.trade_id:05d}"
    
    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Parse action based on action space type
        action_type = "TRADE"  # Default for legacy mode
        leverage = 0.0
        risk_percentage = 1.0
        
        if self.use_advanced_action_space:
            # Enhanced action space: Dict with action_type, leverage and risk_percentage
            if isinstance(action, dict):
                action_type_raw = action['action_type']
                if isinstance(action_type_raw, (int, np.integer)):
                    action_type_idx = int(action_type_raw)
                elif isinstance(action_type_raw, np.ndarray):
                    action_type_idx = int(action_type_raw[0])
                else:
                    action_type_idx = int(action_type_raw)
                
                leverage = action['leverage'][0] if isinstance(action['leverage'], np.ndarray) else action['leverage']
                risk_percentage = action['risk_percentage'][0] if isinstance(action['risk_percentage'], np.ndarray) else action['risk_percentage']
            else:
                # If action is from wrapper, it's a numpy array [action_type, leverage, risk_percentage]
                action_type_idx = int(action[0])
                leverage = action[1]
                risk_percentage = action[2]
            
            # Map action type
            action_types = ["HOLD", "BUY", "SELL", "CANCEL"]
            action_type = action_types[action_type_idx]
            
            # Clip values to valid ranges
            leverage = float(np.clip(leverage, 0.1, self.max_leverage))
            risk_percentage = float(np.clip(risk_percentage, 0.01, 1.0))
            
            # Convert BUY/SELL to leverage values
            if action_type == "BUY":
                leverage = abs(leverage)  # Positive leverage for long
            elif action_type == "SELL":
                leverage = -abs(leverage)  # Negative leverage for short
            elif action_type == "HOLD":
                leverage = 0.0  # No new position
            elif action_type == "CANCEL":
                leverage = 0.0  # Close position
                risk_percentage = 1.0  # Use full risk for closing
        else:
            # Legacy action space: single leverage value with threshold
            if isinstance(action, dict):
                # This shouldn't happen in legacy mode, but handle gracefully
                leverage = action.get('leverage', 0.0)
                if isinstance(leverage, (list, np.ndarray)):
                    leverage = leverage[0]
            else:
                leverage = action[0] if isinstance(action, (list, np.ndarray)) else action
            
            # PENALTY FOR EXTREME LEVERAGE REQUESTS - penalize before clipping
            original_leverage = float(leverage)
            extreme_leverage_penalty = 0.0
            
            if abs(original_leverage) > self.max_leverage:
                leverage_excess = abs(original_leverage) / self.max_leverage - 1.0
                extreme_leverage_penalty = min(leverage_excess * 0.2, 1.0)  # Up to -1.0 penalty
                # Silent penalty - only log at debug level since penalty is applied
                logging.debug(f"EXTREME_LEVERAGE_PENALTY: Requested {original_leverage:.1f}x > {self.max_leverage}x limit")
                logging.debug(f"LEVERAGE_EXCESS_PENALTY: -{extreme_leverage_penalty:.4f} for {leverage_excess*100:.1f}% excess")
            
            # Store penalty for reward calculation
            if not hasattr(self, 'extreme_leverage_penalty'):
                self.extreme_leverage_penalty = 0.0
            self.extreme_leverage_penalty = extreme_leverage_penalty
            
            leverage = float(np.clip(leverage, -self.max_leverage, self.max_leverage))
            
            # Apply trading threshold - treat small leverage as HOLD
            if abs(leverage) < self.trading_threshold:
                action_type = "HOLD"
                leverage = 0.0
            else:
                action_type = "BUY" if leverage > 0 else "SELL"
            
            risk_percentage = 1.0  # Use full equity (legacy behavior)
        
        # Store action type for reward calculation
        self.last_action_type = action_type
        
        # Count action types for statistics
        if action_type in self.action_type_counts:
            self.action_type_counts[action_type] += 1
        
        # Update action streaks
        if action_type == "HOLD":
            self.hold_streak += 1
            self.trade_streak = 0
        else:
            self.trade_streak += 1
            self.hold_streak = 0
        
        # Store previous state
        prev_equity = self.equity
        
        # EPISODE TERMINATION FIX: Check episode boundary BEFORE any trading actions
        next_step = self.current_step + 1
        
        # If we're at or near the end of data, force episode termination
        if next_step >= len(self.price_data) - 1:
            # FORCE EPISODE TERMINATION to prevent infinite loop at boundary
            current_price = self._safe_get_price_data(self.current_step, 'close')
            
            # Close any open position WITHOUT charging fees (episode cleanup)
            if self.position_size != 0:
                self._force_close_position_no_fees(current_price, "EPISODE_END")
            
            # Return terminal state immediately - NO MORE TRADING ALLOWED
            terminated = True
            truncated = True
            
            # Create terminal observation
            observation = {
                'market_features': np.zeros((self.window_size, len(self.feature_columns.columns)), dtype=np.float32),
                'portfolio_features': np.zeros(9, dtype=np.float32)
            }
            
            # Final reward calculation
            reward = self._calculate_enhanced_reward(prev_equity)
            info = self._get_info()
            
            logging.info(f"EPISODE_TERMINATED at step {self.current_step}: No more trading allowed at episode boundary")
            return observation, reward, terminated, truncated, info
        
        # Safe to continue - move to next step for trading
        self.current_step = next_step
        current_price = self._safe_get_price_data(self.current_step, 'close')
        current_high = self._safe_get_price_data(self.current_step, 'high', current_price)
        current_low = self._safe_get_price_data(self.current_step, 'low', current_price)
        
        # Validate that current_price is not zero (CRITICAL FIX)
        if current_price <= 0:
            logging.error(f"ZERO_PRICE_DETECTED in step(): current_price={current_price} at step {self.current_step}")
            # Get a valid price from the last available data
            current_price = self._safe_get_price_data(self.current_step, 'close')
            episode_logger.info(f"PRICE_CORRECTED in step(): Using price={current_price} instead")
            # Also fix current_high and current_low if they're invalid
            if current_high <= 0:
                current_high = current_price
            if current_low <= 0:
                current_low = current_price
        
        # Update unrealized PnL - FIXED: Use abs(position_size) for correct short position calculations
        if self.position_size != 0:
            if self.position_side == 1:  # Long
                self.unrealized_pnl = abs(self.position_size) * (current_price - self.entry_price)
            else:  # Short
                self.unrealized_pnl = abs(self.position_size) * (self.entry_price - current_price)
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
            if action_type == "HOLD":
                # Do nothing - just hold current position
                pass
            elif action_type == "CANCEL":
                # CANCEL should close any open position and log the closure
                if self.position_size != 0:
                    # Get current price for logging
                    current_price = self._safe_get_price_data(self.current_step, 'close')
                    
                    # Calculate PnL for the cancelled position - FIXED: Use abs(position_size) for correct short calculations
                    if self.position_side == 1:  # Long
                        pnl = abs(self.position_size) * (current_price - self.entry_price)
                    else:  # Short
                        pnl = abs(self.position_size) * (self.entry_price - current_price)
                    
                    # Update balance (minimal fees for CANCEL)
                    cancel_fee = abs(self.position_size * current_price) * (self.taker_fee * 0.5)  # Reduced fee
                    self.balance += pnl - cancel_fee
                    self.total_fees += cancel_fee
                    self.total_realized_pnl += pnl
                    
                    # Log the CANCEL closure
                    if self.logger:
                        cancel_duration_steps = self.current_step - (self.trade_start_step or self.current_step)
                        cancel_duration_hours = cancel_duration_steps * 0.25  # 15min intervals
                        
                        trade_data = {
                            'trade_id': self._get_unique_trade_id(),  # Same ID as open trade
                            'training_step': self.current_step,
                            'training_iteration': getattr(self, 'training_iteration', 0),
                            'entry_datetime': self.trade_entry_datetime or self._safe_get_df_data(self.trade_start_step or self.current_step, 'timestamp', f"step_{self.trade_start_step or self.current_step}"),
                            'close_datetime': self._safe_get_df_data(self.current_step, 'timestamp', f"step_{self.current_step}"),
                            'side': 'FLAT',
                            'entry_action': 'CANCEL_CLOSE',
                            'entry_price': round(self.trade_entry_price or self.entry_price, 4),  # Use stored trade entry price
                            'close_price': round(current_price, 4),
                            'net_pnl': round(pnl, 6),
                            'close_reward': 0,
                            'entry_net_worth': getattr(self, 'entry_equity', self.equity),  # Use stored entry equity
                            'close_net_worth': self.equity,
                            'trade_duration_hours': cancel_duration_hours,
                            'status': 'CLOSED',
                            'win_loss': 'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'NEUTRAL',
                            'position_size': 0.0,  # Position is now closed
                            'fees_paid': cancel_fee,
                            'stop_loss_price': '',
                            'take_profit_price': '',
                            'close_reason': 'CANCEL_ACTION'
                        }
                        self.logger.log_trade(trade_data)
                        
                        episode_logger.info(f"CANCEL_CLOSE: Closed trade {self.trade_id:05d} via CANCEL action")
                
                # Reset position state
                self.position_size = 0.0
                self.position_side = 0
                self.entry_price = 0.0
                self.trade_entry_price = 0.0  # Reset trade entry price
                self.margin_used = 0.0
                self.unrealized_pnl = 0.0
                self.stop_loss_price = None
                self.take_profit_price = None
                self.liquidation_price = None
                self.trade_start_step = None
                self.trade_entry_datetime = None  # Reset entry datetime when position closes
            else:
                # Execute BUY/SELL action
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
        
        # Store reward for current trade (if position is open)
        if self.position_size != 0:
            if not hasattr(self, 'current_trade_reward'):
                self.current_trade_reward = 0.0
            self.current_trade_reward += reward
        
        # Note: current_step may have been incremented to get correct price if data available
        
        # Check terminal conditions
        terminated = self.equity <= 0 or self.liquidated or terminated_early
        truncated = self.current_step >= len(self.price_data)
        
        # Get next observation only if not terminated/truncated
        if not (terminated or truncated):
            observation = self._get_observation()
        else:
            # Return a dummy observation for terminal states
            observation = {
                'market_features': np.zeros((self.window_size, len(self.feature_columns.columns)), dtype=np.float32),
                'portfolio_features': np.zeros(9, dtype=np.float32)  # Updated to 9 features
            }
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _safe_get_price_data(self, step: int, column: str, default_value: float = None):
        """Safely get price data at a specific step"""
        if step < 0 or step >= len(self.price_data):
            # Instead of returning 0.0, use the last available price
            if default_value is not None:
                return default_value
            # Use last available price if out of bounds
            if len(self.price_data) > 0:
                last_step = len(self.price_data) - 1
                return self.price_data.iloc[last_step][column]
            return 0.0  # Only if no data at all
        return self.price_data.iloc[step][column]
    
    def _safe_get_feature_data(self, step: int, column: str, default_value: float = 0.0):
        """Safely get feature data at a specific step"""
        if step < 0 or step >= len(self.feature_columns):
            return default_value
        return self.feature_columns.iloc[step][column]
    
    def _safe_get_df_data(self, step: int, column: str, default_value=None):
        """Safely get dataframe data at a specific step"""
        if step < 0 or step >= len(self.df):
            return default_value or f"step_{step}"
        return self.df.iloc[step][column]

    def _execute_action(self, target_leverage: float, risk_percentage: float, current_price: float):
        """Execute trading action based on target leverage and risk percentage"""
        
        # Validate that current_price is not zero
        if current_price <= 0:
            logging.error(f"ZERO_PRICE_DETECTED: current_price={current_price} at step {self.current_step}")
            # Get a valid price from the last available data
            current_price = self._safe_get_price_data(self.current_step, 'close')
            episode_logger.info(f"PRICE_CORRECTED: Using price={current_price} instead")
        
        # Validate current price against market data
        market_price = self._safe_get_price_data(self.current_step, 'close')
        timestamp = self._safe_get_price_data(self.current_step, 'timestamp', 0)
        
        is_valid, validation_details = self.price_validator.validate_price(
            trade_price=current_price,
            market_price=market_price,
            timestamp=timestamp,
            step=self.current_step,
            context="TRADE_EXECUTION"
        )
        
        if not is_valid:
            logging.error(f"PRICE_VALIDATION_FAILED during trade execution: {validation_details}")
        
        # IMPROVED: Scale risk percentage to meaningful range instead of hard clamping
        # This helps the agent learn more efficiently by mapping its output to actual useful values
        # Use configurable max_risk_per_trade instead of hardcoded value
        
        # Scale risk from agent's [0,1] output to [0, max_risk_per_trade] range
        # This way, agent learns to use the full range meaningfully
        if risk_percentage > 1.0:
            risk_percentage = 1.0  # Cap at 100% if agent outputs higher
        risk_percentage = risk_percentage * self.max_risk_per_trade  # Scale to 0-max_risk% range
        
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
        
        # 🚨 CRITICAL FIX: Limit position value to prevent excessive fees
        # Maximum position should never exceed available equity * max_leverage
        max_safe_position_value = self.equity * min(abs(target_leverage), self.max_leverage)
        
        # Additional safety: Never allow position value > 50% of available equity at max leverage
        absolute_max_position = self.equity * self.max_leverage * 0.5
        max_safe_position_value = min(max_safe_position_value, absolute_max_position)
        
        # Apply the safety limit
        safety_intervention_penalty = 0.0
        if abs(target_position_value) > max_safe_position_value:
            intervention_severity = abs(target_position_value) / max_safe_position_value - 1.0
            safety_intervention_penalty += min(intervention_severity * 0.1, 0.5)  # Up to -0.5 penalty
            logging.debug(f"POSITION_SIZE_LIMITED: Requested ${abs(target_position_value):.2f}, limited to ${max_safe_position_value:.2f}")
            logging.debug(f"SAFETY_PENALTY: -{safety_intervention_penalty:.3f} for excessive position request")
            target_position_value = np.sign(target_position_value) * max_safe_position_value
        
        target_position_size = target_position_value / current_price if current_price > 0 else 0
        
        # 🚨 EMERGENCY BRAKE: Absolute position size limit (in BTC)
        # Never allow more than equity/price worth of BTC to be traded
        max_btc_position = (self.equity * self.max_leverage * 0.2) / current_price  # 20% of max theoretical
        
        # 🛑 ABSOLUTE EMERGENCY LIMIT: Never exceed $50,000 position value
        absolute_max_position_value = 50000.0  # $50K absolute maximum
        absolute_max_btc = absolute_max_position_value / current_price
        max_btc_position = min(max_btc_position, absolute_max_btc)
        
        if abs(target_position_size) > max_btc_position:
            btc_excess = abs(target_position_size) / max_btc_position - 1.0
            safety_intervention_penalty += min(btc_excess * 0.2, 1.0)  # Up to -1.0 penalty for extreme BTC requests
            penalty_logger.error(f"EMERGENCY_POSITION_BRAKE: Position {target_position_size:.6f} BTC > limit {max_btc_position:.6f} BTC at step {getattr(self, 'current_step', 'unknown')}")
            penalty_logger.error(f"POSITION_VALUE: ${abs(target_position_size * current_price):.2f} > max ${max_btc_position * current_price:.2f}")
            penalty_logger.error(f"SEVERE_SAFETY_PENALTY: -{min(btc_excess * 0.2, 1.0):.3f} for extreme BTC position")
            logging.debug(f"EMERGENCY_POSITION_BRAKE: Position {target_position_size:.6f} BTC > limit {max_btc_position:.6f} BTC")
            target_position_size = np.sign(target_position_size) * max_btc_position
        
        # Calculate trade size needed
        trade_size = target_position_size - self.position_size
        
        # 🚨 FINAL SAFETY CHECK: Prevent trades that would cause excessive fees
        trade_value = abs(trade_size * current_price)
        estimated_fee = trade_value * self.taker_fee
        
        # If fee would be > 1% of equity, reduce trade size
        max_fee_allowed = self.equity * 0.01  # 1% of equity
        if estimated_fee > max_fee_allowed:
            reduction_factor = max_fee_allowed / estimated_fee
            fee_excess = estimated_fee / max_fee_allowed - 1.0
            safety_intervention_penalty += min(fee_excess * 0.05, 0.3)  # Up to -0.3 penalty for excessive fees
            trade_size *= reduction_factor
            target_position_size = self.position_size + trade_size
            logging.debug(f"TRADE_SIZE_REDUCED: Fee would be ${estimated_fee:.2f}, reduced by {(1-reduction_factor)*100:.1f}%")
            logging.debug(f"FEE_SAFETY_PENALTY: -{min(fee_excess * 0.05, 0.3):.3f} for excessive fee attempt")
            
        # Store the total safety penalty for the reward calculation
        if not hasattr(self, 'safety_intervention_penalty'):
            self.safety_intervention_penalty = 0.0
        self.safety_intervention_penalty = safety_intervention_penalty
        
        # 🧹 DUST POSITION FILTER: Prevent tiny positions that cause chaos
        min_position_size = 0.001  # Minimum 0.001 BTC (≈$30-50) to be considered a real position
        
        # If final position would be dust, round to zero instead
        if abs(target_position_size) < min_position_size:
            if abs(target_position_size) > 0:
                logging.debug(f"DUST_POSITION_FILTERED: Position {target_position_size:.6f} BTC < {min_position_size} BTC minimum, setting to 0")
                # Apply small penalty for creating dust positions
                if not hasattr(self, 'dust_position_penalty'):
                    self.dust_position_penalty = 0.0
                self.dust_position_penalty += 0.01  # Small penalty to discourage dust creation
            target_position_size = 0.0
            trade_size = -self.position_size  # Close existing position completely
        
        # If trade would create a dust position, filter it out
        elif abs(trade_size) > 0 and abs(trade_size * current_price) < 20.0:  # Less than $20 trade
            logging.debug(f"DUST_TRADE_FILTERED: Trade ${abs(trade_size * current_price):.2f} < $20 minimum, skipping")
            if not hasattr(self, 'dust_position_penalty'):
                self.dust_position_penalty = 0.0
            self.dust_position_penalty += 0.005  # Very small penalty for dust attempts
            trade_size = 0.0  # No trade
            target_position_size = self.position_size  # Keep current position
            
        # Recalculate final trade value and fee after all safety checks
        final_trade_value = abs(trade_size * current_price)
        final_estimated_fee = final_trade_value * self.taker_fee
        
        # 🛑 FINAL EMERGENCY BRAKE: Absolutely prevent trades > $50K
        if final_trade_value > 50000.0:
            penalty_logger.error(f"CRITICAL_TRADE_BLOCK: Trade value ${final_trade_value:.2f} > $50K limit, BLOCKING TRADE")
            penalty_logger.error(f"  trade_size={trade_size:.6f} BTC, current_price=${current_price:.2f}")
            penalty_logger.error(f"  position_size={self.position_size:.6f} -> {target_position_size:.6f}")
            # Apply severe penalty and block the trade
            if not hasattr(self, 'critical_trade_block_penalty'):
                self.critical_trade_block_penalty = 0.0
            self.critical_trade_block_penalty += 1.0  # Severe penalty
            trade_size = 0.0  # Block the trade completely
            final_trade_value = 0.0
            final_estimated_fee = 0.0
        
        # Log if this is still a large trade for monitoring
        if final_estimated_fee > 100:
            episode_logger.info(f"LARGE_TRADE_DETECTED: Trade value=${final_trade_value:.2f}, Fee=${final_estimated_fee:.2f}")
            episode_logger.info(f"  Position: {self.position_size:.6f} -> {target_position_size:.6f} BTC")
            logging.info(f"  Leverage: {target_leverage:.2f}x, Risk: {risk_percentage*100:.1f}%")
        
        if abs(trade_size) > 0.001:  # Only trade if significant change
            # Efficient trade execution - single order instead of close + open
            self._execute_efficient_trade(target_position_size, current_price)
    
    def _execute_efficient_trade(self, target_position_size: float, current_price: float):
        """
        Execute trade efficiently by calculating net position change.
        This simulates real exchange behavior where position flips are handled as single orders.
        """
        trade_size = target_position_size - self.position_size
        
        # ZERO PnL PREVENTION: Enhanced minimum trade size validation
        if abs(trade_size) < 0.001:
            return  # No significant trade needed
        
        # ZERO PnL PREVENTION: Validate target position size is meaningful
        min_position_value = 1.0  # Minimum $1 position value
        target_position_value = abs(target_position_size * current_price)
        if target_position_value < min_position_value and abs(target_position_size) > 0:
            logging.debug(f"TINY_POSITION_PREVENTED: Position value ${target_position_value:.6f} below minimum ${min_position_value}")
            return  # Prevent meaningless tiny positions that create zero PnL trades
        
        # ADDITIONAL CHECK: Prevent excessive position modifications in same step
        if hasattr(self, '_last_modification_step') and self._last_modification_step == self.current_step:
            if hasattr(self, '_modifications_this_step'):
                self._modifications_this_step += 1
            else:
                self._modifications_this_step = 1
            
            if self._modifications_this_step > 3:  # Max 3 modifications per step
                logging.debug(f"EXCESSIVE_MODIFICATIONS: Preventing {self._modifications_this_step}th modification in step {self.current_step}")
                # Apply penalty for excessive modifications
                if not hasattr(self, 'excessive_modification_penalty'):
                    self.excessive_modification_penalty = 0.0
                self.excessive_modification_penalty += 0.05  # Strong penalty for excessive modifications
                penalty_logger.error(f"EXCESSIVE_MODIFICATION_PENALTY: step={self.current_step}, modifications={self._modifications_this_step}, penalty=0.05")
                return
        else:
            self._last_modification_step = self.current_step
            self._modifications_this_step = 1
        
        # Count ANY significant trade execution (BUY/SELL action)
        self.episode_trades += 1
        
        # Calculate realized PnL if we have an existing position being modified
        realized_pnl = 0.0
        
        # ZERO PnL PREVENTION: Validate prices before calculating PnL
        if current_price <= 0 or np.isnan(current_price):
            logging.error(f"ZERO_PNL_PREVENTION: Invalid current_price={current_price} at step {self.current_step}")
            return  # Abort trade to prevent zero PnL phantom execution
        
        # ZERO PnL PREVENTION: Validate position size is not NaN
        if np.isnan(target_position_size):
            logging.error(f"ZERO_PNL_PREVENTION: NaN target_position_size at step {self.current_step}")
            return  # Abort trade to prevent invalid execution
        
        if self.position_size != 0:
            # ZERO PnL PREVENTION: Validate existing position has valid entry price
            if self.entry_price <= 0 or np.isnan(self.entry_price):
                # Silent penalty - only log at debug level since penalty is applied
                logging.debug(f"ZERO_PNL_PREVENTION: Existing position has invalid entry_price={self.entry_price}")
                # EMERGENCY FIX: Set entry price to current price to prevent zero PnL
                self.entry_price = current_price
                episode_logger.info(f"EMERGENCY_ENTRY_PRICE_FIX: Set entry_price to current_price={current_price}")
                # Apply penalty for causing this emergency fix
                if not hasattr(self, 'zero_pnl_prevention_penalty'):
                    self.zero_pnl_prevention_penalty = 0.0
                self.zero_pnl_prevention_penalty += 0.15  # Heavy penalty for invalid state
                logging.debug(f"ZERO_PNL_PREVENTION_PENALTY: -{self.zero_pnl_prevention_penalty:.4f} for invalid entry price")
                # Also ensure trade_start_step is set
                if not self.trade_start_step:
                    self.trade_start_step = self.current_step
                # Continue with trade execution - this is now safe
            
            # We're modifying an existing position
            if np.sign(target_position_size) != np.sign(self.position_size):
                # Position flip: calculate PnL on the closed portion - FIXED: Use abs(position_size) for correct short calculations
                if self.position_side == 1:  # Closing long
                    realized_pnl = abs(self.position_size) * (current_price - self.entry_price)
                else:  # Closing short
                    realized_pnl = abs(self.position_size) * (self.entry_price - current_price)
                
                # ZERO PnL PREVENTION: Validate calculated PnL makes sense
                if abs(realized_pnl) < 0.001 and abs(current_price - self.entry_price) > 0.001:
                    logging.debug(f"ZERO_PNL_DETECTED: Suspiciously small PnL={realized_pnl:.6f} despite price movement from {self.entry_price} to {current_price}")
                    # Apply penalty for zero PnL detection
                    if not hasattr(self, 'zero_pnl_detection_penalty'):
                        self.zero_pnl_detection_penalty = 0.0
                    self.zero_pnl_detection_penalty += 0.025  # Penalty for zero PnL anomaly
                    penalty_logger.error(f"ZERO_PNL_DETECTION_PENALTY: PnL={realized_pnl:.6f}, price_move={abs(current_price - self.entry_price):.6f}, penalty=0.025")
                    # Still allow the trade but flag it for investigation
                
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
        base_fee = trade_value * self.taker_fee
        
        # EMERGENCY FEE CAP: Prevent unrealistic fees (only as safety net for truly broken trades)
        max_reasonable_fee = trade_value * 0.01  # 1% maximum fee rate
        trading_fee = min(base_fee, max_reasonable_fee)
        
        # DIAGNOSTIC LOGGING: Track high fee scenarios
        if trading_fee > 100:  # Log any fee over $100
            logging.error(f"HIGH_FEE_DETECTED: Fee=${trading_fee:.2f} on trade_value=${trade_value:.2f}")
            logging.error(f"  - trade_size={trade_size:.6f}, current_price=${current_price:.2f}")
            logging.error(f"  - position_size={self.position_size:.6f} -> {target_position_size:.6f}")
            logging.error(f"  - step={self.current_step}, taker_fee={self.taker_fee}")
        
        # Log when fee cap is applied for debugging
        if trading_fee < base_fee:
            logging.debug(f"FEE_CAP_APPLIED at step {self.current_step}: Fee capped from ${base_fee:.2f} to ${trading_fee:.2f} on trade value ${trade_value:.2f}")
            # Apply penalty for hitting fee cap
            if not hasattr(self, 'fee_cap_penalty'):
                self.fee_cap_penalty = 0.0
            self.fee_cap_penalty += 0.08  # Strong penalty for hitting fee cap
            penalty_logger.error(f"FEE_CAP_PENALTY: base_fee=${base_fee:.2f}, capped_fee=${trading_fee:.2f}, penalty=0.08")
        
        # ZERO PnL PREVENTION: If fees would exceed position value, abort trade
        if trading_fee > trade_value * 0.5:  # Fees can't be more than 50% of trade value
            logging.error(f"EXCESSIVE_FEE_PREVENTED: Fee ${trading_fee:.2f} > 50% of trade value ${trade_value:.2f}")
            return  # Abort trade
        
        # Update balance with realized PnL and fees
        self.balance += realized_pnl - trading_fee
        self.total_fees += trading_fee
        
        # Update position
        old_position_size = self.position_size
        self.position_size = target_position_size
        
        # Store the target position size for logging BEFORE validation (CRITICAL FIX)
        # This ensures trade logs show the intended position size, not the validated size
        final_position_size_for_logging = self.position_size
        
        # POSITION STATE VALIDATION: Ensure consistency after position update
        # Only validate if there was a meaningful change to avoid excessive corrections
        if abs(old_position_size - self.position_size) > 0.0001:
            self._validate_and_fix_position_state()
        
        if abs(self.position_size) > 0.001:
            # We have a position
            self.position_side = 1 if self.position_size > 0 else -1
            
            # Update entry price for new or flipped positions
            if old_position_size == 0 or np.sign(old_position_size) != np.sign(self.position_size):
                # New position or position flip
                # SAFETY CHECK: Ensure entry price is valid
                if current_price <= 0:
                    logging.error(f"INVALID_ENTRY_PRICE: Cannot set entry_price to {current_price}, using fallback")
                    current_price = self._safe_get_price_data(self.current_step, 'close', 50000.0)
                
                self.entry_price = current_price
                # Only increment trade_id for non-FLIP operations (FLIP handles its own ID management)
                if old_position_size == 0:  # True new position, not a flip
                    self.trade_id += 1
                self.trade_start_step = self.current_step
                # CRITICAL FIX: Store the original entry datetime for consistent logging
                self.trade_entry_datetime = self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}"
                # CRITICAL FIX: Store the original entry price for trade logging consistency
                self.trade_entry_price = current_price
                # Store equity at trade entry for consistent net worth tracking
                self.entry_equity = self.equity
                # Reset reward accumulation for new trade
                self.current_trade_reward = 0.0
                
                logging.debug(f"NEW_POSITION: entry_price=${self.entry_price:.2f}, size={self.position_size:.6f}, step={self.current_step}")
            
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
            # Determine clear action type based on position changes
            if old_position_size == 0:
                action_type = "OPEN_LONG" if self.position_size > 0 else "OPEN_SHORT"
            elif abs(self.position_size) < 0.001:
                action_type = "CLOSE_LONG" if old_position_size > 0 else "CLOSE_SHORT" 
            elif old_position_size != 0 and np.sign(old_position_size) != np.sign(self.position_size):
                # CRITICAL FIX: FLIP operations need TWO separate trade logs
                if abs(old_position_size) > 0.001:
                    # Calculate trade duration for the closing trade
                    close_duration_steps = self.current_step - (self.trade_start_step or self.current_step)
                    close_duration_hours = close_duration_steps * 0.25  # 15min intervals
                    
                    # Log the CLOSE of the old position first
                    close_action_type = "CLOSE_LONG" if old_position_size > 0 else "CLOSE_SHORT"
                    close_trade_data = {
                        'trade_id': self._get_unique_trade_id(),
                        'training_step': self.current_step,
                        'training_iteration': getattr(self, 'training_iteration', 0),
                        'entry_datetime': self.trade_entry_datetime or (self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}"),
                        'close_datetime': self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}",
                        'side': 'FLAT',
                        'entry_action': close_action_type,
                        'entry_price': round(self.trade_entry_price or self.entry_price, 4),  # FIXED: Use stored trade entry price for consistency
                        'close_price': round(current_price, 4),
                        'net_pnl': realized_pnl,
                        'close_reward': 0,
                        'entry_net_worth': getattr(self, 'entry_equity', self.equity),  # Use stored entry equity
                        'close_net_worth': self.equity,
                        'trade_duration_hours': close_duration_hours,
                        'status': 'CLOSED',
                        'win_loss': 'WIN' if realized_pnl > 0 else 'LOSS' if realized_pnl < 0 else 'NEUTRAL',
                        'position_size': abs(old_position_size),  # FIXED: Log actual closed position size instead of 0.0
                        'fees_paid': trading_fee * 0.5,  # Split fee between close and open
                        'stop_loss_price': '',
                        'take_profit_price': '',
                        'close_reason': close_action_type
                    }
                    self.logger.log_trade(close_trade_data)
                    
                    # Increment trade ID for the new position
                    self.trade_id += 1
                    
                    # CRITICAL FIX: Reset trade_start_step for the new trade
                    self.trade_start_step = self.current_step
                    
                    # CRITICAL FIX: Reset entry datetime for the new trade
                    self.trade_entry_datetime = self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}"
                    # CRITICAL FIX: Store the original entry price for trade logging consistency  
                    self.trade_entry_price = current_price
                    # Store equity at trade entry for consistent net worth tracking
                    self.entry_equity = self.equity
                    
                    # Log the OPEN of the new position (new trade starts with 0 duration)
                    open_action_type = "OPEN_LONG" if final_position_size_for_logging > 0 else "OPEN_SHORT"
                    open_trade_data = {
                        'trade_id': self._get_unique_trade_id(),
                        'training_step': self.current_step,
                        'training_iteration': getattr(self, 'training_iteration', 0),
                        'entry_datetime': self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}",
                        'close_datetime': '',
                        'side': 'LONG' if final_position_size_for_logging > 0 else 'SHORT',
                        'entry_action': open_action_type,
                        'entry_price': current_price,
                        'close_price': '',
                        'net_pnl': 0.0,  # New position starts with zero PnL
                        'close_reward': 0,
                        'entry_net_worth': self.equity,  # For new trades, current equity is the entry equity
                        'close_net_worth': self.equity,
                        'trade_duration_hours': 0,  # New trade starts with 0 duration
                        'status': 'OPEN',
                        'win_loss': 'NEUTRAL',
                        'position_size': final_position_size_for_logging,  # Use intended position size for FLIP OPEN
                        'fees_paid': trading_fee * 0.5,  # Split fee between close and open
                        'stop_loss_price': getattr(self, 'stop_loss_price', ''),
                        'take_profit_price': getattr(self, 'take_profit_price', ''),
                        'close_reason': ''  # New OPEN trade has no close reason yet
                    }
                    self.logger.log_trade(open_trade_data)
                    
                    # Set flag to prevent duplicate logging
                    self._efficient_trade_logged = True
                    return  # Exit early since we've handled FLIP logging
                else:
                    # No position to flip, treat as new position
                    action_type = "OPEN_LONG" if self.position_size > 0 else "OPEN_SHORT"
                    print(f"WARNING: Step {self.current_step}: Cannot flip position {old_position_size}, treating as OPEN")
            else:
                action_type = "ADJUST_LONG" if self.position_size > 0 else "ADJUST_SHORT"
            
            # Standard single-trade logging for non-FLIP operations
            # Determine the correct entry price for logging
            # For new positions/opens, use current_price
            # For closes, use the original trade_entry_price to preserve consistency
            log_entry_price = current_price
            if action_type in ["CLOSE_LONG", "CLOSE_SHORT"] and hasattr(self, 'trade_entry_price') and self.trade_entry_price > 0:
                log_entry_price = self.trade_entry_price
            
            # Calculate trade duration for closing trades
            if action_type in ["CLOSE_LONG", "CLOSE_SHORT"]:
                # DURATION FIX: Ensure we have a valid trade_start_step
                if self.trade_start_step is not None and self.trade_start_step <= self.current_step:
                    duration_steps = self.current_step - self.trade_start_step
                    duration_hours = duration_steps * 0.25  # 15min intervals
                else:
                    # Fallback: if trade_start_step is invalid, assume 1 step minimum
                    duration_hours = 0.25  # Minimum 15 minutes
                    logging.debug(f"DURATION_FALLBACK: trade_start_step={self.trade_start_step}, using minimum duration")
            else:
                duration_hours = 0  # New trades start with 0 duration
            
            # Determine the correct entry datetime
            # CRITICAL FIX: Always use the stored trade_entry_datetime for consistency
            if hasattr(self, 'trade_entry_datetime') and self.trade_entry_datetime is not None:
                entry_datetime = self.trade_entry_datetime
            elif action_type in ["OPEN_LONG", "OPEN_SHORT"]:
                # For new positions, use current timestamp and store it
                entry_datetime = self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}"
                self.trade_entry_datetime = entry_datetime  # Store for future use in this trade
            else:
                # Fallback: try to reconstruct from trade_start_step (but DON'T overwrite trade_entry_datetime)
                if hasattr(self, 'trade_start_step') and self.trade_start_step is not None and self.trade_start_step < len(self.df):
                    entry_datetime = self.df.iloc[self.trade_start_step]['timestamp']
                    # DO NOT overwrite trade_entry_datetime here - it should persist from original entry
                else:
                    # Last resort fallback (also don't overwrite)
                    entry_datetime = self.df.iloc[self.current_step]['timestamp'] if self.current_step < len(self.df) else f"step_{self.current_step}"
            
            # Create trade data dictionary for logging
            trade_data = {
                'trade_id': self._get_unique_trade_id(),
                'training_step': self.current_step,
                'training_iteration': getattr(self, 'training_iteration', 0),
                'entry_datetime': entry_datetime,
                'close_datetime': self.df.iloc[self.current_step]['timestamp'] if action_type in ["CLOSE_LONG", "CLOSE_SHORT"] else '',  # Set close datetime for closed trades
                'side': 'LONG' if final_position_size_for_logging > 0 else 'SHORT' if final_position_size_for_logging < 0 else 'FLAT',
                'entry_action': action_type,
                'entry_price': round(log_entry_price, 4),  # Round to 4 decimal places for precision
                'close_price': round(current_price, 4) if action_type in ["CLOSE_LONG", "CLOSE_SHORT"] else '',  # Round and set price for closed trades
                'net_pnl': round(realized_pnl, 6) if action_type in ["CLOSE_LONG", "CLOSE_SHORT"] else 0.0,  # Round PnL and only show on trade closure
                'close_reward': 0,  # Will be filled when trade closes
                'entry_net_worth': getattr(self, 'entry_equity', self.equity),  # Use stored entry equity
                'close_net_worth': self.equity,
                'trade_duration_hours': duration_hours,
                'status': 'OPEN' if abs(final_position_size_for_logging) > 0.001 else 'CLOSED',
                'win_loss': 'WIN' if realized_pnl > 0 and action_type in ["CLOSE_LONG", "CLOSE_SHORT"] else 'LOSS' if realized_pnl < 0 and action_type in ["CLOSE_LONG", "CLOSE_SHORT"] else 'NEUTRAL',
                'position_size': final_position_size_for_logging,  # Use the intended position size, not the validated one
                'fees_paid': trading_fee,
                'stop_loss_price': getattr(self, 'stop_loss_price', ''),
                'take_profit_price': getattr(self, 'take_profit_price', ''),
                'entry_signal': action_type if action_type in ["OPEN_LONG", "OPEN_SHORT"] else '',  # Entry signal for opening trades
                'exit_signal': action_type if action_type in ["CLOSE_LONG", "CLOSE_SHORT"] else '',  # Exit signal for closing trades
                'close_reason': action_type if action_type in ["CLOSE_LONG", "CLOSE_SHORT"] else ''  # Only show close reason when trade is actually closed
            }
            
            self.logger.log_trade(trade_data)
            
            # Set flag to prevent duplicate logging in _close_position
            if action_type in ["CLOSE_LONG", "CLOSE_SHORT"]:
                self._efficient_trade_logged = True
    
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
            current_atr = self._safe_get_feature_data(self.current_step, 'atr', 0.01)
            
            # Handle NaN or invalid ATR values
            if pd.isna(current_atr) or current_atr <= 0:
                # Fallback: use recent ATR or calculate a simple estimate
                start_idx = max(0, self.current_step-10)
                end_idx = min(self.current_step+1, len(self.feature_columns))
                if end_idx > start_idx:
                    recent_atr_values = self.feature_columns.iloc[start_idx:end_idx]['atr'].dropna()
                    if len(recent_atr_values) > 0:
                        current_atr = recent_atr_values.iloc[-1]
                    else:
                        # Final fallback: estimate ATR as 1% of current price
                        current_atr = current_price * 0.01
                else:
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
            current_price = self._safe_get_price_data(self.current_step, 'close', 0.0)
            current_atr = self._get_current_atr(current_price)
            atr_percentage = current_atr / current_price if current_price > 0 else 0
            
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
    
    def _validate_and_fix_position_state(self):
        """
        POSITION STATE VALIDATION (Fix #3): Validate and fix position state consistency.
        
        Ensures:
        1. Position side matches position size sign
        2. Entry price is valid when position is open
        3. Position variables are consistent
        4. Edge cases are handled gracefully
        
        CRITICAL: Now tracks corrections and applies penalties to discourage invalid states
        """
        # Initialize position state penalty tracking
        if not hasattr(self, 'position_state_penalty'):
            self.position_state_penalty = 0.0
        
        position_state_penalty = 0.0
        corrections_made = 0
        
        # Fix position side based on position size
        if abs(self.position_size) < 0.001:
            # No meaningful position
            if self.position_side != 0:
                # Silent penalty - only log at debug level since penalty is applied
                logging.debug(f"POSITION_STATE_FIX: Correcting position_side from {self.position_side} to 0 (no position)")
                position_state_penalty += 0.02  # Small penalty for incorrect position side
                corrections_made += 1
                self.position_side = 0
            
            # Reset position-related variables for closed positions
            # BUT ONLY if position_size is actually zero (not just small)
            # CRITICAL FIX: Use stricter threshold to avoid resetting valid small positions
            if abs(self.position_size) < 0.00001:  # Much stricter threshold (0.00001 BTC ≈ $0.50)
                if self.entry_price != 0.0 or self.margin_used != 0.0:
                    episode_logger.info("POSITION_STATE_FIX: Resetting position variables for truly closed position")
                    self.entry_price = 0.0
                    self.margin_used = 0.0
                    self.unrealized_pnl = 0.0
                    self.stop_loss_price = None
                    self.take_profit_price = None
                    self.liquidation_price = None
            else:
                # Small but non-zero position - ensure it has valid entry price
                if self.entry_price <= 0:
                    current_price = self._safe_get_price_data(self.current_step, 'close', 50000.0)
                    # Silent penalty - only log at debug level since penalty is applied
                    logging.debug(f"POSITION_STATE_FIX: Small position {self.position_size:.6f} missing entry price, using {current_price}")
                    position_state_penalty += 0.05  # Larger penalty for invalid entry price
                    corrections_made += 1
                    self.entry_price = current_price
                    if not self.trade_start_step:
                        self.trade_start_step = self.current_step
        else:
            # We have a meaningful position
            expected_side = 1 if self.position_size > 0 else -1
            
            if self.position_side != expected_side:
                # Silent penalty - only log at debug level since penalty is applied
                logging.debug(f"POSITION_STATE_FIX: Correcting position_side from {self.position_side} to {expected_side}")
                position_state_penalty += 0.03  # Penalty for wrong position side
                corrections_made += 1
                self.position_side = expected_side
            
            # Validate entry price for open positions
            if self.entry_price <= 0:
                # Try to get current price as fallback
                current_price = self._safe_get_price_data(self.current_step, 'close', 50000.0)
                # Silent penalty - only log at debug level since penalty is applied
                logging.debug(f"POSITION_STATE_FIX: Invalid entry_price {self.entry_price}, using fallback {current_price}")
                position_state_penalty += 0.1  # Heavy penalty for invalid entry price on active position
                corrections_made += 1
                self.entry_price = current_price
                # Also set trade start step if missing
                if not self.trade_start_step:
                    self.trade_start_step = self.current_step
                    episode_logger.info(f"POSITION_STATE_FIX: Set missing trade_start_step to {self.current_step}")
        
        # Validate numeric consistency
        if np.isnan(self.position_size):
            penalty_logger.error(f"POSITION_STATE_FIX: NaN position_size detected, resetting to 0 at step {getattr(self, 'current_step', 'unknown')}")
            logging.debug("POSITION_STATE_FIX: NaN position_size detected, resetting to 0")
            position_state_penalty += 0.2  # Very heavy penalty for NaN states
            corrections_made += 1
            self.position_size = 0.0
            self.position_side = 0
        
        if np.isnan(self.entry_price):
            current_price = self._safe_get_price_data(self.current_step, 'close', 50000.0)
            penalty_logger.error(f"POSITION_STATE_FIX: NaN entry_price detected, using {current_price} at step {getattr(self, 'current_step', 'unknown')}")
            logging.debug(f"POSITION_STATE_FIX: NaN entry_price detected, using {current_price}")
            position_state_penalty += 0.15  # Heavy penalty for NaN entry price
            corrections_made += 1
            self.entry_price = current_price if abs(self.position_size) > 0.001 else 0.0
        
        # Apply escalating penalties for multiple corrections in same step
        if corrections_made > 1:
            chaos_multiplier = 1 + corrections_made * 0.5
            position_state_penalty *= chaos_multiplier  # Multiply penalty for chaos
            # Log chaos to separate file - keeps terminal clean
            penalty_logger.error(f"POSITION_STATE_CHAOS_PENALTY: {corrections_made} corrections needed, penalty x{chaos_multiplier:.1f} at step {getattr(self, 'current_step', 'unknown')}")
            # Optional: Also log to debug for development
            logging.debug(f"POSITION_STATE_CHAOS_PENALTY: {corrections_made} corrections needed, penalty x{chaos_multiplier:.1f}")
        
        # Store the penalty for reward calculation
        self.position_state_penalty = position_state_penalty
        
        if position_state_penalty > 0:
            # Silent penalty - only log at debug level since penalty is applied
            logging.debug(f"POSITION_STATE_PENALTY: -{position_state_penalty:.4f} for {corrections_made} state corrections")
        
        # Log state after validation (for debugging) - reduced frequency
        if hasattr(self, 'current_step') and self.current_step % 500 == 0:  # Reduced from 100 to 500
            logging.debug(f"POSITION_STATE: size={self.position_size:.6f}, side={self.position_side}, "
                         f"entry=${self.entry_price:.2f}, margin=${self.margin_used:.2f}")
    
    def _open_or_adjust_position(self, target_position_size: float, current_price: float):
        """
        DEPRECATED: This method is kept for compatibility but is inefficient.
        Use _execute_efficient_trade instead for normal position changes.
        """
        logging.warning("Using deprecated _open_or_adjust_position. Consider using _execute_efficient_trade instead.")
        
        # For compatibility, just call the efficient method
        self._execute_efficient_trade(target_position_size, current_price)
    
    # REMOVED: _close_position method (legacy code)
    # This method was unused - all trade closures now go through _execute_efficient_trade() for consistency
    
    def _force_close_position_no_fees(self, current_price: float, reason: str):
        """
        Force close position at episode end without charging fees.
        CRITICAL FIX: Updates existing OPEN trade instead of creating new entry.
        """
        if self.position_size == 0:
            return
        
        # Validate price
        if current_price <= 0:
            current_price = self._safe_get_price_data(self.current_step, 'close')
        
        # Calculate PnL without fees
        # Calculate PnL for forced closure - FIXED: Use abs(position_size) for correct short calculations  
        if self.position_side == 1:  # Long
            pnl = abs(self.position_size) * (current_price - self.entry_price)
        else:  # Short
            pnl = abs(self.position_size) * (self.entry_price - current_price)
        
        # Update balance with PnL only (NO FEES for episode cleanup)
        self.balance += pnl
        self.total_realized_pnl += pnl
        self.last_trade_pnl = pnl
        
        # CRITICAL FIX: Log closure of existing trade (NOT a new trade)
        if self.logger:
            # Calculate duration for episode cleanup
            cleanup_duration_steps = self.current_step - (self.trade_start_step or self.current_step)
            cleanup_duration_hours = cleanup_duration_steps * 0.25  # 15min intervals
            
            # Use SAME trade_id to update existing OPEN trade record
            trade_data = {
                'trade_id': self._get_unique_trade_id(),  # SAME ID as the open trade
                'training_step': self.current_step,
                'training_iteration': getattr(self, 'training_iteration', 0),
                'entry_datetime': self.trade_entry_datetime or self._safe_get_df_data(self.trade_start_step or self.current_step, 'timestamp', f"step_{self.trade_start_step or self.current_step}"),
                'close_datetime': self._safe_get_df_data(self.current_step, 'timestamp', f"step_{self.current_step}"),
                'side': 'FLAT',  # Position is now closed
                'entry_action': f"FORCE_CLOSE_{reason}",  # Clear action indicating forced closure
                'entry_price': self.entry_price,
                'close_price': current_price,
                'net_pnl': pnl,
                'close_reward': 0.0,
                'entry_net_worth': getattr(self, 'entry_equity', self.equity),  # Use stored entry equity
                'close_net_worth': self.equity,
                'trade_duration_hours': cleanup_duration_hours,
                'status': 'CLOSED',  # CRITICAL: Mark as CLOSED, not FORCE_CLOSED
                'win_loss': 'WIN' if pnl > 0 else 'LOSS' if pnl < 0 else 'NEUTRAL',
                'position_size': 0.0,  # Position is now zero
                'fees_paid': 0.0,  # NO FEES for episode cleanup
                'stop_loss_price': '',
                'take_profit_price': '',
                'close_reason': f"FORCE_CLOSE_{reason}"
            }
            self.logger.log_trade(trade_data)
            
            # Log episode cleanup info for debugging
            episode_logger.info(f"EPISODE_CLEANUP: Closed trade {self.trade_id:05d} at step {self.current_step}")
            episode_logger.info(f"  Position: {self.position_size:.6f} -> 0.0, PnL: ${pnl:.2f}")
        
        # Reset position completely
        self.position_size = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        self.trade_entry_price = 0.0  # Reset trade entry price
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.leverage = 0.0
        self.stop_loss_price = None
        self.take_profit_price = None
        self.liquidation_price = None
        self.current_trade_reward = 0.0
        
        # Update episode stats
        self.episode_profit += pnl
        
        logging.info(f"FORCE_CLOSED position at episode end: PnL={pnl:.2f}, NO FEES CHARGED")
    
    def _log_trade(self, exit_price: float, pnl: float, reason: str):
        """Log completed trade"""
        if not self.trade_start_step:
            return
        
        # PRICE VALIDATION: Ensure exit_price is valid
        if np.isnan(exit_price) or exit_price <= 0:
            # Use current market price as fallback
            current_price = self._safe_get_price_data(self.current_step, 'close', 0)
            if np.isnan(current_price) or current_price <= 0:
                # Last resort: use entry price
                exit_price = getattr(self, 'entry_price', 40000.0)
                print(f"WARNING: Step {self.current_step}: Invalid exit price, using entry price {exit_price}")
            else:
                exit_price = current_price
                print(f"WARNING: Step {self.current_step}: Invalid exit price, using current price {exit_price}")
        
        duration_steps = self.current_step - self.trade_start_step
        duration_hours = duration_steps * 0.25  # 15min intervals
        
        # Use stored entry datetime for consistency, fallback to timestamp lookup
        entry_datetime = self.trade_entry_datetime or self._safe_get_price_data(self.trade_start_step, 'timestamp', 0)
        close_datetime = self._safe_get_price_data(self.current_step, 'timestamp', 0)
        
        # Preserve entry price - if it's zero, try to get it from trade start data
        logged_entry_price = self.entry_price
        if logged_entry_price <= 0.0 and self.trade_start_step:
            # Fall back to using price data from trade start step
            logged_entry_price = self._safe_get_price_data(self.trade_start_step, 'close')
            logging.debug(f"ENTRY_PRICE_FALLBACK: Using price from trade start step {self.trade_start_step}: {logged_entry_price}")
            # Apply penalty for entry price fallback
            if not hasattr(self, 'entry_price_fallback_penalty'):
                self.entry_price_fallback_penalty = 0.0
            self.entry_price_fallback_penalty += 0.01  # Small penalty for entry price fallback
            penalty_logger.error(f"ENTRY_PRICE_FALLBACK_PENALTY: fallback_price={logged_entry_price}, penalty=0.01")
        
        # Determine clear action label based on what actually happened
        if reason in ["CANCEL_ACTION", "LIQUIDATION", "STOP_LOSS", "TAKE_PROFIT"]:
            action_label = "CLOSE"
        elif self.position_side == 1:  # Was long position
            action_label = "CLOSE_LONG"  # Closing a long position (selling)
        else:  # Was short position  
            action_label = "CLOSE_SHORT"  # Closing a short position (covering)
        
        trade_data = {
            'trade_id': self._get_unique_trade_id(),
            'training_step': self.current_step,
            'training_iteration': self.training_iteration,
            'entry_datetime': entry_datetime,
            'close_datetime': close_datetime,
            'side': 'LONG' if self.position_side == 1 else 'SHORT',
            'entry_action': action_label,
            'entry_price': logged_entry_price,
            'close_price': exit_price,
            'net_pnl': pnl,
            'close_reward': getattr(self, 'current_trade_reward', 0.0),  # Actual cumulative reward for this trade
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
        # Calculate unrealized PnL at mark price - FIXED: Use abs(position_size) for correct short calculations
        if self.position_size == 0:
            unrealized_pnl = 0.0
        elif self.position_side == 1:  # Long
            unrealized_pnl = abs(self.position_size) * (mark_price - self.entry_price)
        else:  # Short
            unrealized_pnl = abs(self.position_size) * (self.entry_price - mark_price)
        
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
        
        current_price = self._safe_get_price_data(self.current_step, 'close', 0.0)
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
            self.trade_entry_price = 0.0  # Reset trade entry price
            self.margin_used = 0.0
            self.unrealized_pnl = 0.0
            self.leverage = 0.0
            self.stop_loss_price = None
            self.take_profit_price = None
            
            # Update episode stats
            # Note: episode_trades is now counted in _execute_efficient_trade
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
        
        # === ANTI-OVERTRADING PENALTY (NEW) ===
        overtrading_penalty = 0.0
        
        # Track recent action frequency
        if not hasattr(self, 'recent_actions'):
            self.recent_actions = []
        
        # Record current action (if not HOLD)
        if hasattr(self, 'last_action_type') and self.last_action_type != "HOLD":
            self.recent_actions.append(self.current_step)
            
        # Keep only last 10 steps
        self.recent_actions = [step for step in self.recent_actions if step > self.current_step - 10]
        
        # Penalize excessive trading in recent window
        recent_trades = len(self.recent_actions)
        if recent_trades >= 3:  # 3+ trades in 10 steps = overtrading
            overtrading_penalty = 0.01 * (recent_trades - 2) ** 2  # Quadratic penalty
            
        # Penalize position flipping
        if (hasattr(self, 'last_action_type') and 
            self.last_action_type in ["FLIP_LONG_TO_SHORT", "FLIP_SHORT_TO_LONG"]):
            overtrading_penalty += 0.005  # Additional flip penalty
        
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
        
        # === PnL-AWARE POSITION MANAGEMENT REWARDS ===
        pnl_management_reward = 0.0
        
        # Track unrealized PnL trend
        if not hasattr(self, 'unrealized_pnl_history'):
            self.unrealized_pnl_history = []
        
        # Record current unrealized PnL
        self.unrealized_pnl_history.append(self.unrealized_pnl)
        
        # Keep only last 5 steps for trend analysis
        self.unrealized_pnl_history = self.unrealized_pnl_history[-5:]
        
        # Calculate PnL trend (positive = improving, negative = deteriorating)
        pnl_trend = 0.0
        if len(self.unrealized_pnl_history) >= 3:
            recent_pnl = np.mean(self.unrealized_pnl_history[-2:])
            older_pnl = np.mean(self.unrealized_pnl_history[-3:-1])
            if abs(older_pnl) > 0.01:  # Avoid division by very small numbers
                pnl_trend = (recent_pnl - older_pnl) / abs(older_pnl)
            else:
                pnl_trend = recent_pnl - older_pnl  # Simple difference for small values
        
        # Position-based rewards
        if hasattr(self, 'last_action_type'):
            if self.last_action_type == "HOLD":
                if abs(self.position_size) > 0.001:  # Holding an open position
                    # RULE 1 & 3: Encourage HOLD when unrealized PnL is improving
                    if pnl_trend > 0:
                        pnl_management_reward += 0.01 * min(pnl_trend, 0.5)  # Scale with trend strength
                        pnl_management_reward += 0.005  # Base reward for holding profitable trend
                    
                    # Slight penalty for holding when PnL is deteriorating significantly
                    elif pnl_trend < -0.1:  # Only if strong negative trend
                        pnl_management_reward -= 0.002 * min(abs(pnl_trend), 0.3)
                    
                    # Additional reward for holding profitable positions
                    if self.unrealized_pnl > 0:
                        pnl_management_reward += 0.002  # Encourage keeping profits
                else:
                    # No position - small reward for patience
                    pnl_management_reward += 0.001
            
            elif self.last_action_type in ["CANCEL", "CLOSE"]:
                if abs(self.position_size) < 0.001:  # Successfully closed position
                    # RULE 2: Encourage CLOSE when unrealized PnL was deteriorating
                    if pnl_trend < -0.05:  # Strong negative trend
                        pnl_management_reward += 0.01 * min(abs(pnl_trend), 0.5)  # Reward cutting losses
                        pnl_management_reward += 0.008  # Base reward for smart exit
                    
                    # Reward taking profits (even if trend was positive)
                    if len(self.unrealized_pnl_history) >= 2 and self.unrealized_pnl_history[-2] > 0:
                        pnl_management_reward += 0.005  # Reward profit-taking
                    
                    # Penalty for closing during strong positive PnL trend
                    if pnl_trend > 0.1:  # Strong positive trend
                        pnl_management_reward -= 0.005 * min(pnl_trend, 0.3)  # Penalty for early exit
            
            elif self.last_action_type in ["FLIP_LONG_TO_SHORT", "FLIP_SHORT_TO_LONG"]:
                # Evaluate flip based on PnL trend of previous position
                if len(self.unrealized_pnl_history) >= 2:
                    prev_pnl = self.unrealized_pnl_history[-2]
                    if pnl_trend < -0.05 and prev_pnl < 0:  # Was losing, good to flip
                        pnl_management_reward += 0.003
                    elif pnl_trend > 0.05 and prev_pnl > 0:  # Was winning, bad to flip
                        pnl_management_reward -= 0.008
        
        # === TRADITIONAL POSITIVE REWARDS (SIMPLIFIED) ===
        positive_bonus = pnl_management_reward  # Start with PnL-aware rewards
        
        # Penalize excessive trading (overtrading)
        if hasattr(self, 'trade_streak') and self.trade_streak > 5:
            positive_bonus -= 0.001 * (self.trade_streak - 5)  # Increasing penalty for overtrading
        
        # Reward patient holding streaks (but not too long)
        if hasattr(self, 'hold_streak'):
            if 3 <= self.hold_streak <= 10:  # Optimal patience range
                positive_bonus += 0.0005 * self.hold_streak
            elif self.hold_streak > 20:  # Too much inaction
                positive_bonus -= 0.0005 * (self.hold_streak - 20)
        
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
        
        # === NEW REWARD COMPONENTS FOR IMPROVED TRADING BEHAVIOR ===
        
        # ISSUE 1: Trend Following Bonus - Reward holding while PnL grows
        if (abs(self.position_size) > 0.001 and 
            hasattr(self, 'last_action_type') and self.last_action_type == "HOLD"):
            
            # Check if unrealized PnL is improving
            if (len(self.unrealized_pnl_history) >= 2 and 
                self.unrealized_pnl > 0 and
                'trend_following_bonus' in self.reward_config):
                
                recent_pnl_change = self.unrealized_pnl_history[-1] - self.unrealized_pnl_history[-2]
                if recent_pnl_change > self.reward_config.get('trend_following_threshold', 0.01):
                    positive_bonus += self.reward_config['trend_following_bonus']
        
        # ISSUE 2: Quick Loss Cut Bonus - Reward cutting losses quickly
        if (hasattr(self, 'last_action_type') and 
            self.last_action_type in ["CANCEL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT"] and
            abs(self.position_size) < 0.001 and  # Position was closed
            'quick_loss_cut_bonus' in self.reward_config):
            
            # Check if this was a loss that was cut quickly
            if (hasattr(self, 'trade_start_step') and self.trade_start_step and
                len(self.unrealized_pnl_history) >= 1 and
                self.unrealized_pnl_history[-1] < self.reward_config.get('loss_cut_threshold', -0.01)):
                
                hold_duration = self.current_step - self.trade_start_step
                max_hold_steps = self.reward_config.get('max_loss_hold_steps', 5)
                if hold_duration <= max_hold_steps:
                    positive_bonus += self.reward_config['quick_loss_cut_bonus']
        
        # ISSUE 3: Exit Strategy Differentiation
        if hasattr(self, 'last_action_type'):
            
            # Penalty for CANCEL_CLOSE (panic exits)
            if (self.last_action_type == "CANCEL" and 
                'cancel_close_penalty' in self.reward_config):
                positive_bonus -= self.reward_config['cancel_close_penalty']
            
            # Bonus for deliberate exits
            elif (self.last_action_type in ["CLOSE_LONG", "CLOSE_SHORT"] and
                  'deliberate_exit_bonus' in self.reward_config):
                positive_bonus += self.reward_config['deliberate_exit_bonus']
                
                # Additional bonus for profit target achievement
                if ('profit_target_achievement_bonus' in self.reward_config and
                    len(self.unrealized_pnl_history) >= 1 and
                    self.unrealized_pnl_history[-1] > self.reward_config.get('profit_target_threshold', 0.005)):
                    positive_bonus += self.reward_config['profit_target_achievement_bonus']
        
        # ISSUE 4: Minimum Profitability Bonus
        if (hasattr(self, 'last_action_type') and 
            self.last_action_type in ["CLOSE", "CLOSE_LONG", "CLOSE_SHORT"] and
            abs(self.position_size) < 0.001 and  # Position was closed
            'minimum_profit_bonus' in self.reward_config):
            
            # Check if profit exceeded minimum threshold
            if (len(self.unrealized_pnl_history) >= 1 and
                self.unrealized_pnl_history[-1] > self.reward_config.get('minimum_profit_threshold', 0.005)):
                positive_bonus += self.reward_config['minimum_profit_bonus']
        
        # === COMBINE ALL COMPONENTS ===
        total_penalty = (
            risk_penalty + 
            balance_penalty + 
            consecutive_loss_penalty + 
            trend_penalty + 
            volatility_penalty + 
            cost_penalty + 
            special_penalty +
            overtrading_penalty  # Include anti-overtrading penalty
        )
        
        # Apply dynamic loss multiplier
        total_penalty *= self.loss_penalty_multiplier
        
        # === NEW PENALTY COMPONENTS FOR IMPROVED TRADING BEHAVIOR ===
        
        # ISSUE 1: Penalty for exiting profitable trends
        exit_profitable_trend_penalty = 0.0
        if (hasattr(self, 'last_action_type') and 
            self.last_action_type in ["CANCEL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT"] and
            abs(self.position_size) < 0.001 and  # Position was closed
            'exit_profitable_trend_penalty' in self.reward_config):
            
            # Check if we exited during a profitable trend
            if (len(self.unrealized_pnl_history) >= 2 and
                self.unrealized_pnl_history[-1] > 0):  # Was profitable
                
                recent_pnl_change = self.unrealized_pnl_history[-1] - self.unrealized_pnl_history[-2]
                profitable_trend_threshold = self.reward_config.get('profitable_trend_threshold', 0.02)
                
                if recent_pnl_change > profitable_trend_threshold:  # Strong positive trend
                    exit_profitable_trend_penalty = self.reward_config['exit_profitable_trend_penalty']
        
        # ISSUE 4: Small position and fee ratio penalties
        small_position_penalty = 0.0
        excessive_fee_penalty = 0.0
        
        if hasattr(self, '_last_fees') and self._last_fees > 0:
            # Penalty for small positions relative to fees
            if ('small_position_penalty' in self.reward_config and
                hasattr(self, 'position_size') and abs(self.position_size) > 0):
                
                current_price = self._safe_get_price_data(self.current_step, 'close', 50000.0)
                position_value = abs(self.position_size) * current_price
                small_threshold = self.reward_config.get('small_position_threshold', 50.0)
                
                if position_value < small_threshold:
                    small_position_penalty = self.reward_config['small_position_penalty']
            
            # Penalty for excessive fee ratios
            if ('excessive_fee_ratio_penalty' in self.reward_config and
                hasattr(self, 'position_size') and abs(self.position_size) > 0):
                
                current_price = self._safe_get_price_data(self.current_step, 'close', 50000.0)
                trade_value = abs(self.position_size) * current_price
                
                if trade_value > 0:
                    fee_ratio = self._last_fees / trade_value
                    fee_threshold = self.reward_config.get('fee_ratio_penalty_threshold', 0.1)
                    
                    if fee_ratio > fee_threshold:
                        excessive_fee_penalty = self.reward_config['excessive_fee_ratio_penalty']
        
        # Add safety intervention penalties (teach agent not to attempt extreme actions)
        safety_penalty = getattr(self, 'safety_intervention_penalty', 0.0)
        extreme_leverage_penalty = getattr(self, 'extreme_leverage_penalty', 0.0)
        position_state_penalty = getattr(self, 'position_state_penalty', 0.0)
        zero_pnl_penalty = getattr(self, 'zero_pnl_prevention_penalty', 0.0)
        dust_position_penalty = getattr(self, 'dust_position_penalty', 0.0)
        
        # Add new penalty categories for cleaner trading
        phantom_trade_penalty = getattr(self, 'phantom_trade_penalty', 0.0)
        invalid_entry_penalty = getattr(self, 'invalid_entry_penalty', 0.0)
        zero_pnl_detection_penalty = getattr(self, 'zero_pnl_detection_penalty', 0.0)
        zero_pnl_close_penalty = getattr(self, 'zero_pnl_close_penalty', 0.0)
        excessive_modification_penalty = getattr(self, 'excessive_modification_penalty', 0.0)
        fee_cap_penalty = getattr(self, 'fee_cap_penalty', 0.0)
        entry_price_fallback_penalty = getattr(self, 'entry_price_fallback_penalty', 0.0)
        critical_trade_block_penalty = getattr(self, 'critical_trade_block_penalty', 0.0)
        
        # Combine all penalty categories
        all_penalty_categories = (
            safety_penalty + extreme_leverage_penalty + position_state_penalty + zero_pnl_penalty + 
            dust_position_penalty + phantom_trade_penalty + invalid_entry_penalty + 
            zero_pnl_detection_penalty + zero_pnl_close_penalty + excessive_modification_penalty + 
            fee_cap_penalty + entry_price_fallback_penalty + critical_trade_block_penalty +
            exit_profitable_trend_penalty + small_position_penalty + excessive_fee_penalty  # NEW PENALTIES
        )
        total_penalty += all_penalty_categories
        
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
        end_idx = min(self.current_step, len(self.feature_columns_scaled))
        
        # Ensure we don't access out-of-bounds indices
        if end_idx <= start_idx or end_idx > len(self.feature_columns_scaled):
            # Return zeros if we're out of bounds
            market_features = np.zeros((self.window_size, len(self.feature_columns.columns)))
        else:
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
        
        # Calculate PnL trend for observation
        pnl_trend_feature = 0.0
        if hasattr(self, 'unrealized_pnl_history') and len(self.unrealized_pnl_history) >= 3:
            recent_pnl = np.mean(self.unrealized_pnl_history[-2:])
            older_pnl = np.mean(self.unrealized_pnl_history[-3:-1])
            if abs(older_pnl) > 0.01:
                pnl_trend_feature = np.clip((recent_pnl - older_pnl) / abs(older_pnl), -1.0, 1.0)
            else:
                pnl_trend_feature = np.clip(recent_pnl - older_pnl, -100.0, 100.0) / 100.0
        
        # ENHANCED 9-FEATURE PORTFOLIO STATE (added PnL trend)
        portfolio_features = np.array([
            # Core state (4 essential features)
            equity_ratio,  # Current equity / initial equity
            self.position_size / 2.0 if abs(self.position_size) < 2.0 else np.sign(self.position_size),  # Normalized position
            self.unrealized_pnl / self.initial_equity if self.initial_equity > 0 else 0,  # Normalized unrealized PnL
            drawdown,  # Current drawdown from peak
            
            # Risk & leverage (2 features)
            self.leverage / self.max_leverage if self.max_leverage > 0 else 0,  # Normalized leverage
            self.margin_used / self.initial_equity if self.initial_equity > 0 else 0,  # Normalized margin used
            
            # Trading behavior & PnL trend (3 features)
            min(self.consecutive_losses / 5.0, 1.0),  # Normalized consecutive losses (cap at 5)
            balance_trend,  # Recent balance change trend
            pnl_trend_feature,  # NEW: Unrealized PnL trend (-1 to +1)
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
            'last_trade_pnl': self.last_trade_pnl,
            
            # Safety intervention tracking
            'safety_intervention_penalty': getattr(self, 'safety_intervention_penalty', 0.0),
            'extreme_leverage_penalty': getattr(self, 'extreme_leverage_penalty', 0.0)
        }
    
    def render(self, mode='human'):
        """Render environment state with enhanced risk information"""
        if mode == 'human':
            current_price = self._safe_get_price_data(self.current_step, 'close', 0.0)
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
        if train_ratio + val_ratio > 1.0001:  # Allow small floating point precision errors
            raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) = {train_ratio + val_ratio:.4f} cannot exceed 1.0")
        
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
            
            # === NEW PARAMETERS FOR IMPROVED TRADING BEHAVIOR ===
            # (Default to 0.0 for backward compatibility)
            
            # ISSUE 1: Trend following bonuses
            'trend_following_bonus': 0.0,
            'trend_following_threshold': 0.01,
            'exit_profitable_trend_penalty': 0.0,
            'profitable_trend_threshold': 0.02,
            
            # ISSUE 2: Quick loss cutting rewards
            'quick_loss_cut_bonus': 0.0,
            'loss_cut_threshold': -0.01,
            'max_loss_hold_steps': 5,
            
            # ISSUE 3: Exit strategy differentiation
            'cancel_close_penalty': 0.0,
            'deliberate_exit_bonus': 0.0,
            'profit_target_achievement_bonus': 0.0,
            'profit_target_threshold': 0.005,
            
            # ISSUE 4: Minimum profitability requirements
            'minimum_profit_threshold': 0.005,
            'minimum_profit_bonus': 0.0,
            'small_position_penalty': 0.0,
            'small_position_threshold': 50.0,
            'fee_ratio_penalty_threshold': 0.1,
            'excessive_fee_ratio_penalty': 0.0,
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
