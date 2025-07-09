"""
Alternative technical analysis implementation using only built-in libraries
For cases where pandas-ta is not available
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class FallbackTA:
    """Simple technical analysis indicators using only pandas and numpy"""
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window).mean()
    
    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=window).mean()
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, std_dev: float = 2.0) -> tuple:
        """Bollinger Bands - returns tuple of (upper, middle, lower)"""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        middle = sma
        
        return upper, middle, lower
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD (Moving Average Convergence Divergence) - returns tuple of (macd_line, signal_line, histogram)"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14) -> pd.Series:
        """Stochastic Oscillator %K"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k_percent
    
    @staticmethod
    def stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Stochastic Oscillator %K (alias for stochastic method)"""
        return FallbackTA.stochastic(high, low, close, window)

def add_technical_indicators_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators using fallback implementation
    Use this if pandas-ta is not available
    """
    df_enhanced = df.copy()
    
    try:
        # Try to use pandas-ta first
        import pandas_ta as ta
        
        # Basic price features
        df_enhanced['returns'] = df_enhanced['close'].pct_change()
        df_enhanced['log_returns'] = np.log(df_enhanced['close'] / df_enhanced['close'].shift(1))
        df_enhanced['high_low_pct'] = (df_enhanced['high'] - df_enhanced['low']) / df_enhanced['close']
        df_enhanced['close_open_pct'] = (df_enhanced['close'] - df_enhanced['open']) / df_enhanced['open']
        
        # Technical indicators using pandas-ta
        df_enhanced['sma_10'] = ta.sma(df_enhanced['close'], length=10)
        df_enhanced['sma_20'] = ta.sma(df_enhanced['close'], length=20)
        df_enhanced['ema_10'] = ta.ema(df_enhanced['close'], length=10)
        df_enhanced['ema_20'] = ta.ema(df_enhanced['close'], length=20)
        df_enhanced['rsi'] = ta.rsi(df_enhanced['close'], length=14)
        
        # Bollinger Bands
        bb = ta.bbands(df_enhanced['close'], length=20)
        df_enhanced['bb_upper'] = bb['BBU_20_2.0']
        df_enhanced['bb_lower'] = bb['BBL_20_2.0']
        df_enhanced['bb_middle'] = bb['BBM_20_2.0']
        
        # MACD
        macd = ta.macd(df_enhanced['close'])
        df_enhanced['macd'] = macd['MACD_12_26_9']
        df_enhanced['macd_signal'] = macd['MACDs_12_26_9']
        df_enhanced['macd_histogram'] = macd['MACDh_12_26_9']
        
        # ATR
        df_enhanced['atr'] = ta.atr(df_enhanced['high'], df_enhanced['low'], df_enhanced['close'], length=14)
        
        # Stochastic
        stoch = ta.stoch(df_enhanced['high'], df_enhanced['low'], df_enhanced['close'])
        df_enhanced['stoch_k'] = stoch['STOCHk_14_3_3']
        
        print("✅ Using pandas-ta for technical indicators")
        
    except ImportError:
        print("⚠️  pandas-ta not available, using fallback implementation")
        
        # Use fallback implementation
        simple_ta = FallbackTA()
        
        # Basic price features
        df_enhanced['returns'] = df_enhanced['close'].pct_change()
        df_enhanced['log_returns'] = np.log(df_enhanced['close'] / df_enhanced['close'].shift(1))
        df_enhanced['high_low_pct'] = (df_enhanced['high'] - df_enhanced['low']) / df_enhanced['close']
        df_enhanced['close_open_pct'] = (df_enhanced['close'] - df_enhanced['open']) / df_enhanced['open']
        
        # Technical indicators using fallback
        df_enhanced['sma_10'] = simple_ta.sma(df_enhanced['close'], 10)
        df_enhanced['sma_20'] = simple_ta.sma(df_enhanced['close'], 20)
        df_enhanced['ema_10'] = simple_ta.ema(df_enhanced['close'], 10)
        df_enhanced['ema_20'] = simple_ta.ema(df_enhanced['close'], 20)
        df_enhanced['rsi'] = simple_ta.rsi(df_enhanced['close'], 14)
        
        # Bollinger Bands
        bb = simple_ta.bollinger_bands(df_enhanced['close'], 20)
        df_enhanced['bb_upper'] = bb['bb_upper']
        df_enhanced['bb_lower'] = bb['bb_lower']
        df_enhanced['bb_middle'] = bb['bb_middle']
        
        # MACD
        macd = simple_ta.macd(df_enhanced['close'])
        df_enhanced['macd'] = macd['macd']
        df_enhanced['macd_signal'] = macd['macd_signal']
        df_enhanced['macd_histogram'] = macd['macd_histogram']
        
        # ATR
        df_enhanced['atr'] = simple_ta.atr(
            df_enhanced['high'], 
            df_enhanced['low'], 
            df_enhanced['close'], 
            14
        )
        
        # Stochastic
        df_enhanced['stoch_k'] = simple_ta.stochastic(
            df_enhanced['high'], 
            df_enhanced['low'], 
            df_enhanced['close'], 
            14
        )
    
    # Additional derived features
    df_enhanced['bb_width'] = (df_enhanced['bb_upper'] - df_enhanced['bb_lower']) / df_enhanced['bb_middle']
    df_enhanced['volume_sma'] = df_enhanced['volume'].rolling(window=20).mean()
    df_enhanced['volume_ratio'] = df_enhanced['volume'] / df_enhanced['volume_sma']
    df_enhanced['price_position'] = (df_enhanced['close'] - df_enhanced['sma_20']) / df_enhanced['sma_20']
    
    # Fill NaN values
    df_enhanced = df_enhanced.fillna(method='bfill').fillna(0)
    
    return df_enhanced

if __name__ == "__main__":
    # Test the fallback implementation
    print("Testing fallback technical analysis implementation...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.02, 100)
    prices = 40000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'close': prices * (1 + np.random.normal(0, 0.01, 100)),
        'volume': np.random.uniform(100, 1000, 100),
        'timestamp': [int(d.timestamp()) for d in dates]
    })
    
    # Test the function
    df_with_indicators = add_technical_indicators_fallback(df)
    
    print(f"✅ Successfully added {len(df_with_indicators.columns) - len(df.columns)} technical indicators")
    print("Available indicators:", [col for col in df_with_indicators.columns if col not in df.columns])
