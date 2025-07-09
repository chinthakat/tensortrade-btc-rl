"""
Utility Functions for Data Processing and Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

def validate_data_format(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate if DataFrame has the required format for trading
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    
    validation_results = {
        'has_required_columns': all(col in df.columns for col in required_columns),
        'no_missing_values': not df[required_columns].isnull().any().any(),
        'correct_data_types': True,
        'chronological_order': True,
        'valid_ohlc': True,
        'positive_volume': True
    }
    
    try:
        # Check data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                pd.to_numeric(df[col], errors='raise')
        
        # Check timestamp
        if 'timestamp' in df.columns:
            pd.to_numeric(df['timestamp'], errors='raise')
            
        # Check chronological order
        if 'timestamp' in df.columns:
            timestamps = pd.to_numeric(df['timestamp'])
            validation_results['chronological_order'] = timestamps.is_monotonic_increasing
        
        # Check OHLC logic
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            high_valid = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
            low_valid = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
            validation_results['valid_ohlc'] = high_valid.all() and low_valid.all()
        
        # Check positive volume
        if 'volume' in df.columns:
            validation_results['positive_volume'] = (df['volume'] >= 0).all()
            
    except:
        validation_results['correct_data_types'] = False
    
    return validation_results

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for trading
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Convert data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove rows with invalid OHLC
    if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
        valid_ohlc = (
            (df_clean['high'] >= df_clean['low']) &
            (df_clean['high'] >= df_clean['open']) &
            (df_clean['high'] >= df_clean['close']) &
            (df_clean['low'] <= df_clean['open']) &
            (df_clean['low'] <= df_clean['close'])
        )
        df_clean = df_clean[valid_ohlc]
    
    # Remove rows with negative volume
    if 'volume' in df_clean.columns:
        df_clean = df_clean[df_clean['volume'] >= 0]
    
    # Sort by timestamp
    if 'timestamp' in df_clean.columns:
        df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    
    return df_clean

def resample_data(df: pd.DataFrame, target_interval: str = '1H') -> pd.DataFrame:
    """
    Resample data to different timeframe
    
    Args:
        df: DataFrame with timestamp column
        target_interval: Target interval (e.g., '1H', '4H', '1D')
    
    Returns:
        Resampled DataFrame
    """
    df_resampled = df.copy()
    
    # Convert timestamp to datetime
    df_resampled['datetime'] = pd.to_datetime(df_resampled['timestamp'], unit='s')
    df_resampled.set_index('datetime', inplace=True)
    
    # Resample
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    df_resampled = df_resampled.resample(target_interval).agg(ohlc_dict).dropna()
    
    # Convert back to timestamp
    df_resampled['timestamp'] = df_resampled.index.astype(int) // 10**9
    df_resampled.reset_index(drop=True, inplace=True)
    
    return df_resampled

def calculate_basic_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate basic statistics for the dataset
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    if 'close' in df.columns:
        close_prices = df['close']
        stats['price_stats'] = {
            'min_price': close_prices.min(),
            'max_price': close_prices.max(),
            'mean_price': close_prices.mean(),
            'std_price': close_prices.std(),
            'total_return': (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
        }
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        stats['return_stats'] = {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    if 'volume' in df.columns:
        volume = df['volume']
        stats['volume_stats'] = {
            'mean_volume': volume.mean(),
            'std_volume': volume.std(),
            'min_volume': volume.min(),
            'max_volume': volume.max()
        }
    
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'], unit='s')
        stats['time_stats'] = {
            'start_date': timestamps.min(),
            'end_date': timestamps.max(),
            'duration_days': (timestamps.max() - timestamps.min()).days,
            'total_periods': len(df)
        }
    
    return stats

def detect_outliers(df: pd.DataFrame, method: str = 'iqr', columns: List[str] = None) -> pd.DataFrame:
    """
    Detect outliers in the data
    
    Args:
        df: DataFrame to analyze
        method: Method to use ('iqr', 'zscore')
        columns: Columns to analyze (default: all numeric columns)
    
    Returns:
        DataFrame with outlier flags
    """
    df_outliers = df.copy()
    
    if columns is None:
        columns = ['open', 'high', 'low', 'close', 'volume']
        columns = [col for col in columns if col in df.columns]
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_outliers[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_outliers[f'{col}_outlier'] = z_scores > 3
    
    return df_outliers

def create_data_quality_report(df: pd.DataFrame, save_path: Optional[str] = None) -> Dict:
    """
    Create comprehensive data quality report
    
    Args:
        df: DataFrame to analyze
        save_path: Path to save the report plot
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'validation': validate_data_format(df),
        'basic_stats': calculate_basic_stats(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_shape': df.shape
    }
    
    # Add outlier analysis
    outlier_df = detect_outliers(df)
    outlier_columns = [col for col in outlier_df.columns if col.endswith('_outlier')]
    report['outliers'] = {col: outlier_df[col].sum() for col in outlier_columns}
    
    # Create visualization if requested
    if save_path and 'close' in df.columns:
        create_data_visualization(df, save_path)
    
    return report

def create_data_visualization(df: pd.DataFrame, save_path: str):
    """
    Create comprehensive data visualization
    
    Args:
        df: DataFrame with OHLCV data
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Quality Analysis', fontsize=16)
    
    # Price chart
    if 'close' in df.columns:
        axes[0, 0].plot(df.index, df['close'], linewidth=1)
        axes[0, 0].set_title('Price Chart')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Volume chart
    if 'volume' in df.columns:
        axes[0, 1].plot(df.index, df['volume'], linewidth=1, color='orange')
        axes[0, 1].set_title('Volume Chart')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Return distribution
    if 'close' in df.columns:
        returns = df['close'].pct_change().dropna()
        axes[1, 0].hist(returns, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Missing values heatmap
    missing_data = df.isnull()
    if missing_data.any().any():
        sns.heatmap(missing_data, cbar=True, ax=axes[1, 1])
        axes[1, 1].set_title('Missing Values')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Missing Values', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Missing Values')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def split_data_for_training(
    df: pd.DataFrame, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets
    
    Args:
        df: DataFrame to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    return train_df, val_df, test_df

def merge_multiple_symbols(file_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Merge data from multiple trading symbols
    
    Args:
        file_paths: Dictionary of {symbol: file_path}
    
    Returns:
        Merged DataFrame with symbol column
    """
    dfs = []
    
    for symbol, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        df['symbol'] = symbol
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
    
    return merged_df

if __name__ == "__main__":
    # Example usage
    print("Data utilities module - Use as import")
