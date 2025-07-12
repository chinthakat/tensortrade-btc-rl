#!/usr/bin/env python3
"""
CoinAPI Multi-Data Downloader
Downloads OHLCV and funding rate data using real CoinAPI endpoints
"""

import pandas as pd
import numpy as np
import requests
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

class CoinAPIDownloader:
    """
    Clean implementation of CoinAPI data downloader supporting multiple data types
    """
    
    def __init__(self, data_dir: str = "data/coinapi"):
        """Initialize the downloader with configuration"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # API Configuration
        self.api_key = os.getenv("COINAPI_API_KEY", "***REMOVED***")
        self.base_url = "https://rest.coinapi.io/v1"
        self.headers = {
            "X-CoinAPI-Key": self.api_key,
            "Accept": "application/json"
        }
        
        # Rate limiting
        self.min_request_interval = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        # Data type endpoints and configurations (updated to remove open_interest)
        self.data_configs = {
            'ohlcv': {
                'endpoint': 'ohlcv/{symbol_id}/history',
                'required_params': ['period_id'],
                'optional_params': ['time_start', 'time_end', 'limit'],
                'default_params': {'limit': 10000}
            },
            'funding_rates': {
                'endpoint': 'metrics/symbol/history',  # Updated to correct endpoint
                'required_params': ['exchange_id', 'symbol_id', 'metric_id', 'period_id'],
                'optional_params': ['time_start', 'time_end', 'limit'],
                'default_params': {
                    'exchange_id': 'BINANCEFTS',
                    'metric_id': 'DERIVATIVES_FUNDING_RATE_CURRENT',
                    'period_id': '8HRS'
                }
            }
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized CoinAPI downloader")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling"""
        if not self.api_key:
            raise ValueError("CoinAPI key required")
        
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def _map_symbol_to_coinapi(self, symbol: str, exchange: str = 'BINANCE') -> str:
        """Map trading symbol to CoinAPI format"""
        symbol = symbol.upper().replace('-', '')  # Remove dashes
        
        # If already in CoinAPI format, return as-is
        if symbol.startswith(f'{exchange}FTS_'):
            return symbol
        
        # Common mappings for Binance Futures
        mappings = {
            'BTCUSDT': 'BINANCEFTS_PERP_BTC_USDT',
            'ETHUSDT': 'BINANCEFTS_PERP_ETH_USDT',
            'ADAUSDT': 'BINANCEFTS_PERP_ADA_USDT',
            'SOLUSDT': 'BINANCEFTS_PERP_SOL_USDT',
            'BTC': 'BINANCEFTS_PERP_BTC_USDT',
            'ETH': 'BINANCEFTS_PERP_ETH_USDT'
        }
        
        if symbol in mappings:
            return mappings[symbol]
        
        # Try to auto-generate for USDT pairs
        if symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"BINANCEFTS_PERP_{base}_USDT"
        
        # Default fallback
        return "BINANCEFTS_PERP_BTC_USDT"
    
    def _map_interval_to_period(self, interval: str, data_type: str = 'ohlcv') -> str:
        """Map interval to CoinAPI period ID based on data type"""
        if data_type == 'funding_rates':
            return '8HRS'  # Funding rates are typically every 8 hours
        elif data_type == 'open_interest':
            # Map intervals to appropriate periods for open interest
            mapping = {
                '15m': '15MIN',
                '1h': '1HRS',
                '4h': '4HRS',
                '1d': '1DAY'
            }
            return mapping.get(interval, '1HRS')
        else:  # ohlcv
            mapping = {
                '1m': '1MIN',
                '5m': '5MIN',
                '15m': '15MIN',
                '30m': '30MIN',
                '1h': '1HRS',
                '4h': '4HRS',
                '1d': '1DAY'
            }
            return mapping.get(interval, '15MIN')
    
    def download_single_data_type(self, symbol: str, data_type: str, interval: str,
                                 start_date: str, end_date: str, exchange: str = 'BINANCE') -> Optional[pd.DataFrame]:
        """Download data for a single data type and date range"""
        if data_type not in self.data_configs:
            self.logger.error(f"Unsupported data type: {data_type}")
            return None
        
        # Map symbol and prepare parameters
        symbol_id = self._map_symbol_to_coinapi(symbol, exchange)
        config = self.data_configs[data_type]
        
        # Prepare parameters based on data type
        params = config['default_params'].copy()
        params['period_id'] = self._map_interval_to_period(interval, data_type)
        params['time_start'] = f"{start_date}T00:00:00"
        params['time_end'] = f"{end_date}T23:59:59"
        
        # For funding rates, we need special handling for the endpoint format
        if data_type == 'funding_rates':
            endpoint = config['endpoint']  # Don't format with symbol_id for metrics endpoint
            params['symbol_id'] = symbol_id  # Add symbol_id as parameter instead
        else:
            endpoint = config['endpoint'].format(symbol_id=symbol_id)
        
        self.logger.info(f"Downloading {data_type} for {symbol_id}: {start_date} to {end_date}")
        self.logger.debug(f"Endpoint: {endpoint}, Params: {params}")
        
        try:
            data = self._make_request(endpoint, params)
            
            if not data:
                self.logger.warning(f"No {data_type} data received for {symbol_id}")
                return None
            
            df = pd.DataFrame(data)
            df = self._standardize_data(df, data_type)
            
            self.logger.info(f"Downloaded {len(df)} {data_type} records for {symbol_id}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download {data_type} for {symbol_id}: {e}")
            return None
    
    def _standardize_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Standardize data format based on data type"""
        if df.empty:
            return df
        
        try:
            if data_type == 'ohlcv':
                # Standardize OHLCV data
                column_mapping = {
                    'time_period_start': 'timestamp',
                    'price_open': 'open',
                    'price_high': 'high',
                    'price_low': 'low',
                    'price_close': 'close',
                    'volume_traded': 'volume'
                }
                df = df.rename(columns=column_mapping)
                
                # Ensure required columns exist
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        if col == 'volume':
                            df[col] = 0.0
                        else:
                            self.logger.warning(f"Missing column {col} in OHLCV data")
                
            elif data_type == 'funding_rates':
                # Standardize funding rates data (CoinAPI returns 'last' for current rate)
                if 'time_period_start' in df.columns:
                    df = df.rename(columns={'time_period_start': 'timestamp'})
                if 'last' in df.columns:  # CoinAPI uses 'last' for funding rate value
                    df = df.rename(columns={'last': 'funding_rate'})
                elif 'sum' in df.columns:  # Fallback to 'sum' if 'last' not available
                    df = df.rename(columns={'sum': 'funding_rate'})
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to standardize {data_type} data: {e}")
            return df
    
    def download_daily_data_batch(self, symbol: str, interval: str = '15m',
                                 start_date: str = None, end_date: str = None,
                                 data_types: List[str] = None, exchange: str = 'BINANCE',
                                 force_redownload: bool = False) -> bool:
        """
        Download data in daily batches for multiple data types
        
        Args:
            symbol: Trading symbol
            interval: Data interval (15m, 1h, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_types: List of data types ['ohlcv', 'funding_rates']
            exchange: Exchange name
            force_redownload: Force re-download existing files
            
        Returns:
            bool: Success status
        """
        if data_types is None:
            data_types = ['ohlcv']
        
        # Validate data types
        valid_types = [dt for dt in data_types if dt in self.data_configs]
        if not valid_types:
            self.logger.error("No valid data types specified")
            return False
        
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            total_days = (end - start).days + 1
            
            # Create directory structure
            symbol_clean = symbol.replace('/', '_').replace(':', '_')
            base_dir = self.data_dir / symbol_clean / interval
            
            data_dirs = {}
            for data_type in valid_types:
                data_dirs[data_type] = base_dir / data_type
                data_dirs[data_type].mkdir(parents=True, exist_ok=True)
            
            print(f"🚀 Starting batch download for {symbol}")
            print(f"📅 Date range: {start_date} to {end_date} ({total_days} days)")
            print(f"📊 Data types: {', '.join(valid_types)}")
            print(f"📁 Base directory: {base_dir}")
            
            # Track progress
            stats = {dt: {'success': 0, 'skipped': 0, 'failed': 0} for dt in valid_types}
            
            current_date = start
            while current_date <= end:
                date_str = current_date.strftime('%Y-%m-%d')
                progress = (current_date - start).days + 1
                
                print(f"\n🔄 Processing {date_str} ({progress}/{total_days})")
                
                for data_type in valid_types:
                    filename = f"{symbol_clean}_{data_type}_{interval}_{date_str}.csv"
                    filepath = data_dirs[data_type] / filename
                    
                    # Check if file exists
                    if not force_redownload and self._is_valid_file(filepath):
                        record_count = len(pd.read_csv(filepath))
                        print(f"   ⏭️  {data_type}: Skip existing ({record_count} records)")
                        stats[data_type]['skipped'] += 1
                        continue
                    
                    # Download data
                    try:
                        df = self.download_single_data_type(
                            symbol=symbol,
                            data_type=data_type,
                            interval=interval,
                            start_date=date_str,
                            end_date=date_str,
                            exchange=exchange
                        )
                        
                        if df is not None and len(df) > 0:
                            df.to_csv(filepath, index=True)
                            print(f"   💾 {data_type}: Saved {len(df)} records")
                            stats[data_type]['success'] += 1
                        else:
                            print(f"   ⚠️  {data_type}: No data received")
                            stats[data_type]['failed'] += 1
                            
                    except Exception as e:
                        print(f"   ❌ {data_type}: Failed - {str(e)[:50]}...")
                        stats[data_type]['failed'] += 1
                    
                    # Rate limiting between data type requests
                    time.sleep(0.2)
                
                current_date += timedelta(days=1)
                time.sleep(0.1)  # Brief pause between days
            
            # Print summary
            self._print_download_summary(stats, total_days, valid_types)
            
            # Return success if we have some data for each type
            return all(stats[dt]['success'] + stats[dt]['skipped'] > 0 for dt in valid_types)
            
        except Exception as e:
            self.logger.error(f"Batch download failed: {e}")
            return False
    
    def _is_valid_file(self, filepath: Path) -> bool:
        """Check if file exists and is valid"""
        try:
            if not filepath.exists():
                return False
            
            if filepath.stat().st_size < 100:  # Too small
                return False
            
            # Try to read file
            df = pd.read_csv(filepath)
            return len(df) > 0
            
        except Exception:
            return False
    
    def _print_download_summary(self, stats: Dict, total_days: int, data_types: List[str]):
        """Print download summary"""
        print(f"\n📊 Download Summary:")
        print(f"   Total days: {total_days}")
        
        for data_type in data_types:
            s = stats[data_type]
            total_available = s['success'] + s['skipped']
            coverage = (total_available / total_days) * 100 if total_days > 0 else 0
            
            print(f"   📈 {data_type.upper()}:")
            print(f"      Downloaded: {s['success']}")
            print(f"      Skipped: {s['skipped']}")
            print(f"      Failed: {s['failed']}")
            print(f"      Coverage: {coverage:.1f}%")
    
    def combine_daily_files(self, symbol: str, interval: str, data_type: str,
                           start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Combine daily files into a single DataFrame
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            data_type: Type of data to combine
            start_date: Start date (optional, uses all available if not specified)
            end_date: End date (optional, uses all available if not specified)
            
        Returns:
            Combined DataFrame or None if no data found
        """
        symbol_clean = symbol.replace('/', '_').replace(':', '_')
        data_dir = self.data_dir / symbol_clean / interval / data_type
        
        if not data_dir.exists():
            self.logger.error(f"Data directory not found: {data_dir}")
            return None
        
        # Get all CSV files
        pattern = f"{symbol_clean}_{data_type}_{interval}_*.csv"
        files = list(data_dir.glob(pattern))
        
        if not files:
            self.logger.warning(f"No data files found for {symbol} {data_type}")
            return None
        
        # Filter by date range if specified
        if start_date or end_date:
            filtered_files = []
            for file in files:
                # Extract date from filename
                parts = file.stem.split('_')
                if len(parts) >= 4:
                    file_date = parts[-1]  # Last part should be date
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    filtered_files.append(file)
            files = filtered_files
        
        if not files:
            self.logger.warning(f"No files found in date range {start_date} to {end_date}")
            return None
        
        # Read and combine files
        dataframes = []
        for file in sorted(files):
            try:
                df = pd.read_csv(file, index_col=0, parse_dates=True)
                if len(df) > 0:
                    dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"Failed to read file {file}: {e}")
                continue
        
        if not dataframes:
            self.logger.error("No valid data files found")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, axis=0)
        combined_df = combined_df.sort_index()
        
        # Remove duplicates
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        self.logger.info(f"Combined {len(files)} files into DataFrame with {len(combined_df)} records")
        return combined_df
    
    def get_available_data_summary(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Get summary of available data for a symbol"""
        symbol_clean = symbol.replace('/', '_').replace(':', '_')
        base_dir = self.data_dir / symbol_clean / interval
        
        summary = {
            'symbol': symbol,
            'interval': interval,
            'data_types': {}
        }
        
        for data_type in self.data_configs.keys():
            data_dir = base_dir / data_type
            summary['data_types'][data_type] = {
                'available': data_dir.exists(),
                'file_count': 0,
                'date_range': None
            }
            
            if data_dir.exists():
                pattern = f"{symbol_clean}_{data_type}_{interval}_*.csv"
                files = list(data_dir.glob(pattern))
                summary['data_types'][data_type]['file_count'] = len(files)
                
                if files:
                    # Extract dates from filenames
                    dates = []
                    for file in files:
                        parts = file.stem.split('_')
                        if len(parts) >= 4:
                            dates.append(parts[-1])
                    
                    if dates:
                        dates.sort()
                        summary['data_types'][data_type]['date_range'] = {
                            'start': dates[0],
                            'end': dates[-1]
                        }
        
        return summary
    
    def merge_multi_data_types(self, symbol: str, interval: str, 
                              start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Merge multiple data types into a single DataFrame
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Merged DataFrame with all data types
        """
        # Get OHLCV data as base
        ohlcv_df = self.combine_daily_files(symbol, interval, 'ohlcv', start_date, end_date)
        
        if ohlcv_df is None:
            self.logger.error("No OHLCV data found to use as base")
            return None
        
        merged_df = ohlcv_df.copy()
        
        # Add funding rates if available
        funding_df = self.combine_daily_files(symbol, interval, 'funding_rates', start_date, end_date)
        if funding_df is not None:
            # Resample funding rates to match OHLCV frequency
            funding_resampled = funding_df.resample(f"{interval}").ffill()
            merged_df = merged_df.join(funding_resampled, how='left')
            self.logger.info("Added funding rates to merged data")
        
        self.logger.info(f"Merged dataset contains {len(merged_df)} records with {len(merged_df.columns)} columns")
        return merged_df


# Convenience functions for easy usage
def download_crypto_data(symbol: str, interval: str = '15m', days: int = 30,
                        data_types: List[str] = None) -> bool:
    """
    Simple function to download recent crypto data
    
    Args:
        symbol: Crypto symbol (e.g., 'BTCUSDT')
        interval: Time interval ('15m', '1h', '4h', '1d')
        days: Number of days to download
        data_types: List of data types to download
        
    Returns:
        Success status
    """
    if data_types is None:
        data_types = ['ohlcv', 'funding_rates']
    
    downloader = CoinAPIDownloader()
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    return downloader.download_daily_data_batch(
        symbol=symbol,
        interval=interval,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        data_types=data_types
    )


def load_combined_data(symbol: str, interval: str = '15m', days: int = None) -> Optional[pd.DataFrame]:
    """
    Load and combine all available data for a symbol
    
    Args:
        symbol: Crypto symbol
        interval: Time interval
        days: Number of recent days to load (optional)
        
    Returns:
        Combined DataFrame with all data types
    """
    downloader = CoinAPIDownloader()
    
    start_date = None
    end_date = None
    
    if days:
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    
    return downloader.merge_multi_data_types(symbol, interval, start_date, end_date)


if __name__ == "__main__":
    # Example usage
    print("🚀 CoinAPI Multi-Data Downloader")
    print("=" * 50)
    
    # Download recent data for BTC
    success = download_crypto_data(
        symbol='BTCUSDT',
        interval='15m',
        days=7,
        data_types=['ohlcv', 'funding_rates']
    )
    
    if success:
        print("✅ Download completed successfully")
        
        # Load and display summary
        df = load_combined_data('BTCUSDT', '15m', days=7)
        if df is not None:
            print(f"📊 Loaded {len(df)} records with columns: {list(df.columns)}")
            print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        else:
            print("❌ Failed to load combined data")
    else:
        print("❌ Download failed")
