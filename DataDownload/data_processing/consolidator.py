#!/usr/bin/env python3
"""
Data Consolidator for BinanceFutureP2

This module consolidates daily CSV files into a single file for a given date range.
It verifies data integrity, sanitizes data, and reports missing files.

Author: AI Assistant
Date: June 16, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import logging
import argparse
import sys
import os
from dataclasses import dataclass
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class ConsolidationReport:
    """Report of the consolidation process"""
    symbol: str
    interval: str
    start_date: str
    end_date: str
    total_days_requested: int
    found_files: int
    missing_files: int
    missing_dates: List[str]
    total_records: int
    duplicated_records: int
    invalid_records: int
    output_file: str
    success: bool
    errors: List[str]

class DataConsolidator:
    """
    Consolidates daily CSV files into a single file with data validation and sanitization.
    """
    
    def __init__(self, base_data_dir: str = "data/coinapi", output_dir: str = "data/processed"):
        """
        Initialize the DataConsolidator.
        
        Args:
            base_data_dir: Base directory containing the daily files
            output_dir: Directory to save consolidated files
        """
        self.base_data_dir = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Expected columns for OHLCV data
        self.expected_columns = {
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        }
        
        # Column name mappings (handle different naming conventions)
        self.column_mappings = {
            'time_period_start': 'timestamp',
            'time_period_end': 'timestamp_end',
            'time_open': 'timestamp',
            'time_close': 'timestamp_end',
            'price_open': 'open',
            'price_high': 'high',
            'price_low': 'low',
            'price_close': 'close',
            'volume_traded': 'volume',
            'trades_count': 'trades'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # File handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def consolidate_files(self, symbol: str, interval: str, start_date: str, end_date: str,
                         force_overwrite: bool = False, validate_data: bool = True) -> ConsolidationReport:
        """
        Consolidate daily files for the given date range.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '15m')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_overwrite: Whether to overwrite existing consolidated file
            validate_data: Whether to perform data validation and sanitization
            
        Returns:
            ConsolidationReport: Detailed report of the consolidation process
        """
        self.logger.info(f"🚀 Starting consolidation for {symbol} {interval} from {start_date} to {end_date}")
        
        # Initialize report
        report = ConsolidationReport(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            total_days_requested=0,
            found_files=0,
            missing_files=0,
            missing_dates=[],
            total_records=0,
            duplicated_records=0,
            invalid_records=0,
            output_file="",
            success=False,
            errors=[]
        )
        
        try:
            # Step 1: Generate date range and check for missing files
            date_range, missing_dates = self._check_file_availability(symbol, interval, start_date, end_date)
            
            report.total_days_requested = len(date_range)
            report.missing_dates = missing_dates
            report.missing_files = len(missing_dates)
            report.found_files = len(date_range) - len(missing_dates)
            
            # Error out if any files are missing
            if missing_dates:
                error_msg = f"❌ Missing {len(missing_dates)} daily files: {missing_dates[:10]}{'...' if len(missing_dates) > 10 else ''}"
                self.logger.error(error_msg)
                report.errors.append(error_msg)
                return report
            
            # Step 2: Check if consolidated file already exists
            output_file = self._get_output_filename(symbol, interval, start_date, end_date)
            report.output_file = str(output_file)
            
            if output_file.exists() and not force_overwrite:
                self.logger.info(f"✅ Consolidated file already exists: {output_file}")
                # Verify existing file
                if self._verify_consolidated_file(output_file):
                    report.success = True
                    return report
                else:
                    self.logger.warning("📝 Existing file failed verification, regenerating...")
            
            # Step 3: Load and consolidate files
            self.logger.info(f"📂 Loading {len(date_range)} daily files...")
            consolidated_df = self._load_and_combine_files(symbol, interval, date_range)
            
            if consolidated_df is None or consolidated_df.empty:
                error_msg = "❌ Failed to load any data from daily files"
                self.logger.error(error_msg)
                report.errors.append(error_msg)
                return report
            
            report.total_records = len(consolidated_df)
            self.logger.info(f"📊 Loaded {report.total_records} total records")
            
            # Step 4: Data validation and sanitization
            if validate_data:
                self.logger.info("🔍 Validating and sanitizing data...")
                consolidated_df, validation_report = self._validate_and_sanitize_data(consolidated_df)
                report.duplicated_records = validation_report.get('duplicates_removed', 0)
                report.invalid_records = validation_report.get('invalid_records_removed', 0)
                
                if consolidated_df.empty:
                    error_msg = "❌ No valid data remaining after sanitization"
                    self.logger.error(error_msg)
                    report.errors.append(error_msg)
                    return report
            
            # Step 5: Save consolidated file
            self.logger.info(f"💾 Saving consolidated file to {output_file}")
            consolidated_df.to_csv(output_file, index=True)
            
            # Step 6: Verify saved file
            if self._verify_consolidated_file(output_file):
                self.logger.info(f"✅ Successfully created consolidated file: {output_file}")
                self.logger.info(f"📈 Final dataset: {len(consolidated_df)} records from {report.found_files} files")
                report.success = True
            else:
                error_msg = "❌ Consolidated file failed verification"
                self.logger.error(error_msg)
                report.errors.append(error_msg)
            
        except Exception as e:
            error_msg = f"❌ Consolidation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            report.errors.append(error_msg)
        
        return report
    
    def _check_file_availability(self, symbol: str, interval: str, start_date: str, end_date: str) -> Tuple[List[str], List[str]]:
        """Check which daily files are available and which are missing"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_range = []
        missing_dates = []
        
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            date_range.append(date_str)
            
            # Check if file exists
            file_path = self._get_daily_file_path(symbol, interval, date_str)
            if not file_path.exists():
                missing_dates.append(date_str)
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"📊 Date range check: {len(date_range)} total days, {len(missing_dates)} missing")
        return date_range, missing_dates
    
    def _get_daily_file_path(self, symbol: str, interval: str, date: str) -> Path:
        """Get the path to a daily file"""
        # Try different directory structures
        candidates = [
            self.base_data_dir / symbol / interval / f"{symbol}_{interval}_{date}.csv",
            self.base_data_dir / f"{symbol}_{interval}_{date}.csv",
            self.base_data_dir / symbol / f"{symbol}_{interval}_{date}.csv"
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # Return the most likely path (even if it doesn't exist)
        return candidates[0]
    
    def _get_output_filename(self, symbol: str, interval: str, start_date: str, end_date: str) -> Path:
        """Generate output filename for consolidated file"""
        filename = f"{symbol}_{interval}_{start_date}_to_{end_date}_consolidated.csv"
        return self.output_dir / filename
    
    def _load_and_combine_files(self, symbol: str, interval: str, date_range: List[str]) -> Optional[pd.DataFrame]:
        """Load and combine daily files into a single DataFrame"""
        dataframes = []
        
        for date in date_range:
            file_path = self._get_daily_file_path(symbol, interval, date)
            
            try:
                # Try different ways to read the CSV
                df = None
                
                # Method 1: Standard read
                try:
                    df = pd.read_csv(file_path, parse_dates=True)
                except Exception:
                    pass
                
                # Method 2: Read with index_col=0
                if df is None:
                    try:
                        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    except Exception:
                        pass
                
                # Method 3: Read without parsing dates first
                if df is None:
                    df = pd.read_csv(file_path)
                
                if df is not None and not df.empty:
                    # Standardize column names
                    df = self._standardize_columns(df)
                    dataframes.append(df)
                    self.logger.debug(f"📄 Loaded {len(df)} records from {file_path.name}")
                else:
                    self.logger.warning(f"⚠️ Empty or invalid file: {file_path}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to load {file_path}: {str(e)}")
                raise
        
        if not dataframes:
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"🔗 Combined {len(dataframes)} files into {len(combined_df)} records")
        
        return combined_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using mappings"""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Convert all column names to lowercase
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Apply column mappings
        df = df.rename(columns=self.column_mappings)
        
        # Ensure timestamp column exists and is properly formatted
        timestamp_cols = ['timestamp', 'time_period_start', 'time_open', 'datetime']
        timestamp_col = None
        
        for col in timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col and timestamp_col != 'timestamp':
            df = df.rename(columns={timestamp_col: 'timestamp'})
        
        return df
    
    def _validate_and_sanitize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate and sanitize the consolidated data"""
        original_count = len(df)
        validation_report = {}
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        validation_report['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            self.logger.info(f"🧹 Removed {duplicates_removed} duplicate records")
        
        # Validate required columns
        missing_cols = self.expected_columns - set(df.columns)
        if missing_cols:
            self.logger.warning(f"⚠️ Missing columns: {missing_cols}")
        
        # Remove rows with invalid OHLCV data
        initial_count = len(df)
        
        # Remove rows where OHLC values are invalid
        ohlc_cols = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_cols if col in df.columns]
        
        if available_ohlc:
            # Remove rows with non-positive prices
            for col in available_ohlc:
                df = df[df[col] > 0]
            
            # Remove rows where high < low (impossible)
            if 'high' in df.columns and 'low' in df.columns:
                df = df[df['high'] >= df['low']]
            
            # Remove rows where open/close are outside high/low range
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                df = df[
                    (df['open'] >= df['low']) & 
                    (df['open'] <= df['high']) &
                    (df['close'] >= df['low']) & 
                    (df['close'] <= df['high'])
                ]
        
        # Remove rows with invalid volume
        if 'volume' in df.columns:
            df = df[df['volume'] >= 0]
        
        invalid_records_removed = initial_count - len(df)
        validation_report['invalid_records_removed'] = invalid_records_removed
        
        if invalid_records_removed > 0:
            self.logger.info(f"🧹 Removed {invalid_records_removed} invalid records")
        
        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            df = df.reset_index(drop=True)
        
        final_count = len(df)
        validation_report['final_count'] = final_count
        validation_report['data_quality'] = final_count / original_count if original_count > 0 else 0
        
        self.logger.info(f"✅ Data validation complete: {final_count}/{original_count} records retained ({validation_report['data_quality']:.1%})")
        
        return df, validation_report
    
    def _verify_consolidated_file(self, file_path: Path) -> bool:
        """Verify that the consolidated file is valid"""
        try:
            df = pd.read_csv(file_path, index_col=0, nrows=10)  # Just check first 10 rows
            if df.empty:
                return False
            
            # Check that it has data
            if len(df) == 0:
                return False
            
            self.logger.debug(f"✅ Consolidated file verification passed: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Consolidated file verification failed: {str(e)}")
            return False

def main():
    """Command-line interface for the data consolidator"""
    parser = argparse.ArgumentParser(description="Consolidate daily CSV files into a single file")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("--interval", required=True, help="Time interval (e.g., 15m)")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--base-data-dir", default="data/coinapi", help="Base directory containing daily files")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for consolidated files")
    parser.add_argument("--force-overwrite", action="store_true", help="Overwrite existing consolidated file")
    parser.add_argument("--no-validation", action="store_true", help="Skip data validation and sanitization")
    parser.add_argument("--report-file", help="Save consolidation report to JSON file")
    
    args = parser.parse_args()
    
    # Create consolidator
    consolidator = DataConsolidator(
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir
    )
    
    # Run consolidation
    report = consolidator.consolidate_files(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        force_overwrite=args.force_overwrite,
        validate_data=not args.no_validation
    )
    
    # Print summary
    print("\n" + "="*80)
    print("📊 CONSOLIDATION REPORT")
    print("="*80)
    print(f"Symbol: {report.symbol}")
    print(f"Interval: {report.interval}")
    print(f"Date Range: {report.start_date} to {report.end_date}")
    print(f"Total Days: {report.total_days_requested}")
    print(f"Found Files: {report.found_files}")
    print(f"Missing Files: {report.missing_files}")
    if report.missing_dates:
        print(f"Missing Dates: {report.missing_dates[:5]}{'...' if len(report.missing_dates) > 5 else ''}")
    print(f"Total Records: {report.total_records}")
    print(f"Duplicates Removed: {report.duplicated_records}")
    print(f"Invalid Records Removed: {report.invalid_records}")
    print(f"Output File: {report.output_file}")
    print(f"Success: {'✅' if report.success else '❌'}")
    if report.errors:
        print(f"Errors: {report.errors}")
    print("="*80)
    
    # Save report to file if requested
    if args.report_file:
        report_data = {
            'symbol': report.symbol,
            'interval': report.interval,
            'start_date': report.start_date,
            'end_date': report.end_date,
            'total_days_requested': report.total_days_requested,
            'found_files': report.found_files,
            'missing_files': report.missing_files,
            'missing_dates': report.missing_dates,
            'total_records': report.total_records,
            'duplicated_records': report.duplicated_records,
            'invalid_records': report.invalid_records,
            'output_file': report.output_file,
            'success': report.success,
            'errors': report.errors,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(args.report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"📄 Report saved to: {args.report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if report.success else 1)

if __name__ == "__main__":
    main()
