#!/usr/bin/env python3
"""
Data Validator for BinanceFutureP2

This module provides data validation and quality checking utilities.

Author: AI Assistant
Date: June 16, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class ValidationReport:
    """Report of data validation process"""
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    missing_values: Dict[str, int]
    data_quality_score: float
    issues: List[str]
    recommendations: List[str]

class DataValidator:
    """
    Validates trading data quality and integrity.
    """
    
    def __init__(self):
        """Initialize the DataValidator"""
        self.logger = logging.getLogger(__name__)
        
        # Expected columns for OHLCV data
        self.required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        self.optional_columns = {'trades', 'vwap', 'timestamp_end'}
        
        # Data quality thresholds
        self.quality_thresholds = {
            'min_price': 0.0001,  # Minimum valid price
            'max_price_change': 0.5,  # Maximum price change between consecutive records (50%)
            'min_volume': 0,  # Minimum valid volume
            'max_gap_minutes': 60  # Maximum gap between timestamps in minutes
        }
    
    def validate_dataframe(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> ValidationReport:
        """
        Comprehensive validation of a trading data DataFrame.
        
        Args:
            df: DataFrame to validate
            symbol: Trading symbol for context
            
        Returns:
            ValidationReport: Detailed validation report
        """
        self.logger.info(f"🔍 Starting validation for {symbol} data ({len(df)} records)")
        
        original_count = len(df)
        issues = []
        recommendations = []
        
        # Check basic structure
        structure_issues = self._validate_structure(df)
        issues.extend(structure_issues)
        
        # Check data types
        dtype_issues = self._validate_data_types(df)
        issues.extend(dtype_issues)
        
        # Check for missing values
        missing_values = self._check_missing_values(df)
        
        # Check for duplicates
        duplicate_count = self._count_duplicates(df)
        
        # Validate OHLCV data
        ohlcv_issues = self._validate_ohlcv_data(df)
        issues.extend(ohlcv_issues)
        
        # Check timestamp continuity
        timestamp_issues = self._validate_timestamps(df)
        issues.extend(timestamp_issues)
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(df, len(issues), duplicate_count, missing_values)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df, issues, quality_score)
        
        # Count valid records (records without critical issues)
        valid_records = self._count_valid_records(df)
        
        report = ValidationReport(
            total_records=original_count,
            valid_records=valid_records,
            invalid_records=original_count - valid_records,
            duplicate_records=duplicate_count,
            missing_values=missing_values,
            data_quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )
        
        self._log_validation_summary(report, symbol)
        return report
    
    def _validate_structure(self, df: pd.DataFrame) -> List[str]:
        """Validate DataFrame structure"""
        issues = []
        
        if df.empty:
            issues.append("DataFrame is empty")
            return issues
        
        # Check required columns
        missing_required = self.required_columns - set(df.columns)
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")
        
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            issues.append(f"Found unnamed columns: {unnamed_cols}")
        
        return issues
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """Validate data types"""
        issues = []
        
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except (ValueError, TypeError):
                        issues.append(f"Column '{col}' contains non-numeric data")
        
        # Check timestamp column
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    pd.to_datetime(df['timestamp'], errors='raise')
                except (ValueError, TypeError):
                    issues.append("Timestamp column contains invalid datetime data")
        
        return issues
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for missing values in each column"""
        missing_values = {}
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                missing_values[col] = int(null_count)
        
        return missing_values
    
    def _count_duplicates(self, df: pd.DataFrame) -> int:
        """Count duplicate records"""
        return int(df.duplicated().sum())
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> List[str]:
        """Validate OHLCV data integrity"""
        issues = []
        
        ohlc_cols = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_cols if col in df.columns]
        
        if len(available_ohlc) < 4:
            issues.append(f"Incomplete OHLC data: only {available_ohlc} available")
            return issues
        
        # Check for negative or zero prices
        for col in available_ohlc:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                issues.append(f"Found {negative_count} non-positive values in {col}")
        
        # Check OHLC relationships
        high_low_issues = (df['high'] < df['low']).sum()
        if high_low_issues > 0:
            issues.append(f"Found {high_low_issues} records where high < low")
        
        open_range_issues = ((df['open'] < df['low']) | (df['open'] > df['high'])).sum()
        if open_range_issues > 0:
            issues.append(f"Found {open_range_issues} records where open is outside high-low range")
        
        close_range_issues = ((df['close'] < df['low']) | (df['close'] > df['high'])).sum()
        if close_range_issues > 0:
            issues.append(f"Found {close_range_issues} records where close is outside high-low range")
        
        # Check for extreme price changes
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > self.quality_thresholds['max_price_change']).sum()
            if extreme_changes > 0:
                issues.append(f"Found {extreme_changes} records with extreme price changes (>{self.quality_thresholds['max_price_change']:.0%})")
        
        # Check volume
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Found {negative_volume} records with negative volume")
        
        return issues
    
    def _validate_timestamps(self, df: pd.DataFrame) -> List[str]:
        """Validate timestamp data"""
        issues = []
        
        if 'timestamp' not in df.columns:
            issues.append("No timestamp column found")
            return issues
        
        # Convert to datetime if not already
        try:
            timestamps = pd.to_datetime(df['timestamp'])
        except Exception as e:
            issues.append(f"Cannot parse timestamps: {str(e)}")
            return issues
        
        # Check for duplicate timestamps
        duplicate_timestamps = timestamps.duplicated().sum()
        if duplicate_timestamps > 0:
            issues.append(f"Found {duplicate_timestamps} duplicate timestamps")
        
        # Check chronological order
        if not timestamps.is_monotonic_increasing:
            issues.append("Timestamps are not in chronological order")
        
        # Check for large gaps
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dt.total_seconds() / 60  # Convert to minutes
            large_gaps = (time_diffs > self.quality_thresholds['max_gap_minutes']).sum()
            if large_gaps > 0:
                max_gap = time_diffs.max()
                issues.append(f"Found {large_gaps} large time gaps (max: {max_gap:.1f} minutes)")
        
        return issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, issue_count: int, 
                                duplicate_count: int, missing_values: Dict[str, int]) -> float:
        """Calculate overall data quality score (0-1)"""
        if df.empty:
            return 0.0
        
        total_records = len(df)
        total_cells = total_records * len(df.columns)
        
        # Penalties
        issue_penalty = min(issue_count * 0.1, 0.5)  # Max 50% penalty for issues
        duplicate_penalty = min(duplicate_count / total_records * 0.3, 0.3)  # Max 30% penalty for duplicates
        
        total_missing = sum(missing_values.values())
        missing_penalty = min(total_missing / total_cells * 0.2, 0.2)  # Max 20% penalty for missing values
        
        # Calculate score
        score = max(1.0 - issue_penalty - duplicate_penalty - missing_penalty, 0.0)
        return round(score, 3)
    
    def _count_valid_records(self, df: pd.DataFrame) -> int:
        """Count records that pass basic validation"""
        if df.empty:
            return 0
        
        valid_mask = pd.Series([True] * len(df))
        
        # Check OHLCV validity
        ohlc_cols = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_cols if col in df.columns]
        
        if len(available_ohlc) >= 4:
            # Remove records with invalid OHLC relationships
            valid_mask &= (df['high'] >= df['low'])
            valid_mask &= (df['open'] >= df['low']) & (df['open'] <= df['high'])
            valid_mask &= (df['close'] >= df['low']) & (df['close'] <= df['high'])
            
            # Remove records with non-positive prices
            for col in available_ohlc:
                valid_mask &= (df[col] > 0)
        
        # Check volume
        if 'volume' in df.columns:
            valid_mask &= (df['volume'] >= 0)
        
        return int(valid_mask.sum())
    
    def _generate_recommendations(self, df: pd.DataFrame, issues: List[str], 
                                 quality_score: float) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Data quality is poor - consider re-downloading or using alternative sources")
        elif quality_score < 0.9:
            recommendations.append("Data quality is moderate - review and clean data before use")
        
        if any("duplicate" in issue.lower() for issue in issues):
            recommendations.append("Remove duplicate records before processing")
        
        if any("missing" in issue.lower() or "null" in issue.lower() for issue in issues):
            recommendations.append("Handle missing values through interpolation or removal")
        
        if any("timestamp" in issue.lower() for issue in issues):
            recommendations.append("Fix timestamp issues and ensure chronological order")
        
        if any("extreme" in issue.lower() for issue in issues):
            recommendations.append("Investigate extreme price changes - may indicate data errors")
        
        if len(df) < 100:
            recommendations.append("Dataset is very small - consider extending date range")
        
        return recommendations
    
    def _log_validation_summary(self, report: ValidationReport, symbol: str):
        """Log validation summary"""
        self.logger.info(f"📊 Validation complete for {symbol}:")
        self.logger.info(f"   Quality Score: {report.data_quality_score:.1%}")
        self.logger.info(f"   Valid Records: {report.valid_records}/{report.total_records}")
        
        if report.issues:
            self.logger.warning(f"   Issues Found: {len(report.issues)}")
            for issue in report.issues[:5]:  # Log first 5 issues
                self.logger.warning(f"     - {issue}")
            if len(report.issues) > 5:
                self.logger.warning(f"     - ... and {len(report.issues) - 5} more issues")
        
        if report.recommendations:
            self.logger.info(f"   Recommendations: {len(report.recommendations)}")
            for rec in report.recommendations:
                self.logger.info(f"     - {rec}")

def validate_file(file_path: str, symbol: str = None) -> ValidationReport:
    """
    Convenience function to validate a CSV file.
    
    Args:
        file_path: Path to the CSV file
        symbol: Trading symbol for context
        
    Returns:
        ValidationReport: Validation results
    """
    if symbol is None:
        symbol = Path(file_path).stem
    
    # Load the file
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except Exception:
        df = pd.read_csv(file_path)
    
    # Validate
    validator = DataValidator()
    return validator.validate_dataframe(df, symbol)
