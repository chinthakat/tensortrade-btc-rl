"""
Trade Anomaly Analyzer

This script analyzes trade files against market data to identify price anomalies.
It compares entry_price and close_price from trades with actual market prices
at the corresponding timestamps (entry_datetime and close_datetime).

Author: Trade Analysis System
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Optional
import warnings

# Optional imports with fallbacks
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Charts will use basic matplotlib styling.")

warnings.filterwarnings('ignore')

class TradeAnomalyAnalyzer:
    def __init__(self, market_data_path: str):
        """
        Initialize the Trade Anomaly Analyzer
        
        Args:
            market_data_path: Path to the market data CSV file
        """
        self.market_data_path = market_data_path
        self.market_data = None
        self.trade_data = None
        self.analysis_results = {}
        self.anomalies = []
        
    def load_market_data(self) -> pd.DataFrame:
        """Load and prepare market data"""
        print(f"Loading market data from: {self.market_data_path}")
        
        try:
            # Load market data
            self.market_data = pd.read_csv(self.market_data_path)
            
            # Convert timestamp to datetime
            if 'timestamp' in self.market_data.columns:
                self.market_data['datetime'] = pd.to_datetime(self.market_data['timestamp'], unit='s')
            else:
                # Assume the last column is timestamp
                timestamp_col = self.market_data.columns[-1]
                self.market_data['datetime'] = pd.to_datetime(self.market_data[timestamp_col], unit='s')
            
            # Sort by datetime
            self.market_data = self.market_data.sort_values('datetime')
            
            # Remove duplicate timestamps, keeping the first occurrence
            original_count = len(self.market_data)
            self.market_data = self.market_data.drop_duplicates(subset=['datetime'], keep='first')
            removed_count = original_count - len(self.market_data)
            
            if removed_count > 0:
                print(f"  - Removed {removed_count} duplicate timestamps")
            
            # Set datetime as index for fast lookups
            self.market_data.set_index('datetime', inplace=True)
            
            # Double-check for index uniqueness after setting index
            if not self.market_data.index.is_unique:
                print("  - Warning: Index still not unique after duplicate removal, forcing uniqueness...")
                self.market_data = self.market_data[~self.market_data.index.duplicated(keep='first')]
                print(f"  - Final record count: {len(self.market_data):,}")
            
            # Verify index is now unique
            if not self.market_data.index.is_unique:
                raise ValueError("Failed to create unique datetime index for market data")
            
            print(f"✓ Market data loaded successfully")
            print(f"  - Records: {len(self.market_data):,}")
            print(f"  - Date range: {self.market_data.index.min()} to {self.market_data.index.max()}")
            print(f"  - Columns: {list(self.market_data.columns)}")
            
            return self.market_data
            
        except Exception as e:
            print(f"✗ Error loading market data: {e}")
            raise
    
    def load_trade_data(self, trade_file_path: str) -> pd.DataFrame:
        """Load and prepare trade data"""
        print(f"\nLoading trade data from: {trade_file_path}")
        
        try:
            # Load trade data
            self.trade_data = pd.read_csv(trade_file_path)
            
            # Convert timestamps to datetime
            self.trade_data['entry_datetime_parsed'] = self.parse_datetime_column(
                self.trade_data['entry_datetime']
            )
            
            # Handle close_datetime (might be empty for open trades)
            if 'close_datetime' in self.trade_data.columns:
                self.trade_data['close_datetime_parsed'] = self.parse_datetime_column(
                    self.trade_data['close_datetime']
                )
            
            # Filter out invalid trades
            valid_trades = self.trade_data['entry_datetime_parsed'].notna()
            self.trade_data = self.trade_data[valid_trades]
            
            print(f"✓ Trade data loaded successfully")
            print(f"  - Total trades: {len(self.trade_data):,}")
            print(f"  - Closed trades: {self.trade_data['close_datetime_parsed'].notna().sum():,}")
            print(f"  - Open trades: {self.trade_data['close_datetime_parsed'].isna().sum():,}")
            
            return self.trade_data
            
        except Exception as e:
            print(f"✗ Error loading trade data: {e}")
            raise
    
    def parse_datetime_column(self, datetime_series: pd.Series) -> pd.Series:
        """Parse datetime column handling both Unix timestamps and date strings"""
        parsed_dates = []
        
        for value in datetime_series:
            if pd.isna(value) or value == '':
                parsed_dates.append(pd.NaT)
                continue
            
            try:
                # Try as Unix timestamp first
                if isinstance(value, (int, float)) or str(value).replace('.', '').isdigit():
                    timestamp = float(value)
                    # Handle both seconds and milliseconds
                    if timestamp > 1e10:  # Likely milliseconds
                        timestamp = timestamp / 1000
                    parsed_dates.append(pd.to_datetime(timestamp, unit='s'))
                else:
                    # Try as date string
                    parsed_dates.append(pd.to_datetime(value))
            except:
                parsed_dates.append(pd.NaT)
        
        return pd.Series(parsed_dates)
    
    def get_market_price_at_time(self, target_time: pd.Timestamp, price_type: str = 'close') -> Optional[float]:
        """
        Get market price at a specific time using nearest available data
        
        Args:
            target_time: Target timestamp
            price_type: Type of price ('open', 'high', 'low', 'close')
        
        Returns:
            Market price at the target time or None if not found
        """
        if target_time is pd.NaT or self.market_data is None:
            return None
        
        try:
            # Find the nearest market data point
            nearest_idx = self.market_data.index.get_indexer([target_time], method='nearest')[0]
            
            if nearest_idx >= 0:
                nearest_time = self.market_data.index[nearest_idx]
                time_diff = abs((target_time - nearest_time).total_seconds())
                
                # Only accept if within 15 minutes (900 seconds)
                if time_diff <= 900:
                    return self.market_data.iloc[nearest_idx][price_type]
            
            return None
            
        except Exception as e:
            print(f"Error getting market price: {e}")
            return None
    
    def analyze_trade_anomalies(self, tolerance_pct: float = 1.0) -> Dict:
        """
        Analyze trades for price anomalies
        
        Args:
            tolerance_pct: Acceptable percentage difference between trade and market price
        
        Returns:
            Dictionary containing analysis results
        """
        print(f"\nAnalyzing trade anomalies (tolerance: ±{tolerance_pct}%)")
        
        if self.trade_data is None or self.market_data is None:
            raise ValueError("Both trade and market data must be loaded first")
        
        anomalies = []
        analysis_stats = {
            'total_trades': len(self.trade_data),
            'entry_anomalies': 0,
            'close_anomalies': 0,
            'entry_price_differences': [],
            'close_price_differences': [],
            'trades_without_market_data': 0
        }
        
        for idx, trade in self.trade_data.iterrows():
            trade_analysis = {
                'trade_id': trade.get('trade_id', idx),
                'entry_datetime': trade['entry_datetime_parsed'],
                'close_datetime': trade.get('close_datetime_parsed'),
                'trade_entry_price': trade.get('entry_price'),
                'trade_close_price': trade.get('close_price'),
                'side': trade.get('side', 'UNKNOWN'),
                'status': trade.get('status', 'UNKNOWN'),
                'anomalies': []
            }
            
            # Check entry price anomaly
            if pd.notna(trade['entry_datetime_parsed']) and pd.notna(trade.get('entry_price')):
                market_price = self.get_market_price_at_time(trade['entry_datetime_parsed'], 'close')
                
                if market_price is not None:
                    trade_analysis['market_entry_price'] = market_price
                    price_diff_pct = abs(trade['entry_price'] - market_price) / market_price * 100
                    trade_analysis['entry_price_diff_pct'] = price_diff_pct
                    analysis_stats['entry_price_differences'].append(price_diff_pct)
                    
                    if price_diff_pct > tolerance_pct:
                        analysis_stats['entry_anomalies'] += 1
                        trade_analysis['anomalies'].append({
                            'type': 'ENTRY_PRICE_ANOMALY',
                            'trade_price': trade['entry_price'],
                            'market_price': market_price,
                            'difference_pct': price_diff_pct,
                            'timestamp': trade['entry_datetime_parsed']
                        })
                else:
                    analysis_stats['trades_without_market_data'] += 1
                    trade_analysis['anomalies'].append({
                        'type': 'NO_MARKET_DATA_AT_ENTRY',
                        'timestamp': trade['entry_datetime_parsed']
                    })
            
            # Check close price anomaly (only for closed trades)
            if (pd.notna(trade.get('close_datetime_parsed')) and 
                pd.notna(trade.get('close_price')) and 
                trade.get('status') == 'CLOSED'):
                
                market_price = self.get_market_price_at_time(trade['close_datetime_parsed'], 'close')
                
                if market_price is not None:
                    trade_analysis['market_close_price'] = market_price
                    price_diff_pct = abs(trade['close_price'] - market_price) / market_price * 100
                    trade_analysis['close_price_diff_pct'] = price_diff_pct
                    analysis_stats['close_price_differences'].append(price_diff_pct)
                    
                    if price_diff_pct > tolerance_pct:
                        analysis_stats['close_anomalies'] += 1
                        trade_analysis['anomalies'].append({
                            'type': 'CLOSE_PRICE_ANOMALY',
                            'trade_price': trade['close_price'],
                            'market_price': market_price,
                            'difference_pct': price_diff_pct,
                            'timestamp': trade['close_datetime_parsed']
                        })
            
            # Store if there are anomalies
            if trade_analysis['anomalies']:
                anomalies.append(trade_analysis)
        
        self.anomalies = anomalies
        self.analysis_results = {
            'stats': analysis_stats,
            'anomalies': anomalies
        }
        
        print(f"✓ Analysis completed")
        print(f"  - Entry price anomalies: {analysis_stats['entry_anomalies']}")
        print(f"  - Close price anomalies: {analysis_stats['close_anomalies']}")
        print(f"  - Trades without market data: {analysis_stats['trades_without_market_data']}")
        
        return self.analysis_results
    
    def generate_report(self, output_dir: str = "trade_analysis_reports", trade_file_path: str = "") -> str:
        """Generate comprehensive analysis report"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"trade_anomaly_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRADE ANOMALY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Market Data: {self.market_data_path}\n")
            if trade_file_path:
                f.write(f"Trade File: {trade_file_path}\n")
            f.write("\n")
            
            # Summary Statistics
            stats = self.analysis_results['stats']
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Trades Analyzed: {stats['total_trades']:,}\n")
            f.write(f"Entry Price Anomalies: {stats['entry_anomalies']:,} ({stats['entry_anomalies']/stats['total_trades']*100:.2f}%)\n")
            f.write(f"Close Price Anomalies: {stats['close_anomalies']:,} ({stats['close_anomalies']/stats['total_trades']*100:.2f}%)\n")
            f.write(f"Trades Without Market Data: {stats['trades_without_market_data']:,}\n")
            f.write("\n")
            
            # Price Difference Statistics
            if stats['entry_price_differences']:
                entry_diffs = np.array(stats['entry_price_differences'])
                f.write("ENTRY PRICE DIFFERENCE STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Difference: {entry_diffs.mean():.4f}%\n")
                f.write(f"Median Difference: {np.median(entry_diffs):.4f}%\n")
                f.write(f"Max Difference: {entry_diffs.max():.4f}%\n")
                f.write(f"95th Percentile: {np.percentile(entry_diffs, 95):.4f}%\n")
                f.write("\n")
            
            if stats['close_price_differences']:
                close_diffs = np.array(stats['close_price_differences'])
                f.write("CLOSE PRICE DIFFERENCE STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Difference: {close_diffs.mean():.4f}%\n")
                f.write(f"Median Difference: {np.median(close_diffs):.4f}%\n")
                f.write(f"Max Difference: {close_diffs.max():.4f}%\n")
                f.write(f"95th Percentile: {np.percentile(close_diffs, 95):.4f}%\n")
                f.write("\n")
            
            # Detailed Anomalies
            f.write("DETAILED ANOMALY REPORT\n")
            f.write("-" * 40 + "\n")
            
            if not self.anomalies:
                f.write("No anomalies detected within the specified tolerance.\n")
            else:
                for i, trade in enumerate(self.anomalies, 1):
                    f.write(f"\nANOMALY #{i}\n")
                    f.write(f"Trade ID: {trade['trade_id']}\n")
                    f.write(f"Side: {trade['side']}\n")
                    f.write(f"Status: {trade['status']}\n")
                    f.write(f"Entry Time: {trade['entry_datetime']}\n")
                    
                    for anomaly in trade['anomalies']:
                        f.write(f"\n  {anomaly['type']}:\n")
                        if 'trade_price' in anomaly:
                            f.write(f"    Trade Price: ${anomaly['trade_price']:,.2f}\n")
                            f.write(f"    Market Price: ${anomaly['market_price']:,.2f}\n")
                            f.write(f"    Difference: {anomaly['difference_pct']:.4f}%\n")
                        f.write(f"    Timestamp: {anomaly['timestamp']}\n")
                    
                    f.write("\n" + "-" * 40)
        
        print(f"✓ Report generated: {report_file}")
        return report_file
    
    def create_visualizations(self, output_dir: str = "trade_analysis_reports"):
        """Create visualization charts for the analysis"""
        
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the plotting style
        plt.style.use('default')
        if HAS_SEABORN:
            sns.set_palette("husl")
        
        # Figure 1: Price Difference Distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trade Price Analysis - Anomaly Detection', fontsize=16, fontweight='bold')
        
        stats = self.analysis_results['stats']
        
        # Entry price differences histogram
        if stats['entry_price_differences']:
            axes[0, 0].hist(stats['entry_price_differences'], bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Entry Price Differences Distribution')
            axes[0, 0].set_xlabel('Percentage Difference (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(1.0, color='red', linestyle='--', label='1% Tolerance')
            axes[0, 0].legend()
        
        # Close price differences histogram
        if stats['close_price_differences']:
            axes[0, 1].hist(stats['close_price_differences'], bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Close Price Differences Distribution')
            axes[0, 1].set_xlabel('Percentage Difference (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(1.0, color='red', linestyle='--', label='1% Tolerance')
            axes[0, 1].legend()
        
        # Anomaly summary bar chart
        anomaly_counts = [stats['entry_anomalies'], stats['close_anomalies']]
        anomaly_labels = ['Entry Price\nAnomalies', 'Close Price\nAnomalies']
        bars = axes[1, 0].bar(anomaly_labels, anomaly_counts, color=['red', 'orange'], alpha=0.7)
        axes[1, 0].set_title('Anomaly Counts')
        axes[1, 0].set_ylabel('Number of Anomalies')
        
        # Add value labels on bars
        for bar, count in zip(bars, anomaly_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        # Trade status pie chart
        if self.trade_data is not None:
            status_counts = self.trade_data['status'].value_counts()
            axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Trade Status Distribution')
        
        plt.tight_layout()
        chart_file = os.path.join(output_dir, f"trade_analysis_charts_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Charts saved: {chart_file}")
        
        # Figure 2: Time Series of Anomalies (if any)
        if self.anomalies:
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # Plot market prices
            if self.market_data is not None:
                ax.plot(self.market_data.index, self.market_data['close'], 
                       alpha=0.7, color='gray', label='Market Price')
            
            # Plot anomalous trades
            entry_anomalies = []
            close_anomalies = []
            
            for trade in self.anomalies:
                for anomaly in trade['anomalies']:
                    if anomaly['type'] == 'ENTRY_PRICE_ANOMALY':
                        entry_anomalies.append((anomaly['timestamp'], anomaly['trade_price']))
                    elif anomaly['type'] == 'CLOSE_PRICE_ANOMALY':
                        close_anomalies.append((anomaly['timestamp'], anomaly['trade_price']))
            
            if entry_anomalies:
                times, prices = zip(*entry_anomalies)
                ax.scatter(times, prices, color='red', s=100, marker='^', 
                          label=f'Entry Anomalies ({len(entry_anomalies)})', zorder=5)
            
            if close_anomalies:
                times, prices = zip(*close_anomalies)
                ax.scatter(times, prices, color='orange', s=100, marker='v', 
                          label=f'Close Anomalies ({len(close_anomalies)})', zorder=5)
            
            ax.set_title('Trade Anomalies Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            timeseries_file = os.path.join(output_dir, f"anomaly_timeseries_{timestamp}.png")
            plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Time series chart saved: {timeseries_file}")


def scan_for_trade_files(episodes_dir: str = "episodes") -> List[str]:
    """Scan for available trade files"""
    trade_files = []
    
    # Look for trade files in episode directories
    pattern = os.path.join(episodes_dir, "episode_*", "logs", "trades_*.csv")
    trade_files.extend(glob.glob(pattern))
    
    # Look for test trade files
    pattern = os.path.join(episodes_dir, "test_*", "logs", "trades_*.csv")
    trade_files.extend(glob.glob(pattern))
    
    # Also look for trade files directly in data directory for legacy support
    pattern = os.path.join("data", "trades_*.csv")
    trade_files.extend(glob.glob(pattern))
    
    return sorted(trade_files)


def main():
    """Main function to run the trade anomaly analysis"""
    
    print("=" * 60)
    print("TRADE ANOMALY ANALYZER")
    print("=" * 60)
    
    # Default paths
    default_market_data = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
    
    # Check if market data exists
    if not os.path.exists(default_market_data):
        print(f"Market data file not found: {default_market_data}")
        print("Please ensure the market data file exists.")
        return
    
    # Scan for trade files
    trade_files = scan_for_trade_files()
    
    if not trade_files:
        print("No trade files found in the data directory.")
        print("Please ensure trade files exist in episode directories.")
        return
    
    # Display available trade files
    print(f"\nFound {len(trade_files)} trade file(s):")
    for i, file in enumerate(trade_files, 1):
        print(f"  {i}. {file}")
    
    # Prompt user to select trade file
    while True:
        try:
            choice = input(f"\nSelect trade file (1-{len(trade_files)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Analysis cancelled.")
                return
            
            choice = int(choice)
            if 1 <= choice <= len(trade_files):
                selected_trade_file = trade_files[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number or 'q'.")
    
    # Get tolerance setting
    while True:
        try:
            tolerance = input("\nEnter price difference tolerance percentage (default: 1.0): ").strip()
            if not tolerance:
                tolerance = 1.0
            else:
                tolerance = float(tolerance)
            break
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nStarting analysis...")
    print(f"Market Data: {default_market_data}")
    print(f"Trade File: {selected_trade_file}")
    print(f"Tolerance: ±{tolerance}%")
    
    # Run analysis
    try:
        analyzer = TradeAnomalyAnalyzer(default_market_data)
        
        # Load data
        analyzer.load_market_data()
        analyzer.load_trade_data(selected_trade_file)
        
        # Analyze anomalies
        results = analyzer.analyze_trade_anomalies(tolerance_pct=tolerance)
        
        # Generate reports
        report_file = analyzer.generate_report(trade_file_path=selected_trade_file)
        analyzer.create_visualizations()
        
        print(f"\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Report saved: {report_file}")
        print(f"Charts saved in: trade_analysis_reports/")
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
