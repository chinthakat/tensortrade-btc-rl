#!/usr/bin/env python3
"""
Trading Environment Debug Session
================================

Analyzes critical timeframes where price anomalies occur to understand
the root cause of price discrepancies between market data and trade execution.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
import sys
import os
import glob

# Add the parent directory to sys.path to import the trading environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv, PriceValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_session.log'),
        logging.StreamHandler()
    ]
)

class TradingEnvironmentDebugger:
    """Debug session for trading environment price validation"""
    
    def __init__(self, market_data_path: str, trade_file_path: str):
        """Initialize the debugger"""
        self.market_data_path = market_data_path
        self.trade_file_path = trade_file_path
        self.market_data = None
        self.trade_data = None
        self.env = None
        
    def load_data(self):
        """Load market and trade data"""
        print("Loading market data...")
        self.market_data = pd.read_csv(self.market_data_path)
        self.market_data['datetime'] = pd.to_datetime(self.market_data['timestamp'], unit='s')
        print(f"✓ Market data loaded: {len(self.market_data)} records")
        
        print("Loading trade data...")
        self.trade_data = pd.read_csv(self.trade_file_path)
        self.trade_data['entry_datetime_parsed'] = pd.to_datetime(
            self.trade_data['entry_datetime'], unit='s', errors='coerce'
        )
        print(f"✓ Trade data loaded: {len(self.trade_data)} records")
        
        print("Initializing trading environment...")
        self.env = FuturesTradingEnv(
            df=self.market_data,
            initial_equity=10000,
            window_size=10,
            log_file=None
        )
        print(f"✓ Environment initialized with {len(self.env.price_data)} price records")
        
    def find_critical_anomalies(self, max_anomalies: int = 10) -> list:
        """Find the most critical price anomalies for investigation"""
        print("\nAnalyzing trade data for critical anomalies...")
        
        anomalies = []
        validator = PriceValidator(tolerance_pct=1.0, enable_logging=False)
        
        for idx, trade in self.trade_data.iterrows():
            if pd.isna(trade.get('entry_price')) or trade.get('entry_price', 0) == 0:
                continue
                
            entry_time = trade['entry_datetime_parsed']
            if pd.isna(entry_time):
                continue
            
            # Find corresponding market data
            market_row = self.find_market_data_at_time(entry_time)
            if market_row is None:
                continue
            
            # Validate entry price
            is_valid, validation = validator.validate_price(
                trade_price=trade['entry_price'],
                market_price=market_row['close'],
                timestamp=entry_time,
                step=idx,
                context="ENTRY"
            )
            
            if not is_valid:
                anomalies.append({
                    'trade_idx': idx,
                    'trade_id': trade.get('trade_id', f'trade_{idx}'),
                    'entry_time': entry_time,
                    'trade_price': trade['entry_price'],
                    'market_price': market_row['close'],
                    'difference_pct': validation['difference_pct'],
                    'market_row_idx': market_row.name,
                    'validation': validation
                })
        
        # Sort by difference percentage (worst first)
        anomalies.sort(key=lambda x: x['difference_pct'], reverse=True)
        
        print(f"Found {len(anomalies)} entry price anomalies")
        return anomalies[:max_anomalies]
    
    def find_market_data_at_time(self, target_time: pd.Timestamp):
        """Find market data row closest to target time"""
        if target_time is pd.NaT:
            return None
            
        # Find closest timestamp
        time_diffs = abs(self.market_data['datetime'] - target_time)
        closest_idx = time_diffs.idxmin()
        
        # Only return if within 15 minutes
        if time_diffs.loc[closest_idx].total_seconds() <= 900:
            return self.market_data.loc[closest_idx]
        
        return None
    
    def debug_critical_timeframe(self, anomaly: dict):
        """Debug a specific critical timeframe"""
        print(f"\n{'='*80}")
        print(f"DEBUGGING CRITICAL ANOMALY - Trade {anomaly['trade_id']}")
        print(f"{'='*80}")
        
        # Basic anomaly information
        print(f"Trade Index: {anomaly['trade_idx']}")
        print(f"Entry Time: {anomaly['entry_time']}")
        print(f"Trade Price: ${anomaly['trade_price']:,.2f}")
        print(f"Market Price: ${anomaly['market_price']:,.2f}")
        print(f"Difference: {anomaly['difference_pct']:.2f}%")
        
        # Environment step mapping
        env_step = self.map_time_to_env_step(anomaly['entry_time'])
        if env_step is not None:
            print(f"\nEnvironment Step Mapping:")
            print(f"Market Row Index: {anomaly['market_row_idx']}")
            print(f"Environment Step: {env_step}")
            
            # Get environment state at this step
            env_state = self.get_env_state_at_step(env_step)
            if env_state:
                print(f"Environment Current Price: ${env_state['current_price']:,.2f}")
                print(f"Environment Timestamp: {env_state['timestamp']}")
                
        # Check surrounding market data
        self.analyze_surrounding_data(anomaly)
        
    def map_time_to_env_step(self, target_time: pd.Timestamp) -> int:
        """Map timestamp to environment step"""
        if not hasattr(self.env, 'price_data') or target_time is pd.NaT:
            return None
            
        # Find the step that corresponds to this timestamp
        for step, row in enumerate(self.env.price_data.itertuples()):
            row_time = pd.to_datetime(row.timestamp, unit='s')
            if abs((row_time - target_time).total_seconds()) <= 900:  # Within 15 minutes
                return step
        
        return None
    
    def get_env_state_at_step(self, step: int) -> dict:
        """Get environment state at specific step"""
        if step >= len(self.env.price_data):
            return None
            
        row = self.env.price_data.iloc[step]
        return {
            'current_price': row['close'],
            'timestamp': pd.to_datetime(row['timestamp'], unit='s'),
            'step': step,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'volume': row['volume']
        }
    
    def analyze_surrounding_data(self, anomaly: dict):
        """Analyze market data surrounding the anomaly time"""
        print(f"\nSurrounding Market Data Analysis:")
        
        market_idx = anomaly['market_row_idx']
        start_idx = max(0, market_idx - 2)
        end_idx = min(len(self.market_data), market_idx + 3)
        
        surrounding_data = self.market_data.iloc[start_idx:end_idx]
        
        for idx, row in surrounding_data.iterrows():
            marker = " >>> ANOMALY <<<" if idx == market_idx else ""
            print(f"  {row['datetime']}: Close=${row['close']:,.2f} {marker}")
    
    def run_debug_session(self):
        """Run comprehensive debug session"""
        print("🚀 Starting Trading Environment Debug Session")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Find critical anomalies
        critical_anomalies = self.find_critical_anomalies(max_anomalies=5)
        
        if not critical_anomalies:
            print("✅ No critical anomalies found!")
            return
        
        print(f"\n🔍 Analyzing {len(critical_anomalies)} critical anomalies:")
        
        # Debug each critical anomaly
        for i, anomaly in enumerate(critical_anomalies, 1):
            print(f"\n📊 ANOMALY {i}/{len(critical_anomalies)}")
            self.debug_critical_timeframe(anomaly)
            
            if i < len(critical_anomalies):
                input("\nPress Enter to continue to next anomaly...")
        
        print(f"\n✅ Debug session completed!")
        print(f"Check 'debug_session.log' for detailed logs.")


def main():
    """Main debug session runner"""
    print("Trading Environment Debug Session")
    print("=================================")
    
    # Paths to data files
    market_data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
    
    # Find the most recent trade file
    episodes_dir = "episodes"
    trade_files = []
    
    if os.path.exists(episodes_dir):
        for episode_dir in os.listdir(episodes_dir):
            episode_path = os.path.join(episodes_dir, episode_dir)
            if os.path.isdir(episode_path):
                logs_dir = os.path.join(episode_path, "logs")
                if os.path.exists(logs_dir):
                    for file in os.listdir(logs_dir):
                        if file.startswith("trades_") and file.endswith(".csv"):
                            trade_files.append(os.path.join(logs_dir, file))
    
    if not trade_files:
        print("❌ No trade files found in episodes directory!")
        return
    
    # Use the most recent trade file
    trade_file_path = max(trade_files, key=os.path.getctime)
    print(f"Using trade file: {trade_file_path}")
    
    # Create debugger and run session
    debugger = TradingEnvironmentDebugger(market_data_path, trade_file_path)
    debugger.run_debug_session()


if __name__ == "__main__":
    main()
