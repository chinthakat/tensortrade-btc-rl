#!/usr/bin/env python3
"""
Show downloaded data structure and contents
"""

import os
import pandas as pd
from pathlib import Path

def show_data_structure(base_dir="data/coinapi"):
    """Show the structure of downloaded data"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Directory {base_dir} does not exist")
        return
    
    print(f"📁 Data Structure in {base_dir}")
    print("=" * 60)
    
    for symbol_dir in sorted(base_path.iterdir()):
        if symbol_dir.is_dir():
            print(f"\n📊 {symbol_dir.name}")
            
            for interval_dir in sorted(symbol_dir.iterdir()):
                if interval_dir.is_dir():
                    print(f"   ⏱️  {interval_dir.name}")
                    
                    for data_type_dir in sorted(interval_dir.iterdir()):
                        if data_type_dir.is_dir():
                            files = list(data_type_dir.glob("*.csv"))
                            print(f"      📈 {data_type_dir.name}: {len(files)} files")
                            
                            # Show sample data from first file
                            if files:
                                sample_file = files[0]
                                try:
                                    df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
                                    print(f"         📄 Sample: {sample_file.name}")
                                    print(f"         📊 Records: {len(df)}")
                                    print(f"         📅 Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
                                    print(f"         📋 Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                                    
                                    # Show sample values
                                    if data_type_dir.name == 'ohlcv' and 'close' in df.columns:
                                        print(f"         💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                                    elif data_type_dir.name == 'funding_rates' and 'funding_rate' in df.columns:
                                        avg_rate = df['funding_rate'].mean() * 100
                                        print(f"         📈 Avg funding rate: {avg_rate:.4f}%")
                                        
                                except Exception as e:
                                    print(f"         ❌ Error reading file: {e}")

def main():
    show_data_structure()

if __name__ == "__main__":
    main()
