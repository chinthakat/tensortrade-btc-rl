#!/usr/bin/env python3
"""
Analyze New Test Trade Data
==========================

Direct analysis of the new trade data generated after the price alignment fix.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_test_trade_data():
    """Analyze the new test trade data for anomalies"""
    
    print("🔍 ANALYZING NEW TEST TRADE DATA")
    print("=" * 50)
    
    # File paths
    market_data_path = "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv"
    test_trade_path = "episodes/test_fix_20250720_130050/logs/trades_test_fix_20250720_130050.csv"
    
    # Load market data
    print("Loading market data...")
    market_data = pd.read_csv(market_data_path)
    market_data['datetime'] = pd.to_datetime(market_data['timestamp'], unit='s')
    print(f"✓ Market data loaded: {len(market_data)} records")
    
    # Load test trade data
    print("Loading test trade data...")
    trade_data = pd.read_csv(test_trade_path)
    print(f"✓ Trade data loaded: {len(trade_data)} records")
    
    # Analyze entry price accuracy
    print(f"\nAnalyzing entry price accuracy:")
    print("-" * 30)
    
    anomalies = []
    tolerance_pct = 1.0  # 1% tolerance
    
    for idx, trade in trade_data.iterrows():
        if pd.isna(trade.get('entry_price')) or trade.get('entry_price', 0) == 0:
            continue
            
        # Parse entry datetime
        try:
            if pd.isna(trade.get('entry_datetime')):
                continue
            entry_time = pd.to_datetime(trade['entry_datetime'], unit='s')
        except:
            continue
        
        # Find corresponding market data
        time_diffs = abs(market_data['datetime'] - entry_time)
        closest_idx = time_diffs.idxmin()
        
        # Only check if within 15 minutes
        if time_diffs.loc[closest_idx].total_seconds() <= 900:
            market_price = market_data.loc[closest_idx, 'close']
            trade_price = trade['entry_price']
            
            # Calculate difference
            diff_pct = abs(trade_price - market_price) / market_price * 100
            
            print(f"Trade {idx + 1}:")
            print(f"  Entry Time: {entry_time}")
            print(f"  Trade Price: ${trade_price:,.2f}")
            print(f"  Market Price: ${market_price:,.2f}")
            print(f"  Difference: {diff_pct:.4f}%")
            
            if diff_pct > tolerance_pct:
                anomalies.append({
                    'trade_idx': idx,
                    'entry_time': entry_time,
                    'trade_price': trade_price,
                    'market_price': market_price,
                    'difference_pct': diff_pct
                })
                print(f"  ⚠️  ANOMALY: Exceeds {tolerance_pct}% tolerance")
            else:
                print(f"  ✅ GOOD: Within {tolerance_pct}% tolerance")
            print()
    
    # Summary
    print(f"📊 ANALYSIS SUMMARY:")
    print("=" * 30)
    print(f"Total trades analyzed: {len(trade_data)}")
    print(f"Anomalies found: {len(anomalies)}")
    print(f"Anomaly rate: {len(anomalies)/len(trade_data)*100:.2f}%")
    
    if len(anomalies) == 0:
        print(f"🎉 SUCCESS: No price anomalies found!")
        print(f"The price alignment fix appears to be working correctly!")
    else:
        print(f"⚠️  Still found {len(anomalies)} anomalies")
        print("Worst anomalies:")
        for anomaly in sorted(anomalies, key=lambda x: x['difference_pct'], reverse=True)[:3]:
            print(f"  {anomaly['difference_pct']:.2f}% at {anomaly['entry_time']}")

if __name__ == "__main__":
    analyze_test_trade_data()
