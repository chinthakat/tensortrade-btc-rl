#!/usr/bin/env python3
"""
Analyze dangling OPEN records issue
"""

import pandas as pd
import os
import glob

def analyze_dangling_open_trades():
    """Analyze CSV logs to find dangling OPEN trades"""
    
    print("🔍 Analyzing Dangling OPEN Trades Issue")
    print("=" * 50)
    
    # Find the most recent trade CSV file
    csv_pattern = "logs/trades_*.csv"
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("❌ No trade CSV files found")
        print("   Expected location: logs/trades_YYYYMMDD_HHMMSS.csv")
        return
    
    # Use the most recent file
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"📄 Analyzing: {latest_csv}")
    
    try:
        df = pd.read_csv(latest_csv)
        print(f"📊 Total trade records: {len(df)}")
        
        # Get unique trade IDs
        trade_ids = df['trade_id'].unique()
        print(f"🔢 Unique trade IDs: {len(trade_ids)}")
        
        # Analyze each trade ID for closure status
        dangling_trades = []
        properly_closed = []
        episode_cleanups = []
        
        for trade_id in trade_ids:
            trade_records = df[df['trade_id'] == trade_id]
            
            # Skip episode cleanup records (these are separate)
            if trade_id.startswith('EPISODE_END_'):
                episode_cleanups.append(trade_id)
                continue
            
            # Check final status
            final_status = trade_records.iloc[-1]['status']
            
            if final_status == 'OPEN':
                dangling_trades.append({
                    'trade_id': trade_id,
                    'records': len(trade_records),
                    'final_step': trade_records.iloc[-1]['training_step'],
                    'position_size': trade_records.iloc[-1]['position_size'],
                    'entry_action': trade_records.iloc[-1]['entry_action']
                })
            else:
                properly_closed.append(trade_id)
        
        print(f"\n📋 Analysis Results:")
        print("-" * 25)
        print(f"✅ Properly closed trades: {len(properly_closed)}")
        print(f"🧹 Episode cleanup records: {len(episode_cleanups)}")
        print(f"❌ Dangling OPEN trades: {len(dangling_trades)}")
        
        if len(dangling_trades) > 0:
            print(f"\n🚨 Dangling OPEN Trades Details:")
            print("-" * 35)
            for trade in dangling_trades:
                print(f"  {trade['trade_id']}:")
                print(f"    Records: {trade['records']}")
                print(f"    Final Step: {trade['final_step']}")
                print(f"    Position Size: {trade['position_size']:.6f}")
                print(f"    Last Action: {trade['entry_action']}")
                print()
        
        # Check for missing EPISODE_END closures
        if len(dangling_trades) > 0 and len(episode_cleanups) == 0:
            print(f"⚠️  No EPISODE_END cleanup records found!")
            print("   This suggests episode termination cleanup is not working properly.")
        elif len(dangling_trades) > 0 and len(episode_cleanups) > 0:
            print(f"🔄 Found {len(episode_cleanups)} EPISODE_END records")
            print("   Issue: Cleanup creates new records instead of updating existing OPEN trades")
        
        # Check maximum training step to see if episode ended
        max_step = df['training_step'].max()
        print(f"\n📊 Episode Information:")
        print(f"   Maximum training step: {max_step}")
        
        if len(dangling_trades) > 0:
            latest_dangling_step = max(trade['final_step'] for trade in dangling_trades)
            print(f"   Latest dangling trade step: {latest_dangling_step}")
            
            if latest_dangling_step == max_step:
                print("   ❌ Trades were left open at episode end")
            else:
                print("   ⚠️  Trades were left open mid-episode")
        
        print(f"\n🎯 Root Cause Analysis:")
        if len(dangling_trades) > 0:
            print("❌ ISSUE CONFIRMED: Dangling OPEN trades exist")
            print("🔍 Likely causes:")
            print("   1. Episode termination cleanup creates new records instead of updating existing")
            print("   2. _force_close_position_no_fees uses different trade_id format")
            print("   3. Missing proper closure logic for open positions at episode end")
        else:
            print("✅ NO ISSUES: All trades properly closed")
    
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

if __name__ == "__main__":
    analyze_dangling_open_trades()
