#!/usr/bin/env python3
"""
Test script to verify the trade duration calculation fix.
This checks that trade_duration_hours is calculated correctly for closed trades.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_trade_durations(csv_file_path):
    """Analyze a trade CSV file to check for proper duration calculations"""
    
    print(f"Analyzing trade durations in: {csv_file_path}")
    print("=" * 80)
    
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Filter for actual TRADE entries only
    trade_df = df[df['trade_id'].str.startswith('TRADE_', na=False)]
    
    print(f"\n📊 TOTAL TRADES ANALYZED: {len(trade_df)}")
    print("-" * 60)
    
    # Check for closed trades with proper duration
    closed_trades = trade_df[trade_df['status'] == 'CLOSED']
    print(f"Closed trades: {len(closed_trades)}")
    
    # Check trades with 0 duration
    zero_duration_trades = closed_trades[closed_trades['trade_duration_hours'] == 0]
    print(f"Closed trades with 0 duration: {len(zero_duration_trades)} ❌")
    
    # Check trades with proper duration (> 0)
    proper_duration_trades = closed_trades[closed_trades['trade_duration_hours'] > 0]
    print(f"Closed trades with proper duration: {len(proper_duration_trades)} ✅")
    
    if len(zero_duration_trades) > 0:
        print(f"\n❌ DURATION FIX VERIFICATION: FAILED!")
        print(f"Found {len(zero_duration_trades)} closed trades with 0 duration")
        
        print("\nSample trades with incorrect duration:")
        for i, (_, trade) in enumerate(zero_duration_trades.head(5).iterrows()):
            print(f"  {trade['trade_id']}: {trade['entry_action']} - Duration: {trade['trade_duration_hours']}h")
    
    else:
        print(f"\n✅ DURATION FIX VERIFICATION: PASSED!")
        print(f"All {len(closed_trades)} closed trades have proper duration calculation")
    
    # Analyze duration statistics for proper trades
    if len(proper_duration_trades) > 0:
        print(f"\n📈 DURATION STATISTICS:")
        print("-" * 60)
        durations = proper_duration_trades['trade_duration_hours']
        print(f"Average duration: {durations.mean():.2f} hours")
        print(f"Median duration: {durations.median():.2f} hours")
        print(f"Min duration: {durations.min():.2f} hours")
        print(f"Max duration: {durations.max():.2f} hours")
        print(f"Standard deviation: {durations.std():.2f} hours")
    
    # Sample verification with timestamp calculation
    print(f"\n🔍 MANUAL VERIFICATION (Sample Trades):")
    print("-" * 60)
    
    sample_trades = []
    
    # Look for trade pairs (open + close) to verify duration
    for trade_id in closed_trades['trade_id'].unique()[:3]:  # Check first 3 unique trades
        trade_records = trade_df[trade_df['trade_id'] == trade_id].sort_values('training_step')
        
        if len(trade_records) >= 2:  # Has open and close records
            open_record = trade_records.iloc[0]
            close_record = trade_records.iloc[-1]
            
            if (pd.notna(open_record['entry_datetime']) and 
                pd.notna(close_record['entry_datetime'])):
                
                # Calculate manual duration
                entry_timestamp = float(open_record['entry_datetime'])
                close_timestamp = float(close_record['entry_datetime'])
                duration_seconds = close_timestamp - entry_timestamp
                manual_duration_hours = duration_seconds / 3600
                
                logged_duration = close_record['trade_duration_hours']
                
                sample_trades.append({
                    'trade_id': trade_id,
                    'entry_step': open_record['training_step'],
                    'close_step': close_record['training_step'],
                    'step_diff': close_record['training_step'] - open_record['training_step'],
                    'manual_duration_hours': manual_duration_hours,
                    'logged_duration_hours': logged_duration,
                    'duration_match': abs(manual_duration_hours - logged_duration) < 0.1
                })
    
    for trade in sample_trades:
        match_icon = "✅" if trade['duration_match'] else "❌"
        print(f"  {match_icon} {trade['trade_id']}:")
        print(f"     Steps: {trade['entry_step']} → {trade['close_step']} (diff: {trade['step_diff']})")
        print(f"     Manual calc: {trade['manual_duration_hours']:.2f}h")
        print(f"     Logged: {trade['logged_duration_hours']:.2f}h")
        print(f"     Match: {trade['duration_match']}")
        print()
    
    # Overall verification
    print(f"\n🎯 OVERALL VERIFICATION RESULTS:")
    print("=" * 80)
    total_closed = len(closed_trades)
    proper_duration = len(proper_duration_trades)
    zero_duration = len(zero_duration_trades)
    
    if total_closed == 0:
        print("⚠️  No closed trades found - cannot verify duration calculation")
    elif zero_duration == 0:
        print(f"🎉 SUCCESS: All {total_closed} closed trades have proper duration calculation!")
        if sample_trades:
            all_matches = all(trade['duration_match'] for trade in sample_trades)
            if all_matches:
                print(f"🎉 MANUAL VERIFICATION: All {len(sample_trades)} sample calculations match!")
            else:
                mismatches = sum(1 for trade in sample_trades if not trade['duration_match'])
                print(f"⚠️  MANUAL VERIFICATION: {mismatches}/{len(sample_trades)} samples have duration mismatches")
    else:
        success_rate = (proper_duration / total_closed) * 100
        print(f"❌ PARTIAL SUCCESS: {proper_duration}/{total_closed} trades ({success_rate:.1f}%) have proper duration")
        print(f"❌ {zero_duration} trades still have 0 duration - fix incomplete")

if __name__ == "__main__":
    # Find the most recent trade file
    episodes_dir = Path("episodes")
    if episodes_dir.exists():
        episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir() and "episode_" in d.name]
        if episode_dirs:
            latest_episode = max(episode_dirs, key=lambda d: d.name)
            logs_dir = latest_episode / "logs"
            if logs_dir.exists():
                trade_files = list(logs_dir.glob("trades_*.csv"))
                if trade_files:
                    latest_trade_file = max(trade_files, key=lambda f: f.stat().st_mtime)
                    analyze_trade_durations(latest_trade_file)
                else:
                    print("No trade CSV files found in logs directory")
            else:
                print("No logs directory found in latest episode")
        else:
            print("No episode directories found")
    else:
        print("Episodes directory not found")
        print("Please provide the path to a trade CSV file manually:")
        csv_file = input("Enter CSV file path: ").strip()
        if csv_file and Path(csv_file).exists():
            analyze_trade_durations(csv_file)
        else:
            print("Invalid file path")
