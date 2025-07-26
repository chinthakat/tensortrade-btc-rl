#!/usr/bin/env python3
"""
Test script to verify the FLIP trade logging fix.
This demonstrates how FLIP operations should now generate two separate trades.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def analyze_flip_trades(csv_file_path):
    """Analyze a trade CSV file to check for proper FLIP handling"""
    
    print(f"Analyzing trade file: {csv_file_path}")
    print("=" * 80)
    
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Find FLIP-related issues
    print("\n1. CHECKING FOR ORPHANED TRADES (trades that never close):")
    print("-" * 60)
    
    orphaned_trades = []
    for trade_id in df['trade_id'].unique():
        if trade_id.startswith('TRADE_'):
            trade_records = df[df['trade_id'] == trade_id]
            
            # Check if trade has a proper close
            has_close = any(
                record['status'] == 'CLOSED' or 
                record['entry_action'] in ['CLOSE_LONG', 'CLOSE_SHORT'] or
                'CLOSE' in str(record['entry_action'])
                for _, record in trade_records.iterrows()
            )
            
            if not has_close and len(trade_records) == 1:
                # Check if next trade has PnL (indicating misattributed PnL)
                trade_num = int(trade_id.split('_')[1])
                next_trade_id = f"TRADE_{trade_num+1:05d}"
                next_trade = df[df['trade_id'] == next_trade_id]
                
                if not next_trade.empty:
                    next_pnl = next_trade.iloc[0]['net_pnl']
                    if abs(next_pnl) > 0.001:  # Has significant PnL
                        orphaned_trades.append({
                            'trade_id': trade_id,
                            'action': trade_records.iloc[0]['entry_action'],
                            'step': trade_records.iloc[0]['training_step'],
                            'next_trade_pnl': next_pnl
                        })
    
    if orphaned_trades:
        print(f"Found {len(orphaned_trades)} orphaned trades:")
        for trade in orphaned_trades[:5]:  # Show first 5
            print(f"  {trade['trade_id']} ({trade['action']}) at step {trade['step']} - "
                  f"Next trade has PnL: {trade['next_trade_pnl']:.6f}")
        if len(orphaned_trades) > 5:
            print(f"  ... and {len(orphaned_trades) - 5} more")
    else:
        print("✅ No orphaned trades found!")
    
    print("\n2. CHECKING FOR OLD-STYLE FLIP OPERATIONS:")
    print("-" * 60)
    
    old_flips = df[df['entry_action'].str.contains('FLIP_', na=False)]
    if not old_flips.empty:
        print(f"❌ Found {len(old_flips)} old-style FLIP operations:")
        for i, (_, flip) in enumerate(old_flips.head(3).iterrows()):
            print(f"  {flip['trade_id']}: {flip['entry_action']} with PnL {flip['net_pnl']:.6f}")
    else:
        print("✅ No old-style FLIP operations found!")
    
    print("\n3. CHECKING FOR PROPER CLOSE/OPEN PAIRS:")
    print("-" * 60)
    
    # Look for close/open pairs that should represent position flips
    close_actions = ['CLOSE_LONG', 'CLOSE_SHORT']
    open_actions = ['OPEN_LONG', 'OPEN_SHORT']
    
    proper_pairs = 0
    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_trade = df.iloc[i + 1]
        
        # Check if current is a close and next is an open at the same step
        if (current['entry_action'] in close_actions and 
            next_trade['entry_action'] in open_actions and
            current['training_step'] == next_trade['training_step']):
            proper_pairs += 1
    
    print(f"✅ Found {proper_pairs} proper CLOSE/OPEN pairs (representing position flips)")
    
    print("\n4. TRADE SEQUENCE ANALYSIS:")
    print("-" * 60)
    
    # Show first few trades to demonstrate proper structure
    print("First 10 trades:")
    for i, (_, trade) in enumerate(df.head(10).iterrows()):
        status_icon = "🟢" if trade['status'] == 'OPEN' else "🔴" if trade['status'] == 'CLOSED' else "⚫"
        pnl_str = f"PnL: {trade['net_pnl']:8.3f}" if abs(trade['net_pnl']) > 0.001 else "PnL:    0.000"
        print(f"  {status_icon} {trade['trade_id']}: {trade['entry_action']:15} | {pnl_str} | Step: {trade['training_step']}")
    
    print("\nSUMMARY:")
    print("=" * 80)
    total_trades = len(df[df['trade_id'].str.startswith('TRADE_', na=False)])
    closed_trades = len(df[df['status'] == 'CLOSED'])
    open_trades = len(df[df['status'] == 'OPEN'])
    
    print(f"Total TRADE entries: {total_trades}")
    print(f"Closed trades: {closed_trades}")
    print(f"Open trades: {open_trades}")
    print(f"Orphaned trades: {len(orphaned_trades)}")
    print(f"Old-style FLIPs: {len(old_flips)}")
    print(f"Proper CLOSE/OPEN pairs: {proper_pairs}")
    
    if len(orphaned_trades) == 0 and len(old_flips) == 0:
        print("\n🎉 FLIP FIX VERIFICATION: PASSED! No issues found.")
    else:
        print(f"\n❌ FLIP FIX VERIFICATION: FAILED! Found {len(orphaned_trades)} orphaned trades and {len(old_flips)} old-style FLIPs.")

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
                    analyze_flip_trades(latest_trade_file)
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
            analyze_flip_trades(csv_file)
        else:
            print("Invalid file path")
