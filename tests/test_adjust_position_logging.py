#!/usr/bin/env python3
"""
Test script to verify ADJUST position logging issue
"""

import pandas as pd
import sys
import os

def analyze_adjust_position_logs():
    """Analyze CSV logs to check ADJUST position logging"""
    
    # Look for CSV files in logs directory
    logs_dir = "logs"
    csv_files = []
    
    if os.path.exists(logs_dir):
        for file in os.listdir(logs_dir):
            if file.endswith('.csv') and 'trades' in file:
                csv_files.append(os.path.join(logs_dir, file))
    
    if not csv_files:
        print("No trade CSV files found in logs directory")
        return
    
    # Use the most recent CSV file
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"Analyzing: {latest_csv}")
    
    try:
        df = pd.read_csv(latest_csv)
        print(f"Total trades in CSV: {len(df)}")
        
        # Filter for ADJUST actions
        adjust_trades = df[df['entry_action'].str.contains('ADJUST', na=False)]
        print(f"\nADJUST trades found: {len(adjust_trades)}")
        
        if len(adjust_trades) > 0:
            print("\nADJUST Trade Analysis:")
            print("=" * 80)
            
            for idx, trade in adjust_trades.iterrows():
                print(f"Trade ID: {trade['trade_id']}")
                print(f"  Step: {trade['training_step']}")
                print(f"  Action: {trade['entry_action']}")
                print(f"  Status: {trade['status']}")
                print(f"  Position Size: {trade['position_size']}")
                print(f"  Fees Paid: {trade['fees_paid']}")
                print(f"  Side: {trade['side']}")
                print(f"  Entry Price: {trade['entry_price']}")
                
                # Check if position_size is incorrectly 0.0
                if trade['status'] == 'OPEN' and abs(float(trade['position_size'])) < 0.001:
                    print(f"  ❌ ISSUE: Position size is {trade['position_size']} but status is OPEN")
                else:
                    print(f"  ✅ OK: Position size looks correct")
                print()
        
        # Look for patterns where position_size is 0 but status is OPEN
        zero_position_open = df[(abs(df['position_size']) < 0.001) & (df['status'] == 'OPEN')]
        if len(zero_position_open) > 0:
            print(f"\n❌ FOUND {len(zero_position_open)} trades with zero position_size but OPEN status:")
            for idx, trade in zero_position_open.iterrows():
                print(f"  {trade['trade_id']}: {trade['entry_action']} - Position: {trade['position_size']}")
        
        # Look for ADJUST trades specifically
        if len(adjust_trades) > 0:
            zero_adjust = adjust_trades[abs(adjust_trades['position_size']) < 0.001]
            if len(zero_adjust) > 0:
                print(f"\n❌ FOUND {len(zero_adjust)} ADJUST trades with incorrect zero position_size:")
                for idx, trade in zero_adjust.iterrows():
                    print(f"  {trade['trade_id']}: Position size should not be 0 for ADJUST action")
            else:
                print(f"\n✅ All {len(adjust_trades)} ADJUST trades have valid position_size")
    
    except Exception as e:
        print(f"Error analyzing CSV: {e}")

def check_trade_sequence():
    """Check for trade sequence patterns that might reveal the issue"""
    logs_dir = "logs"
    csv_files = []
    
    if os.path.exists(logs_dir):
        for file in os.listdir(logs_dir):
            if file.endswith('.csv') and 'trades' in file:
                csv_files.append(os.path.join(logs_dir, file))
    
    if not csv_files:
        return
    
    latest_csv = max(csv_files, key=os.path.getmtime)
    
    try:
        df = pd.read_csv(latest_csv)
        
        # Group by trade_id to see sequences
        trade_groups = df.groupby('trade_id')
        
        print(f"\nTrade Sequence Analysis:")
        print("=" * 80)
        
        for trade_id, group in trade_groups:
            if len(group) > 1:  # Multi-step trades
                print(f"\nTrade {trade_id} sequence:")
                for idx, step in group.iterrows():
                    print(f"  Step {step['training_step']}: {step['entry_action']} - "
                          f"Position: {step['position_size']} - Status: {step['status']}")
                    
                    # Check for inconsistencies
                    if step['entry_action'].startswith('ADJUST') and abs(float(step['position_size'])) < 0.001:
                        print(f"    ❌ ISSUE: ADJUST action with zero position_size")
    
    except Exception as e:
        print(f"Error in sequence analysis: {e}")

if __name__ == "__main__":
    print("🔍 Analyzing ADJUST Position Logging Issue")
    print("=" * 50)
    
    analyze_adjust_position_logs()
    check_trade_sequence()
    
    print(f"\n📋 Summary:")
    print("This script checks for the reported issue where ADJUST actions")
    print("incorrectly show position_size as 0.0 in the CSV logs.")
    print("If issues are found, they will be marked with ❌")
    print("If everything looks correct, they will be marked with ✅")
