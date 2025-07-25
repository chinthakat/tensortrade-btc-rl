#!/usr/bin/env python3
"""
Future Verification Script for ADJUST Position Logging
Run this after training to verify the fix works
"""

import pandas as pd
import os
import glob

def verify_adjust_fix():
    """Verify that the ADJUST position logging fix is working"""
    
    print("🔍 ADJUST Position Logging Fix Verification")
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
        print(f"📊 Total trades: {len(df)}")
        
        # Check for ADJUST actions
        adjust_trades = df[df['entry_action'].str.contains('ADJUST', na=False)]
        print(f"🔧 ADJUST trades found: {len(adjust_trades)}")
        
        if len(adjust_trades) == 0:
            print("⚠️  No ADJUST trades found in this run")
            return
        
        # Check for the specific bug: ADJUST with position_size = 0.0 but status = OPEN
        problematic_adjust = adjust_trades[
            (abs(adjust_trades['position_size']) < 0.000001) & 
            (adjust_trades['status'] == 'OPEN')
        ]
        
        print(f"\n🎯 Fix Verification Results:")
        print("-" * 30)
        
        if len(problematic_adjust) > 0:
            print(f"❌ BUG STILL EXISTS: Found {len(problematic_adjust)} ADJUST trades with zero position_size")
            print("\nProblematic trades:")
            for idx, trade in problematic_adjust.iterrows():
                print(f"  - {trade['trade_id']}: {trade['entry_action']} at step {trade['training_step']}")
                print(f"    Position: {trade['position_size']}, Status: {trade['status']}")
        else:
            print(f"✅ FIX SUCCESSFUL: All {len(adjust_trades)} ADJUST trades have valid position_size")
        
        # Additional checks
        print(f"\n📈 ADJUST Trade Analysis:")
        print("-" * 25)
        
        open_adjust = adjust_trades[adjust_trades['status'] == 'OPEN']
        closed_adjust = adjust_trades[adjust_trades['status'] == 'CLOSED']
        
        print(f"Open ADJUST trades: {len(open_adjust)}")
        print(f"Closed ADJUST trades: {len(closed_adjust)}")
        
        if len(open_adjust) > 0:
            avg_position = open_adjust['position_size'].mean()
            min_position = open_adjust['position_size'].min()
            max_position = open_adjust['position_size'].max()
            
            print(f"Position size stats for OPEN ADJUST trades:")
            print(f"  Average: {avg_position:.6f} BTC")
            print(f"  Range: {min_position:.6f} to {max_position:.6f} BTC")
            
            # Check if any open adjusts have zero position
            zero_position_open = open_adjust[abs(open_adjust['position_size']) < 0.000001]
            if len(zero_position_open) > 0:
                print(f"  ❌ WARNING: {len(zero_position_open)} OPEN ADJUST trades with ~0 position_size")
            else:
                print(f"  ✅ All OPEN ADJUST trades have valid position sizes")
        
        # Check for data consistency
        print(f"\n🔍 Data Consistency Checks:")
        print("-" * 27)
        
        # Check side vs position_size consistency
        inconsistent_side = 0
        for idx, trade in adjust_trades.iterrows():
            position = trade['position_size']
            side = trade['side']
            
            if position > 0.000001 and side != 'LONG':
                inconsistent_side += 1
            elif position < -0.000001 and side != 'SHORT':
                inconsistent_side += 1
            elif abs(position) < 0.000001 and side != 'FLAT':
                inconsistent_side += 1
        
        if inconsistent_side > 0:
            print(f"❌ Found {inconsistent_side} ADJUST trades with inconsistent side/position_size")
        else:
            print(f"✅ All ADJUST trades have consistent side/position_size")
        
        # Sample some ADJUST trades
        if len(adjust_trades) > 0:
            print(f"\n📋 Sample ADJUST Trades:")
            print("-" * 22)
            sample_size = min(5, len(adjust_trades))
            sample_trades = adjust_trades.head(sample_size)
            
            for idx, trade in sample_trades.iterrows():
                print(f"  {trade['trade_id']}: {trade['entry_action']}")
                print(f"    Position: {trade['position_size']:.6f} BTC")
                print(f"    Status: {trade['status']}, Side: {trade['side']}")
                print(f"    Step: {trade['training_step']}")
                print()
    
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

if __name__ == "__main__":
    verify_adjust_fix()
