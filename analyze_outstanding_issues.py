#!/usr/bin/env python3
"""
Analyze the outstanding issues in trade logging
"""

import pandas as pd
import os
import glob

def analyze_outstanding_issues():
    """Analyze the outstanding trade logging issues"""
    
    print("🔍 Analyzing Outstanding Trade Logging Issues")
    print("=" * 50)
    
    # Look for CSV files in the episode directory structure
    csv_files = []
    
    # Check episode directories
    episode_dirs = glob.glob("episodes/*/logs/")
    for episode_dir in episode_dirs:
        csv_files.extend(glob.glob(f"{episode_dir}trades_*.csv"))
    
    # Also check main logs directory
    csv_files.extend(glob.glob("logs/trades_*.csv"))
    
    if not csv_files:
        print("❌ No trade CSV files found")
        return
    
    # Use the most recent file
    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"📄 Analyzing: {latest_csv}")
    
    try:
        df = pd.read_csv(latest_csv)
        print(f"📊 Total trade records: {len(df)}")
        
        # Issue 1: Inconsistent trade_duration_hours
        print(f"\n🔍 Issue 1: Inconsistent trade_duration_hours")
        print("-" * 40)
        
        # Find trades with 0.0 duration but should have duration
        zero_duration_trades = df[
            (df['trade_duration_hours'] == 0.0) & 
            (df['status'] == 'CLOSED')
        ]
        
        print(f"CLOSED trades with 0.0 duration: {len(zero_duration_trades)}")
        
        if len(zero_duration_trades) > 0:
            print("Sample zero duration trades:")
            for idx, trade in zero_duration_trades.head(5).iterrows():
                print(f"  {trade['trade_id']}: {trade['entry_action']} - Duration: {trade['trade_duration_hours']}h")
        
        # Check if there are trades with non-zero duration for comparison
        non_zero_duration = df[
            (df['trade_duration_hours'] > 0.0) & 
            (df['status'] == 'CLOSED')
        ]
        print(f"CLOSED trades with >0 duration: {len(non_zero_duration)}")
        
        # Issue 2: PnL Recorded on ADJUST Actions
        print(f"\n🔍 Issue 2: PnL on ADJUST Actions")
        print("-" * 35)
        
        adjust_with_pnl = df[
            (df['entry_action'].str.contains('ADJUST', na=False)) &
            (df['net_pnl'] != 0.0) &
            (df['status'] == 'OPEN')
        ]
        
        print(f"ADJUST actions with non-zero PnL: {len(adjust_with_pnl)}")
        
        if len(adjust_with_pnl) > 0:
            print("Sample ADJUST actions with PnL:")
            for idx, trade in adjust_with_pnl.head(5).iterrows():
                print(f"  {trade['trade_id']}: {trade['entry_action']} - PnL: ${trade['net_pnl']:.2f}")
        
        # Issue 3: Unclear close_reason values
        print(f"\n🔍 Issue 3: Unclear close_reason values")
        print("-" * 35)
        
        close_reasons = df['close_reason'].value_counts()
        print("Close reason distribution:")
        for reason, count in close_reasons.items():
            print(f"  {reason}: {count}")
        
        # Focus on the unclear ones
        unclear_reasons = ['CANCEL_CLOSE', 'CANCEL_ACTION']
        for reason in unclear_reasons:
            reason_trades = df[df['close_reason'] == reason]
            if len(reason_trades) > 0:
                print(f"\n{reason} trades: {len(reason_trades)}")
                sample = reason_trades.head(3)
                for idx, trade in sample.iterrows():
                    print(f"  {trade['trade_id']}: {trade['entry_action']} - Status: {trade['status']}")
        
        # Issue 4: Redundant Timestamps on Closing Rows
        print(f"\n🔍 Issue 4: Redundant Timestamps")
        print("-" * 30)
        
        # Check CLOSED trades where entry_datetime == close_datetime
        closed_trades = df[df['status'] == 'CLOSED'].copy()
        closed_trades['same_timestamp'] = closed_trades['entry_datetime'] == closed_trades['close_datetime']
        
        same_timestamp_count = closed_trades['same_timestamp'].sum()
        print(f"CLOSED trades with same entry/close timestamp: {same_timestamp_count}/{len(closed_trades)}")
        
        if same_timestamp_count > 0:
            print("Sample trades with same timestamps:")
            sample_same = closed_trades[closed_trades['same_timestamp']].head(3)
            for idx, trade in sample_same.iterrows():
                print(f"  {trade['trade_id']}: Entry={trade['entry_datetime']}, Close={trade['close_datetime']}")
        
        # Additional analysis: Check if we can find the original OPEN record for these CLOSED trades
        print(f"\n📋 Additional Analysis: Trade Sequence Issues")
        print("-" * 45)
        
        # Group by trade_id to analyze sequences
        trade_groups = df.groupby('trade_id')
        
        problematic_sequences = 0
        good_sequences = 0
        
        for trade_id, group in trade_groups:
            if len(group) > 1:  # Multi-record trades
                open_records = group[group['status'] == 'OPEN']
                closed_records = group[group['status'] == 'CLOSED']
                
                if len(closed_records) > 0:
                    final_closed = closed_records.iloc[-1]
                    
                    # Check if the closed record has proper entry_datetime
                    if len(open_records) > 0:
                        first_open = open_records.iloc[0]
                        if final_closed['entry_datetime'] == final_closed['close_datetime']:
                            # Should use the original open datetime
                            if final_closed['entry_datetime'] != first_open['entry_datetime']:
                                problematic_sequences += 1
                    else:
                        good_sequences += 1
                else:
                    good_sequences += 1
        
        print(f"Trades with proper timestamp sequencing: {good_sequences}")
        print(f"Trades with timestamp issues: {problematic_sequences}")
        
        # Summary of fixes needed
        print(f"\n🎯 Summary of Issues to Fix:")
        print("-" * 30)
        print(f"1. ❌ Trade duration calculation: {len(zero_duration_trades)} trades with 0.0 duration")
        print(f"2. ❌ PnL on ADJUST actions: {len(adjust_with_pnl)} ADJUST records with PnL")
        print(f"3. ⚠️  Unclear close reasons: {len(df[df['close_reason'].isin(unclear_reasons)])} trades")
        print(f"4. ❌ Timestamp redundancy: {same_timestamp_count} trades with same entry/close time")
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

if __name__ == "__main__":
    analyze_outstanding_issues()
