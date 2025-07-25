#!/usr/bin/env python3
"""
Future verification script for dangling OPEN trades fix
Run this after training to verify no dangling OPEN trades exist
"""

import pandas as pd
import os
import glob

def verify_no_dangling_open_trades():
    """Verify that no dangling OPEN trades exist in CSV logs"""
    
    print("🔍 Verifying No Dangling OPEN Trades")
    print("=" * 40)
    
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
        
        # Get unique trade IDs (excluding episode cleanup records)
        regular_trade_ids = [tid for tid in df['trade_id'].unique() 
                           if not tid.startswith('EPISODE_END_')]
        print(f"🔢 Regular trade IDs: {len(regular_trade_ids)}")
        
        # Check for dangling OPEN trades
        dangling_trades = []
        properly_closed = []
        episode_cleanups = []
        
        for trade_id in regular_trade_ids:
            trade_records = df[df['trade_id'] == trade_id]
            final_status = trade_records.iloc[-1]['status']
            
            if final_status == 'OPEN':
                dangling_trades.append({
                    'trade_id': trade_id,
                    'final_step': trade_records.iloc[-1]['training_step'],
                    'position_size': trade_records.iloc[-1]['position_size'],
                    'entry_action': trade_records.iloc[-1]['entry_action']
                })
            else:
                properly_closed.append(trade_id)
        
        # Check for old-style EPISODE_END records
        episode_end_records = df[df['trade_id'].str.startswith('EPISODE_END_', na=False)]
        
        print(f"\n📋 Verification Results:")
        print("-" * 25)
        print(f"✅ Properly closed trades: {len(properly_closed)}")
        print(f"❌ Dangling OPEN trades: {len(dangling_trades)}")
        print(f"🔍 Old EPISODE_END records: {len(episode_end_records)}")
        
        # Main verification
        if len(dangling_trades) == 0:
            print(f"\n🎉 FIX SUCCESSFUL!")
            print(f"✅ No dangling OPEN trades found")
            if len(episode_end_records) == 0:
                print(f"✅ No old EPISODE_END records found")
            else:
                print(f"⚠️  Found {len(episode_end_records)} old EPISODE_END records (from before fix)")
        else:
            print(f"\n❌ FIX FAILED!")
            print(f"Found {len(dangling_trades)} dangling OPEN trades:")
            for trade in dangling_trades:
                print(f"  - {trade['trade_id']}: Step {trade['final_step']}, "
                      f"Size {trade['position_size']:.6f}, Action {trade['entry_action']}")
        
        # Additional checks
        print(f"\n🔬 Additional Analysis:")
        print("-" * 22)
        
        # Check episode boundary behavior
        max_step = df['training_step'].max()
        final_step_trades = df[df['training_step'] == max_step]
        
        print(f"Episode ended at step: {max_step}")
        print(f"Trades at final step: {len(final_step_trades)}")
        
        final_open_at_end = final_step_trades[final_step_trades['status'] == 'OPEN']
        if len(final_open_at_end) > 0:
            print(f"❌ {len(final_open_at_end)} trades left OPEN at episode end")
        else:
            print(f"✅ No OPEN trades at episode end")
        
        # Check for proper closure patterns
        force_close_actions = df[df['entry_action'].str.contains('FORCE_CLOSE', na=False)]
        cancel_close_actions = df[df['entry_action'].str.contains('CANCEL_CLOSE', na=False)]
        
        print(f"FORCE_CLOSE actions: {len(force_close_actions)}")
        print(f"CANCEL_CLOSE actions: {len(cancel_close_actions)}")
        
        # Check trade ID consistency
        all_trade_ids = df['trade_id'].unique()
        regular_pattern = [tid for tid in all_trade_ids if tid.startswith('TRADE_')]
        episode_pattern = [tid for tid in all_trade_ids if tid.startswith('EPISODE_END_')]
        
        print(f"\nTrade ID patterns:")
        print(f"  TRADE_xxxxx: {len(regular_pattern)}")
        print(f"  EPISODE_END_xxxxx: {len(episode_pattern)}")
        
        if len(episode_pattern) == 0:
            print(f"✅ All trades use consistent TRADE_xxxxx format")
        else:
            print(f"⚠️  Found old EPISODE_END_xxxxx patterns (pre-fix data)")
        
        print(f"\n🎯 Fix Status Summary:")
        if len(dangling_trades) == 0 and len(final_open_at_end) == 0:
            print("🎉 COMPLETE SUCCESS: No dangling OPEN trades detected")
            print("   ✅ Episode termination properly closes all trades")
            print("   ✅ All trades have consistent closure logging")
        else:
            print("❌ ISSUES DETECTED: Fix may not be working properly")
            print("   Check episode termination and CANCEL action logic")
    
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

if __name__ == "__main__":
    verify_no_dangling_open_trades()
