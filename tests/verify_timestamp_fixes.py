#!/usr/bin/env python3
"""
Test script to verify the timestamp and duration fixes
"""

import pandas as pd
import numpy as np

def analyze_trade_consistency(csv_file):
    """Analyze trade logging consistency"""
    print(f"Analyzing trade consistency in: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} trade records")
        
        # Group by trade_id to analyze trade sequences
        trade_groups = df.groupby('trade_id')
        
        issues = []
        
        for trade_id, group in trade_groups:
            # Sort by training_step to get chronological order
            group = group.sort_values('training_step')
            
            # Check if entry_datetime is consistent within trade
            unique_entry_times = group['entry_datetime'].unique()
            if len(unique_entry_times) > 1:
                issues.append({
                    'trade_id': trade_id,
                    'issue': 'inconsistent_entry_datetime',
                    'details': f"Multiple entry times: {unique_entry_times}"
                })
            
            # Check for CLOSED trades with 0 duration
            closed_trades = group[group['status'] == 'CLOSED']
            if len(closed_trades) > 0:
                for _, trade in closed_trades.iterrows():
                    if trade['trade_duration_hours'] == 0.0:
                        issues.append({
                            'trade_id': trade_id,
                            'issue': 'zero_duration',
                            'details': f"CLOSED trade with 0.0 duration at step {trade['training_step']}"
                        })
            
            # Check for ADJUST trades with PnL
            adjust_trades = group[group['entry_action'].str.contains('ADJUST', na=False)]
            for _, trade in adjust_trades.iterrows():
                if trade['status'] == 'OPEN' and trade['net_pnl'] != 0.0:
                    issues.append({
                        'trade_id': trade_id,
                        'issue': 'adjust_has_pnl',
                        'details': f"ADJUST trade with PnL {trade['net_pnl']} while status=OPEN"
                    })
            
            # Check for incorrect close_reason on OPEN trades
            open_trades = group[group['status'] == 'OPEN']
            for _, trade in open_trades.iterrows():
                if trade['close_reason'] != '' and not pd.isna(trade['close_reason']):
                    issues.append({
                        'trade_id': trade_id,
                        'issue': 'open_has_close_reason',
                        'details': f"OPEN trade has close_reason: {trade['close_reason']}"
                    })
        
        # Summary
        print(f"\nAnalysis Results:")
        print(f"Total trades analyzed: {len(trade_groups)}")
        print(f"Total issues found: {len(issues)}")
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue['issue']
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        for issue_type, issue_list in issue_types.items():
            print(f"  {issue_type}: {len(issue_list)} occurrences")
            
            # Show first few examples
            for i, issue in enumerate(issue_list[:3]):
                print(f"    - {issue['trade_id']}: {issue['details']}")
            if len(issue_list) > 3:
                print(f"    ... and {len(issue_list) - 3} more")
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return False

def verify_fixes_summary():
    """Display summary of fixes implemented"""
    print("="*60)
    print("TRADE LOGGING FIXES VERIFICATION")
    print("="*60)
    
    print("\nFIXES IMPLEMENTED:")
    print("1. ✅ Added trade_entry_datetime field to store original entry time")
    print("2. ✅ Modified _execute_trade to use stored entry_datetime consistently")
    print("3. ✅ Fixed FLIP operation to preserve entry_datetime for CLOSE logs")
    print("4. ✅ Updated CANCEL_ACTION logging to use stored entry_datetime")
    print("5. ✅ Fixed _force_close_position_no_fees to use stored entry_datetime")
    print("6. ✅ Updated _log_trade method to use stored entry_datetime")
    print("7. ✅ Added proper initialization and reset of trade_entry_datetime")
    
    print("\nEXPECTED RESULTS AFTER FIXES:")
    print("- ✅ All trade records within same trade_id have identical entry_datetime")
    print("- ✅ CLOSED trades have correct duration calculation")
    print("- ✅ ADJUST actions show 0.0 PnL when status=OPEN")
    print("- ✅ OPEN trades have empty close_reason")
    print("- ✅ Entry timestamps reflect original trade open time")
    
    print("\nTo verify fixes:")
    print("1. Run a new training session")
    print("2. Check the generated CSV for consistency")
    print("3. All timestamp/duration issues should be resolved")

if __name__ == "__main__":
    verify_fixes_summary()
    
    # If a CSV file is provided as argument, analyze it
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"\nAnalyzing provided CSV file: {csv_file}")
        is_consistent = analyze_trade_consistency(csv_file)
        if is_consistent:
            print("✅ All trade logging appears consistent!")
        else:
            print("❌ Trade logging issues still present")
    else:
        print(f"\nTo analyze a specific CSV file, run:")
        print(f"python {sys.argv[0]} <path_to_csv_file>")
