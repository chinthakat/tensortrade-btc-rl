#!/usr/bin/env python3
"""
Comprehensive analysis of all cost factors affecting net worth
"""

import pandas as pd

def comprehensive_cost_analysis():
    print("=== COMPREHENSIVE COST ANALYSIS ===\n")
    
    # Load the data
    df = pd.read_csv('episodes/episode_01_20250726_153836/logs/trades_episode_01_20250726_153836_env0.csv')
    
    print("Available columns:")
    print(df.columns.tolist())
    print()
    
    # Extract sub-episode information
    df['sub_episode'] = df['trade_id'].str.extract(r'(EP\d+)')[0]
    
    # Focus on EP002 for detailed analysis
    ep002 = df[df['sub_episode'] == 'EP002']
    closed_ep002 = ep002[ep002['status'] == 'CLOSED'].copy()
    
    print(f"EP002 Analysis - {len(closed_ep002)} closed trades")
    print("=" * 50)
    
    # Basic calculations
    start_nw = closed_ep002['entry_net_worth'].iloc[0]
    end_nw = closed_ep002['close_net_worth'].iloc[-1]
    net_worth_change = end_nw - start_nw
    total_pnl = closed_ep002['net_pnl'].sum()
    total_fees = closed_ep002['fees_paid'].sum()
    
    print(f"Start Net Worth: ${start_nw:.2f}")
    print(f"End Net Worth: ${end_nw:.2f}")
    print(f"Net Worth Change: ${net_worth_change:.2f}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Total Fees Paid: ${total_fees:.2f}")
    print(f"P&L - Fees: ${total_pnl - total_fees:.2f}")
    print(f"Discrepancy: ${net_worth_change - (total_pnl - total_fees):.2f}")
    print()
    
    # Check different close reasons to identify additional costs
    print("Close Reasons Distribution:")
    close_reasons = closed_ep002['close_reason'].value_counts()
    for reason, count in close_reasons.items():
        subset = closed_ep002[closed_ep002['close_reason'] == reason]
        subset_pnl = subset['net_pnl'].sum()
        subset_fees = subset['fees_paid'].sum()
        print(f"  {reason}: {count} trades, P&L: ${subset_pnl:.2f}, Fees: ${subset_fees:.2f}")
    print()
    
    # Look for patterns in the data that might explain discrepancy
    print("Analysis of Net Worth Progression:")
    
    # Calculate what each trade's net worth change should be vs what it actually is
    total_expected_change = 0
    total_actual_change = 0
    
    for i in range(len(closed_ep002)):
        trade = closed_ep002.iloc[i]
        
        if i == 0:
            # First trade - use starting net worth
            expected_close = start_nw + trade['net_pnl'] - trade['fees_paid']
        else:
            # Use previous trade's close as this trade's expected entry
            prev_close = closed_ep002.iloc[i-1]['close_net_worth']
            expected_close = prev_close + trade['net_pnl'] - trade['fees_paid']
        
        actual_close = trade['close_net_worth']
        trade_discrepancy = actual_close - expected_close
        
        total_expected_change += trade['net_pnl'] - trade['fees_paid']
        total_actual_change += (actual_close - (closed_ep002.iloc[i-1]['close_net_worth'] if i > 0 else start_nw))
        
        if i < 5:  # Show first 5 for debugging
            print(f"Trade {i+1}: Expected close ${expected_close:.2f}, Actual ${actual_close:.2f}, Diff: ${trade_discrepancy:.2f}")
    
    print(f"\nTotal expected change: ${total_expected_change:.2f}")
    print(f"Total actual change: ${total_actual_change:.2f}")
    print(f"Overall discrepancy: ${total_actual_change - total_expected_change:.2f}")
    
    # Check for systematic patterns
    print(f"\nPotential causes of discrepancy:")
    print(f"1. Funding costs (not tracked in individual trades)")
    print(f"2. Liquidation fees (0.5% of position)")
    print(f"3. Position management overhead")
    print(f"4. Slippage or spread costs")
    print(f"5. Timestamp-based calculations differences")
    
    # Check if there are gaps between consecutive trades
    print(f"\nGaps between consecutive trades (first 10):")
    for i in range(1, min(11, len(closed_ep002))):
        prev_close = closed_ep002.iloc[i-1]['close_net_worth']
        curr_entry = closed_ep002.iloc[i]['entry_net_worth']
        gap = curr_entry - prev_close
        if abs(gap) > 0.01:  # Only show significant gaps
            print(f"Gap {i}: ${gap:.2f} (${prev_close:.2f} -> ${curr_entry:.2f})")

if __name__ == "__main__":
    comprehensive_cost_analysis()
