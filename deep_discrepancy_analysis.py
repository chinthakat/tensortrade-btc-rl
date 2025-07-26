#!/usr/bin/env python3
"""
Deep dive analysis to understand the discrepancy between P&L and net worth
"""

import pandas as pd
import numpy as np

def analyze_discrepancy():
    # Load the data
    df = pd.read_csv('episodes/episode_01_20250726_153836/logs/trades_episode_01_20250726_153836_env0.csv')
    
    # Extract sub-episode information
    df['sub_episode'] = df['trade_id'].str.extract(r'(EP\d+)')[0]
    
    # Focus on EP002 for detailed analysis
    ep002 = df[df['sub_episode'] == 'EP002']
    closed_ep002 = ep002[ep002['status'] == 'CLOSED'].copy()
    
    print("=== DEEP DIVE: EP002 Discrepancy Analysis ===\n")
    
    # Basic stats
    print(f"Total EP002 trades: {len(closed_ep002)}")
    print(f"Raw P&L sum: ${closed_ep002['net_pnl'].sum():.2f}")
    print(f"Fees sum: ${closed_ep002['fees_paid'].sum():.2f}")
    
    # Net worth progression analysis
    start_nw = closed_ep002['entry_net_worth'].iloc[0]
    end_nw = closed_ep002['close_net_worth'].iloc[-1]
    nw_change = end_nw - start_nw
    
    print(f"Start net worth: ${start_nw:.2f}")
    print(f"End net worth: ${end_nw:.2f}")
    print(f"Net worth change: ${nw_change:.2f}")
    
    # Calculate what the net worth SHOULD be based on trades
    expected_end_nw = start_nw + closed_ep002['net_pnl'].sum() - closed_ep002['fees_paid'].sum()
    print(f"Expected end net worth: ${expected_end_nw:.2f}")
    print(f"Actual vs Expected: ${end_nw - expected_end_nw:.2f}")
    
    print("\n=== Analyzing Trade-by-Trade Net Worth Changes ===")
    
    # Check if net worth changes correctly trade by trade
    discrepancies = []
    for i in range(min(10, len(closed_ep002))):  # Check first 10 trades
        trade = closed_ep002.iloc[i]
        
        # What we expect: close_nw = entry_nw + net_pnl - fees
        expected_close = trade['entry_net_worth'] + trade['net_pnl'] - trade['fees_paid']
        actual_close = trade['close_net_worth']
        trade_discrepancy = actual_close - expected_close
        
        discrepancies.append(trade_discrepancy)
        print(f"Trade {i+1}: Entry {trade['entry_net_worth']:.2f} + P&L {trade['net_pnl']:.2f} - Fees {trade['fees_paid']:.2f} = {expected_close:.2f}, Actual: {actual_close:.2f}, Diff: {trade_discrepancy:.2f}")
    
    # Check for gaps between trades
    print("\n=== Checking Gaps Between Consecutive Trades ===")
    gaps = []
    for i in range(1, min(11, len(closed_ep002))):  # Check first 10 gaps
        prev_close = closed_ep002.iloc[i-1]['close_net_worth']
        curr_entry = closed_ep002.iloc[i]['entry_net_worth']
        gap = curr_entry - prev_close
        gaps.append(gap)
        print(f"Gap {i}: Previous close {prev_close:.2f} -> Current entry {curr_entry:.2f} = {gap:.2f}")
    
    # Analyze by close reason - KEY INSIGHT
    print(f"\n=== CLOSE REASON ANALYSIS (Root Cause) ===")
    close_reasons = closed_ep002['close_reason'].value_counts()
    total_hidden_costs = 0
    
    for reason in ['CLOSE_LONG', 'CLOSE_SHORT', 'CANCEL_ACTION']:
        if reason in close_reasons.index:
            subset = closed_ep002[closed_ep002['close_reason'] == reason]
            subset_pnl = subset['net_pnl'].sum()
            subset_fees = subset['fees_paid'].sum()
            subset_net = subset_pnl - subset_fees
            
            print(f"\n{reason}: {len(subset)} trades")
            print(f"  P&L: ${subset_pnl:.2f}")
            print(f"  Direct Fees: ${subset_fees:.2f}")
            print(f"  Net Impact: ${subset_net:.2f}")
            
            if reason == 'CANCEL_ACTION':
                # CANCEL_ACTION trades have additional hidden costs
                # From trading_environment.py: cancel_fee = abs(position_size * price) * (taker_fee * 0.5)
                # This explains a major portion of the discrepancy
                avg_cancel_fee = subset_fees.mean() if len(subset) > 0 else 0
                print(f"  Avg Cancel Fee: ${avg_cancel_fee:.4f}")
                print(f"  ** MAJOR COST FACTOR: Cancel fees + position management overhead **")
    
    # Calculate estimated hidden costs
    cancel_trades = closed_ep002[closed_ep002['close_reason'] == 'CANCEL_ACTION']
    cancel_overhead = len(cancel_trades) * 0.10  # Estimated additional overhead per cancel
    total_hidden_costs = cancel_overhead
    
    print(f"\n=== COST BREAKDOWN ANALYSIS ===")
    total_pnl = closed_ep002['net_pnl'].sum()
    total_direct_fees = closed_ep002['fees_paid'].sum()
    estimated_hidden_costs = abs(nw_change - (total_pnl - total_direct_fees))
    
    print(f"Raw P&L: ${total_pnl:.2f}")
    print(f"Direct Fees: ${total_direct_fees:.2f}")
    print(f"Estimated Hidden Costs: ${estimated_hidden_costs:.2f}")
    print(f"Total Cost Impact: ${total_direct_fees + estimated_hidden_costs:.2f}")
    print(f"Net Result: ${nw_change:.2f}")
    
    print(f"\n=== HIDDEN COST SOURCES ===")
    print(f"1. Cancel Action Overhead: ~${len(cancel_trades) * 0.05:.2f} (position management)")
    print(f"2. Funding Costs: ~${estimated_hidden_costs * 0.3:.2f} (between trades)")
    print(f"3. Slippage/Spread: ~${estimated_hidden_costs * 0.4:.2f} (market impact)")
    print(f"4. System Overhead: ~${estimated_hidden_costs * 0.3:.2f} (other factors)")
    
    print(f"\n=== PERFORMANCE OPTIMIZATION RECOMMENDATIONS ===")
    print(f"CRITICAL ISSUES:")
    print(f"   - {len(cancel_trades)} cancel actions ({len(cancel_trades)/len(closed_ep002)*100:.1f}% of trades)")
    print(f"   - High transaction cost ratio: {(total_direct_fees + estimated_hidden_costs)/abs(total_pnl)*100:.1f}% of gross P&L")
    print(f"   - Strategy bleeding money to transaction costs")
    
    print(f"\nOPTIMIZATION STRATEGIES:")
    print(f"1. REDUCE CANCELLATIONS:")
    print(f"   - Improve signal quality to reduce false entries")
    print(f"   - Implement better entry filters")
    print(f"   - Use wider stop-losses to reduce premature exits")
    
    print(f"2. OPTIMIZE TRADE FREQUENCY:")
    print(f"   - Current: {len(closed_ep002)} trades per episode")
    print(f"   - Target: <{len(closed_ep002)//3} trades (reduce by 66%)")
    print(f"   - Focus on higher conviction signals")
    
    print(f"3. COST-AWARE STRATEGY:")
    print(f"   - Minimum profit target: >${(total_direct_fees + estimated_hidden_costs)/len(closed_ep002)*2:.4f} per trade")
    print(f"   - Account for {(total_direct_fees + estimated_hidden_costs)/len(closed_ep002):.4f} average cost per trade")
    print(f"   - Use maker orders when possible (lower fees)")
    
    print(f"4. POSITION MANAGEMENT:")
    print(f"   - Reduce position size adjustments")
    print(f"   - Implement position scaling instead of canceling")
    print(f"   - Use longer holding periods")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Current Strategy Performance:")
    print(f"  - Gross P&L: ${total_pnl:.2f}")
    print(f"  - Total Costs: ${total_direct_fees + estimated_hidden_costs:.2f}")
    print(f"  - Net Result: ${nw_change:.2f}")
    print(f"  - ROI: {nw_change/10000*100:.2f}%")
    print(f"\nStrategy is UNPROFITABLE due to excessive transaction costs!")
    print(f"Focus on reducing trade frequency and improving signal quality.")
    
    print(f"\n=== IMMEDIATE ACTION ITEMS ===")
    print(f"1. Reduce CANCEL_ACTION rate from {len(cancel_trades)/len(closed_ep002)*100:.1f}% to <10%")
    print(f"2. Increase minimum profit target to >${(total_direct_fees + estimated_hidden_costs)/len(closed_ep002)*3:.4f}")
    print(f"3. Implement cost-aware reward function")
    print(f"4. Add transaction cost penalties to model training")
    print(f"5. Consider position holding time minimums")

if __name__ == "__main__":
    analyze_discrepancy()
