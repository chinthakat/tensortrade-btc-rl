#!/usr/bin/env python3
"""
Trade Verification Script
Analyzes the problematic trades to identify discrepancies between PnL and net worth changes.
"""

import pandas as pd
import numpy as np

def analyze_trade_discrepancy(trade_data):
    """Analyze discrepancies in trade logic"""
    print(f"\n=== ANALYZING TRADE {trade_data['trade_id']} ===")
    print(f"Close Reason: {trade_data['close_reason']}")
    print(f"Net PnL: ${trade_data['net_pnl']:.2f}")
    print(f"Fees Paid: ${trade_data['fees_paid']:.2f}")
    print(f"Entry Net Worth: ${trade_data['entry_net_worth']:.2f}")
    print(f"Close Net Worth: ${trade_data['close_net_worth']:.2f}")
    
    # Calculate expected vs actual net worth change
    actual_net_worth_change = trade_data['close_net_worth'] - trade_data['entry_net_worth']
    expected_net_worth_change = trade_data['net_pnl'] - trade_data['fees_paid']
    
    print(f"Actual Net Worth Change: ${actual_net_worth_change:.2f}")
    print(f"Expected Net Worth Change: ${expected_net_worth_change:.2f}")
    
    discrepancy = actual_net_worth_change - expected_net_worth_change
    print(f"Discrepancy: ${discrepancy:.2f}")
    
    # Analysis
    if abs(discrepancy) > 0.01:  # More than 1 cent discrepancy
        print("❌ SIGNIFICANT DISCREPANCY DETECTED!")
        
        if abs(trade_data['net_pnl']) < 0.01 and trade_data['fees_paid'] > 0:
            print("🔍 POTENTIAL ZERO PNL ISSUE: High fees with near-zero PnL")
        
        if trade_data['close_reason'] == 'CANCEL_ACTION':
            print("🔍 CANCEL ACTION: This might be a phantom trade issue")
            
        if abs(discrepancy) > abs(trade_data['net_pnl']):
            print("🔍 DISCREPANCY LARGER THAN PNL: Possible fee calculation error")
    else:
        print("✅ Trade accounting appears correct")
    
    return {
        'trade_id': trade_data['trade_id'],
        'discrepancy': discrepancy,
        'net_worth_change': actual_net_worth_change,
        'expected_change': expected_net_worth_change,
        'fees_paid': trade_data['fees_paid'],
        'net_pnl': trade_data['net_pnl'],
        'close_reason': trade_data['close_reason']
    }

# Problematic trades from user's data
problematic_trades = [
    {
        'trade_id': 'TRADE_02571',
        'net_pnl': 0,
        'fees_paid': 3200.585755158958,
        'entry_net_worth': 10232.186418848236,
        'close_net_worth': 5598.3490279963635,
        'close_reason': 'CANCEL_ACTION'
    },
    {
        'trade_id': 'TRADE_02622',
        'net_pnl': 0,
        'fees_paid': 3283.7413062696346,
        'entry_net_worth': 10129.35128107694,
        'close_net_worth': 5736.94139387426,
        'close_reason': 'CANCEL_ACTION'
    },
    {
        'trade_id': 'TRADE_03056',
        'net_pnl': 0,
        'fees_paid': 3640.813259244156,
        'entry_net_worth': 9769.690560277511,
        'close_net_worth': 5474.209102808785,
        'close_reason': 'CANCEL_ACTION'
    },
    {
        'trade_id': 'TRADE_00441',
        'net_pnl': -63.689682936035695,
        'fees_paid': 782.5127510131373,
        'entry_net_worth': 5639.550785082717,
        'close_net_worth': 10162.827175580554,
        'close_reason': 'CANCEL_ACTION'
    },
    {
        'trade_id': 'TRADE_00005',
        'net_pnl': 14.084609711818269,
        'fees_paid': 0.5687270452897858,
        'entry_net_worth': 5385.879703467878,
        'close_net_worth': 9985.833076637276,
        'close_reason': 'CLOSE_SHORT'
    },
    {
        'trade_id': 'TRADE_00438',
        'net_pnl': 12.01636199732161,
        'fees_paid': 778.4301942522084,
        'entry_net_worth': 5738.717731798401,
        'close_net_worth': 10228.455999956526,
        'close_reason': 'CANCEL_ACTION'
    }
]

print("TRADE VERIFICATION ANALYSIS")
print("=" * 50)

results = []
for trade in problematic_trades:
    result = analyze_trade_discrepancy(trade)
    results.append(result)

print(f"\n=== SUMMARY ===")
print("Issues Identified:")

zero_pnl_high_fees = [r for r in results if abs(r['net_pnl']) < 0.01 and r['fees_paid'] > 100]
if zero_pnl_high_fees:
    print(f"1. ZERO PNL + HIGH FEES: {len(zero_pnl_high_fees)} trades")
    for trade in zero_pnl_high_fees:
        print(f"   - {trade['trade_id']}: ${trade['fees_paid']:.2f} fees on ${trade['net_pnl']:.2f} PnL")

large_discrepancies = [r for r in results if abs(r['discrepancy']) > 1000]
if large_discrepancies:
    print(f"2. LARGE DISCREPANCIES (>$1000): {len(large_discrepancies)} trades")
    for trade in large_discrepancies:
        print(f"   - {trade['trade_id']}: ${trade['discrepancy']:.2f} discrepancy")

cancel_action_issues = [r for r in results if r['close_reason'] == 'CANCEL_ACTION']
if cancel_action_issues:
    print(f"3. CANCEL ACTION ISSUES: {len(cancel_action_issues)} trades")
    for trade in cancel_action_issues:
        print(f"   - {trade['trade_id']}: ${trade['discrepancy']:.2f} discrepancy")

print(f"\nRecommendations:")
print("1. Investigate the CANCEL_ACTION trades - they appear to be phantom trades")
print("2. Review fee calculation logic for trades with zero PnL but high fees")
print("3. Check if entry/exit net worth is being tracked correctly across trade sequences")
print("4. Verify that the emergency fee cap (1% max) is being applied correctly")
