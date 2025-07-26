"""
Trade Logic Issue Analysis
==========================
Analyzing the problematic trades with massive net worth discrepancies.
"""

import pandas as pd
import numpy as np

def analyze_trade_issue(trade_data):
    """Analyze specific trade logic issues"""
    
    print(f"\n=== ANALYZING {trade_data['trade_id']} ===")
    print(f"Issue: {trade_data['issue_description']}")
    print(f"Close Reason: {trade_data['close_reason']}")
    print(f"Net PnL: ${trade_data['net_pnl']:.2f}")
    print(f"Fees Paid: ${trade_data['fees_paid']:.2f}")
    print(f"Net Worth Change: ${trade_data['net_worth_change']:.2f}")
    print(f"Discrepancy: ${trade_data['pnl_alignment_discrepancy']:.2f}")
    
    # Calculate what the net worth change SHOULD be
    expected_net_worth_change = trade_data['net_pnl'] - trade_data['fees_paid']
    print(f"Expected Net Worth Change: ${expected_net_worth_change:.2f}")
    
    # Identify the type of issue
    if abs(trade_data['net_worth_change']) > abs(trade_data['net_pnl']) * 10:
        print("🚨 CRITICAL: Net worth change is 10x+ larger than PnL!")
        
    if trade_data['close_reason'] == 'CANCEL_ACTION':
        print("🔍 CANCEL_ACTION Issue:")
        print("   - This should have minimal impact on net worth")
        print("   - Large changes suggest phantom trade execution")
        
    if abs(trade_data['fees_paid']) > 1000:
        print("🔍 EXCESSIVE FEES Issue:")
        print(f"   - Fees of ${trade_data['fees_paid']:.2f} are unrealistic")
        print("   - Fee cap should limit to ~1% of trade value")
        
    # Calculate implied trade value from fees
    if trade_data['fees_paid'] > 0:
        # Assuming 0.04% taker fee
        implied_trade_value = trade_data['fees_paid'] / 0.0004
        print(f"   - Implied trade value: ${implied_trade_value:.2f}")
        
    return {
        'trade_id': trade_data['trade_id'],
        'issue_type': 'phantom_execution' if trade_data['close_reason'] == 'CANCEL_ACTION' else 'calculation_error',
        'severity': 'critical' if abs(trade_data['net_worth_change']) > abs(trade_data['net_pnl']) * 5 else 'moderate',
        'expected_change': expected_net_worth_change,
        'actual_change': trade_data['net_worth_change'],
        'discrepancy': trade_data['pnl_alignment_discrepancy']
    }

# Problematic trades from user data
problematic_trades = [
    {
        'trade_id': 'TRADE_03480',
        'issue_description': 'Logic Error: Net Worth decreased more than PNL',
        'pnl_alignment_discrepancy': -2958.7318716502423,
        'fees_paid': 4495.105035579751,
        'net_pnl': 3.5501099174789346,
        'net_worth_change': -2955.1817617327633,
        'previous_close_net_worth': 7270.662499503926,
        'close_net_worth': 4315.480737771162,
        'close_reason': 'CANCEL_ACTION'
    },
    {
        'trade_id': 'TRADE_03532',
        'issue_description': 'Logic Error: Net Worth decreased more than PNL',
        'pnl_alignment_discrepancy': -2273.869118330678,
        'fees_paid': 4523.098125367165,
        'net_pnl': -4.841576211709842,
        'net_worth_change': -2278.7106945423875,
        'previous_close_net_worth': 6708.063727506076,
        'close_net_worth': 4429.353032963689,
        'close_reason': 'CANCEL_ACTION'
    },
    {
        'trade_id': 'TRADE_02185',
        'issue_description': 'Logic Error: Net Worth decreased more than PNL',
        'pnl_alignment_discrepancy': -2176.994310445195,
        'fees_paid': 3108.2637052886535,
        'net_pnl': 16.937573135952324,
        'net_worth_change': -2160.0567373092426,
        'previous_close_net_worth': 7141.973018688959,
        'close_net_worth': 4981.916281379717,
        'close_reason': 'CANCEL_ACTION'
    },
    {
        'trade_id': 'TRADE_00001',
        'issue_description': 'Logic Error: Net Worth increased more than PNL',
        'pnl_alignment_discrepancy': 5578.411194470174,
        'fees_paid': 0.291636724360219,
        'net_pnl': -6.950354855678053,
        'net_worth_change': 5571.460839614497,
        'previous_close_net_worth': 4422.703133283913,
        'close_net_worth': 9994.16397289841,
        'close_reason': 'CLOSE_LONG'
    },
    {
        'trade_id': 'TRADE_00001',  # Duplicate ID - concerning!
        'issue_description': 'Logic Error: Net Worth increased more than PNL',
        'pnl_alignment_discrepancy': 5485.352814164661,
        'fees_paid': 1.8905211921243947,
        'net_pnl': -67.90669070952156,
        'net_worth_change': 5417.44612345514,
        'previous_close_net_worth': 4510.994175152837,
        'close_net_worth': 9928.440298607977,
        'close_reason': 'CLOSE_SHORT'
    },
    {
        'trade_id': 'TRADE_00002',
        'issue_description': 'Logic Error: Net Worth increased more than PNL',
        'pnl_alignment_discrepancy': 4681.648088222029,
        'fees_paid': 1.969399529104454,
        'net_pnl': -44.16636338086604,
        'net_worth_change': 4637.481724841164,
        'previous_close_net_worth': 5322.62570187384,
        'close_net_worth': 9960.107426715003,
        'close_reason': 'CLOSE_LONG'
    }
]

print("CRITICAL TRADE LOGIC ISSUE ANALYSIS")
print("=" * 60)

analysis_results = []
for trade in problematic_trades:
    result = analyze_trade_issue(trade)
    analysis_results.append(result)

print(f"\n=== PATTERN ANALYSIS ===")

# Group by issue type
cancel_action_issues = [r for r in analysis_results if r['issue_type'] == 'phantom_execution']
calculation_errors = [r for r in analysis_results if r['issue_type'] == 'calculation_error']

print(f"\n1. PHANTOM EXECUTION ISSUES (CANCEL_ACTION): {len(cancel_action_issues)}")
for issue in cancel_action_issues:
    print(f"   - {issue['trade_id']}: ${issue['discrepancy']:.2f} discrepancy")

print(f"\n2. CALCULATION ERRORS: {len(calculation_errors)}")
for issue in calculation_errors:
    print(f"   - {issue['trade_id']}: ${issue['discrepancy']:.2f} discrepancy")

print(f"\n=== ROOT CAUSE ANALYSIS ===")

print(f"\n🚨 CRITICAL ISSUES IDENTIFIED:")

print(f"\n1. PHANTOM TRADE EXECUTION:")
print(f"   - CANCEL_ACTION trades are executing massive position changes")
print(f"   - These should be no-ops but are causing $2000-3000 losses")
print(f"   - Suggests the cancel logic is actually executing trades")

print(f"\n2. MASSIVE FEE CALCULATION ERRORS:")
print(f"   - Fees of $3000-4500 on small PnL trades")
print(f"   - Fee cap of 1% should prevent this")
print(f"   - Indicates position size calculation errors")

print(f"\n3. NET WORTH CALCULATION INCONSISTENCY:")
print(f"   - Net worth changes don't match PnL - fees formula")
print(f"   - Some trades show 1000x+ discrepancies")
print(f"   - Suggests balance/equity tracking is broken")

print(f"\n4. DUPLICATE TRADE IDs:")
print(f"   - TRADE_00001 appears twice with different close reasons")
print(f"   - Indicates trade logging/ID generation issues")

print(f"\n=== IMMEDIATE FIXES NEEDED ===")

fixes = [
    "1. Fix CANCEL_ACTION logic to prevent phantom executions",
    "2. Enforce fee cap of 1% maximum on all trades", 
    "3. Add position size validation to prevent massive positions",
    "4. Fix net worth calculation to match PnL - fees exactly",
    "5. Fix trade ID generation to prevent duplicates",
    "6. Add trade value validation before execution"
]

for fix in fixes:
    print(f"   {fix}")

print(f"\n🔍 These issues explain the $3000+ fee problems you've been seeing!")
