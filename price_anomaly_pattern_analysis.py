#!/usr/bin/env python3
"""
Price Anomaly Pattern Analysis
=============================

Analyzes the systematic pattern discovered in price anomalies
where trades execute using previous timestep prices.
"""

import pandas as pd
import numpy as np

def analyze_timing_pattern():
    """Analyze the timing pattern in price anomalies"""
    
    print("🔍 PRICE ANOMALY PATTERN ANALYSIS")
    print("=" * 50)
    
    # Key findings from debug session
    anomalies = [
        {
            'trade_id': 'TRADE_02468',
            'entry_time': '2024-03-24 02:15:00',
            'trade_price': 15772.29,
            'market_price_at_entry': 15252.65,
            'env_price_prev_step': 15639.92,
            'env_timestamp_prev': '2024-03-24 02:00:00'
        },
        {
            'trade_id': 'TRADE_02625', 
            'entry_time': '2024-03-29 16:15:00',
            'trade_price': 12193.82,
            'market_price_at_entry': 12601.46,
            'env_price_prev_step': 12244.21,
            'env_timestamp_prev': '2024-03-29 16:00:00'
        },
        {
            'trade_id': 'TRADE_02444',
            'entry_time': '2024-03-24 02:15:00', 
            'trade_price': 15740.96,
            'market_price_at_entry': 15252.65,
            'env_price_prev_step': 15639.92,
            'env_timestamp_prev': '2024-03-24 02:00:00'
        }
    ]
    
    print("\n📊 ANOMALY PATTERN ANALYSIS:")
    print("-" * 50)
    
    for i, anomaly in enumerate(anomalies, 1):
        print(f"\nAnomaly {i}: {anomaly['trade_id']}")
        
        # Calculate differences
        trade_vs_market = abs(anomaly['trade_price'] - anomaly['market_price_at_entry']) / anomaly['market_price_at_entry'] * 100
        trade_vs_env_prev = abs(anomaly['trade_price'] - anomaly['env_price_prev_step']) / anomaly['env_price_prev_step'] * 100
        
        print(f"  Entry Time: {anomaly['entry_time']}")
        print(f"  Trade Price: ${anomaly['trade_price']:,.2f}")
        print(f"  Market Price (at entry): ${anomaly['market_price_at_entry']:,.2f}")
        print(f"  Env Price (prev step): ${anomaly['env_price_prev_step']:,.2f}")
        print(f"  Env Timestamp (prev): {anomaly['env_timestamp_prev']}")
        print(f"  Trade vs Market: {trade_vs_market:.2f}% difference")
        print(f"  Trade vs Env(prev): {trade_vs_env_prev:.2f}% difference")
        
        if trade_vs_env_prev < 1.0:
            print(f"  ✅ Trade price matches PREVIOUS environment step!")
        else:
            print(f"  ⚠️  Trade price doesn't match previous step")
    
    print(f"\n🎯 ROOT CAUSE IDENTIFIED:")
    print("=" * 50)
    print("The trading environment is executing trades using PREVIOUS timestep prices!")
    print("This explains the systematic price anomalies we've been observing.")
    print("\n📋 TECHNICAL EXPLANATION:")
    print("- Environment steps through data with 15-minute intervals")
    print("- When action is taken at step N, it uses price from step N-1")
    print("- This creates a 15-minute lag in price execution")
    print("- Result: Trades appear to use 'future' information from analysis perspective")
    
    print(f"\n🔧 RECOMMENDED FIXES:")
    print("-" * 30)
    print("1. Ensure trade execution uses CURRENT step price")
    print("2. Add proper timestamp validation in _execute_action()")
    print("3. Verify current_step indexing in price lookup methods")
    print("4. Update price validation to account for step timing")

if __name__ == "__main__":
    analyze_timing_pattern()
