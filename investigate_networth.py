#!/usr/bin/env python3
"""
Script to investigate the net worth vs P&L discrepancy
"""

import pandas as pd
import os

def investigate_networth_discrepancy():
    episodes_dir = 'episodes'
    
    for episode in os.listdir(episodes_dir):
        log_path = os.path.join(episodes_dir, episode, 'logs')
        if os.path.exists(log_path):
            print(f"\n=== Analyzing Episode: {episode} ===")
            
            for log_file in os.listdir(log_path):
                if log_file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(log_path, log_file))
                    print(f'\nLog File: {log_file}')
                    print('Columns:', list(df.columns))
                    
                    # Extract sub-episode information
                    df['sub_episode'] = df['trade_id'].str.extract(r'(EP\d+)')[0]
                    
                    # Filter for EP002 data only
                    ep002_data = df[df['sub_episode'] == 'EP002']
                    completed_ep002 = ep002_data[ep002_data['status'] == 'CLOSED']
                    
                    if len(completed_ep002) > 0:
                        print(f"\nEP002 Analysis:")
                        print(f"Total EP002 trades: {len(completed_ep002)}")
                        
                        # Show first and last trades
                        first_trade = completed_ep002.iloc[0]
                        last_trade = completed_ep002.iloc[-1]
                        
                        print(f"\nFirst EP002 trade:")
                        print(f"Trade ID: {first_trade['trade_id']}")
                        print(f"Entry Net Worth: ${first_trade['entry_net_worth']:.2f}")
                        print(f"Close Net Worth: ${first_trade['close_net_worth']:.2f}")
                        print(f"Trade P&L: ${first_trade['net_pnl']:.2f}")
                        if 'fee' in df.columns:
                            print(f"Fee: ${first_trade['fee']:.4f}")
                        
                        print(f"\nLast EP002 trade:")
                        print(f"Trade ID: {last_trade['trade_id']}")
                        print(f"Entry Net Worth: ${last_trade['entry_net_worth']:.2f}")
                        print(f"Close Net Worth: ${last_trade['close_net_worth']:.2f}")
                        print(f"Trade P&L: ${last_trade['net_pnl']:.2f}")
                        if 'fee' in df.columns:
                            print(f"Fee: ${last_trade['fee']:.4f}")
                        
                        # Calculate totals
                        start_net_worth = completed_ep002['entry_net_worth'].iloc[0]
                        end_net_worth = completed_ep002['close_net_worth'].iloc[-1]
                        total_pnl = completed_ep002['net_pnl'].sum()
                        net_worth_change = end_net_worth - start_net_worth
                        
                        print(f"\nEP002 Summary:")
                        print(f"Start Net Worth: ${start_net_worth:.2f}")
                        print(f"End Net Worth: ${end_net_worth:.2f}")
                        print(f"Net Worth Change: ${net_worth_change:.2f}")
                        print(f"Sum of Trade P&Ls: ${total_pnl:.2f}")
                        print(f"Discrepancy: ${net_worth_change - total_pnl:.2f}")
                        
                        # Check if fees are included
                        if 'fee' in df.columns:
                            total_fees = completed_ep002['fee'].sum()
                            print(f"Total Fees: ${total_fees:.2f}")
                            print(f"P&L + Fees: ${total_pnl + total_fees:.2f}")
                            print(f"Discrepancy after fees: ${net_worth_change - (total_pnl + total_fees):.2f}")
                        
                        # Show some sample trades to understand the pattern
                        print(f"\nSample of EP002 trades:")
                        sample_trades = completed_ep002[['trade_id', 'net_pnl', 'entry_net_worth', 'close_net_worth']].head(5)
                        if 'fee' in df.columns:
                            sample_trades = completed_ep002[['trade_id', 'net_pnl', 'entry_net_worth', 'close_net_worth', 'fee']].head(5)
                        print(sample_trades.to_string(index=False))
                        
                    break
            break

if __name__ == "__main__":
    investigate_networth_discrepancy()
