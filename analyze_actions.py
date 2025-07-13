#!/usr/bin/env python3
"""
Real-time action analysis for ongoing training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from collections import Counter

def analyze_current_training():
    """Analyze action distribution in current training"""
    
    print("🔍 Analyzing Current Training Action Distribution...")
    
    # Find latest episode
    episodes_dir = Path("episodes")
    if not episodes_dir.exists():
        print("❌ No episodes directory found")
        return
    
    episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
    if not episode_dirs:
        print("❌ No episodes found")
        return
    
    latest_episode = max(episode_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📊 Latest episode: {latest_episode.name}")
    
    # Find all CSV files in logs
    logs_dir = latest_episode / "logs"
    if not logs_dir.exists():
        print("❌ No logs directory found")
        return
    
    csv_files = list(logs_dir.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found")
        return
    
    print(f"📈 Found {len(csv_files)} CSV files")
    
    # Analyze each CSV file
    total_actions = Counter()
    total_trades = 0
    cancel_actions = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            total_trades += len(df)
            
            # Count CANCEL actions
            if 'close_reason' in df.columns:
                cancel_count = (df['close_reason'] == 'CANCEL_ACTION').sum()
                cancel_actions += cancel_count
            
            # Look for action summary entries
            if 'win_loss' in df.columns:
                action_summaries = df[df['win_loss'] == 'ACTION_SUMMARY']
                if len(action_summaries) > 0:
                    latest_summary = action_summaries.iloc[-1]
                    print(f"📊 {csv_file.name}: {latest_summary['status']}")
                    print(f"   Action breakdown: {latest_summary['entry_action']}")
                    print(f"   HOLD percentage: {latest_summary['close_reason']}")
            
            # Count entry actions
            if 'entry_action' in df.columns:
                actions = df['entry_action'].value_counts()
                for action, count in actions.items():
                    total_actions[action] += count
                    
        except Exception as e:
            print(f"⚠️  Error reading {csv_file.name}: {e}")
    
    print(f"\n📊 Overall Trading Statistics:")
    print(f"   Total logged entries: {total_trades}")
    print(f"   CANCEL actions: {cancel_actions}")
    print(f"   CANCEL percentage: {cancel_actions/total_trades*100:.1f}%")
    
    print(f"\n🎯 Top Action Types:")
    for action, count in total_actions.most_common(10):
        percentage = count / sum(total_actions.values()) * 100
        print(f"   {action}: {count} ({percentage:.1f}%)")
    
    # Look for recent action trends
    print(f"\n🔍 Action Patterns:")
    buy_sell_ratio = (total_actions.get('BUY', 0) + total_actions.get('SELL', 0)) / max(sum(total_actions.values()), 1)
    print(f"   Trading activity: {buy_sell_ratio*100:.1f}%")
    print(f"   Position management: {cancel_actions/max(total_trades, 1)*100:.1f}%")
    
    return {
        'total_trades': total_trades,
        'cancel_actions': cancel_actions,
        'action_distribution': dict(total_actions),
        'latest_episode': latest_episode.name
    }

def monitor_training(interval=30):
    """Monitor training in real-time"""
    print(f"🔄 Starting real-time monitoring (every {interval}s)...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            print(f"\n{'='*60}")
            print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
            analyze_current_training()
            print(f"{'='*60}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_training()
    else:
        analyze_current_training()
