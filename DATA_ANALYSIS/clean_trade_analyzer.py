#!/usr/bin/env python3
"""
Clean Trade Statistics Analyzer
==============================

Generates clean action-level statistics focusing on actual trades,
filtering out debug logs and action summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

class CleanTradeStatsAnalyzer:
    """Clean trade statistics analyzer"""
    
    def __init__(self, analysis_dir: str = "DATA_ANALYSIS"):
        self.analysis_dir = Path(analysis_dir)
        self.clean_stats_dir = self.analysis_dir / "clean_stats"
        self.clean_stats_dir.mkdir(exist_ok=True)
    
    def load_and_clean_trade_data(self, trade_file: str) -> pd.DataFrame:
        """Load trade data and filter out non-trade entries"""
        try:
            df = pd.read_csv(trade_file)
            
            # Filter out debug entries and action summaries
            clean_df = df[
                # Exclude debug entries
                ~df['entry_action'].str.contains('DEBUG|ACTION_STATS|ACTION_CALLED|ACTION_SUMMARY', na=False) &
                # Exclude entries with win_loss = 'ACTION_DEBUG' or similar
                ~df['win_loss'].str.contains('ACTION_DEBUG|ACTION_SUMMARY|SKIP', na=False) &
                # Exclude entries where entry_action contains "BUY:XX SELL:XX"
                ~df['entry_action'].str.contains(r'BUY:\d+.*SELL:\d+', na=False, regex=True) &
                # Only include actual trade actions
                df['entry_action'].isin(['BUY', 'SELL', 'OPEN', 'CLOSE', 'ADJUST', 'FLIP'])
            ].copy()
            
            # Convert numeric columns
            numeric_cols = ['net_pnl', 'close_reward', 'trade_duration_hours', 'position_size', 'fees_paid']
            for col in numeric_cols:
                if col in clean_df.columns:
                    clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            
            return clean_df
            
        except Exception as e:
            print(f"Error loading {trade_file}: {e}")
            return pd.DataFrame()
    
    def calculate_action_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate detailed statistics by action type"""
        if df.empty:
            return {}
        
        stats = {}
        
        # Group by action type
        for action_type in df['entry_action'].unique():
            action_df = df[df['entry_action'] == action_type].copy()
            
            if len(action_df) == 0:
                continue
            
            # Basic counts
            count = len(action_df)
            percentage = (count / len(df)) * 100
            
            # P&L statistics
            pnl_values = action_df['net_pnl'].dropna()
            total_pnl = pnl_values.sum() if len(pnl_values) > 0 else 0
            avg_pnl = pnl_values.mean() if len(pnl_values) > 0 else 0
            min_pnl = pnl_values.min() if len(pnl_values) > 0 else 0
            max_pnl = pnl_values.max() if len(pnl_values) > 0 else 0
            std_pnl = pnl_values.std() if len(pnl_values) > 1 else 0
            p95_pnl = pnl_values.quantile(0.95) if len(pnl_values) > 0 else 0
            p05_pnl = pnl_values.quantile(0.05) if len(pnl_values) > 0 else 0
            
            # Reward statistics
            reward_values = action_df['close_reward'].dropna()
            total_reward = reward_values.sum() if len(reward_values) > 0 else 0
            avg_reward = reward_values.mean() if len(reward_values) > 0 else 0
            min_reward = reward_values.min() if len(reward_values) > 0 else 0
            max_reward = reward_values.max() if len(reward_values) > 0 else 0
            std_reward = reward_values.std() if len(reward_values) > 1 else 0
            p95_reward = reward_values.quantile(0.95) if len(reward_values) > 0 else 0
            p05_reward = reward_values.quantile(0.05) if len(reward_values) > 0 else 0
            
            # Win/Loss statistics
            winning_trades = pnl_values[pnl_values > 0]
            losing_trades = pnl_values[pnl_values < 0]
            neutral_trades = pnl_values[pnl_values == 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            neutral_count = len(neutral_trades)
            win_rate = (win_count / len(pnl_values)) * 100 if len(pnl_values) > 0 else 0
            
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
            
            # Risk metrics
            profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() != 0 else float('inf') if len(winning_trades) > 0 else 0
            
            # Duration statistics
            duration_values = action_df['trade_duration_hours'].dropna()
            avg_duration = duration_values.mean() if len(duration_values) > 0 else 0
            min_duration = duration_values.min() if len(duration_values) > 0 else 0
            max_duration = duration_values.max() if len(duration_values) > 0 else 0
            
            stats[action_type] = {
                # Count and percentage
                'count': count,
                'percentage': percentage,
                
                # P&L statistics
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'min_pnl': min_pnl,
                'max_pnl': max_pnl,
                'std_pnl': std_pnl,
                'p95_pnl': p95_pnl,
                'p05_pnl': p05_pnl,
                
                # Reward statistics
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'min_reward': min_reward,
                'max_reward': max_reward,
                'std_reward': std_reward,
                'p95_reward': p95_reward,
                'p05_reward': p05_reward,
                
                # Win/Loss statistics
                'win_count': win_count,
                'loss_count': loss_count,
                'neutral_count': neutral_count,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                
                # Duration statistics
                'avg_duration_hours': avg_duration,
                'min_duration_hours': min_duration,
                'max_duration_hours': max_duration
            }
        
        return stats
    
    def generate_clean_summary_table(self, stats: Dict) -> pd.DataFrame:
        """Generate a clean summary table for display"""
        if not stats:
            return pd.DataFrame()
        
        rows = []
        for action_type, action_stats in stats.items():
            rows.append({
                'Action Type': action_type,
                'Count': f"{action_stats['count']:,}",
                'Percentage': f"{action_stats['percentage']:.1f}%",
                'Total P&L': f"${action_stats['total_pnl']:,.2f}",
                'Avg P&L': f"${action_stats['avg_pnl']:.2f}",
                'Min P&L': f"${action_stats['min_pnl']:.2f}",
                'Max P&L': f"${action_stats['max_pnl']:.2f}",
                'P95 P&L': f"${action_stats['p95_pnl']:.2f}",
                'Win Rate': f"{action_stats['win_rate']:.1f}%",
                'Avg Reward': f"{action_stats['avg_reward']:.6f}",
                'Min Reward': f"{action_stats['min_reward']:.6f}",
                'Max Reward': f"{action_stats['max_reward']:.6f}",
                'P95 Reward': f"{action_stats['p95_reward']:.6f}",
                'Profit Factor': f"{action_stats['profit_factor']:.2f}" if action_stats['profit_factor'] != float('inf') else "∞"
            })
        
        return pd.DataFrame(rows)
    
    def analyze_combined_trades(self, combined_file: str = None) -> Dict:
        """Analyze combined trades file and generate clean statistics"""
        if combined_file is None:
            # Find the most recent combined trades file
            pattern = "combined_trades_*.csv"
            files = list(self.analysis_dir.glob(pattern))
            
            # Also check current directory if we're running from DATA_ANALYSIS
            if not files:
                files = list(Path(".").glob(pattern))
            
            if not files:
                print("No combined trades file found")
                return {}
            
            combined_file = max(files, key=lambda x: x.stat().st_mtime)
        
        print(f"Analyzing: {combined_file}")
        
        # Load and clean data
        df = self.load_and_clean_trade_data(combined_file)
        
        if df.empty:
            print("No clean trade data found after filtering")
            return {}
        
        print(f"Clean trades found: {len(df)}")
        print(f"Action types: {df['entry_action'].value_counts().to_dict()}")
        
        # Calculate statistics
        stats = self.calculate_action_statistics(df)
        
        # Generate summary table
        summary_table = self.generate_clean_summary_table(stats)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed stats JSON
        stats_file = self.clean_stats_dir / f"clean_action_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_stats = {}
            for action, action_stats in stats.items():
                json_stats[action] = {}
                for key, value in action_stats.items():
                    if isinstance(value, (np.integer, np.floating)):
                        json_stats[action][key] = float(value)
                    else:
                        json_stats[action][key] = value
            
            json.dump(json_stats, f, indent=2)
        
        # Save summary CSV
        summary_file = self.clean_stats_dir / f"clean_action_summary_{timestamp}.csv"
        summary_table.to_csv(summary_file, index=False)
        
        print(f"\n[SUCCESS] Clean statistics saved:")
        print(f"   [STATS] Detailed stats: {stats_file.name}")
        print(f"   [SUMMARY] Summary table: {summary_file.name}")
        
        return {
            'stats': stats,
            'summary_table': summary_table,
            'clean_data': df,
            'stats_file': str(stats_file),
            'summary_file': str(summary_file)
        }
    
    def display_clean_summary(self, stats: Dict):
        """Display clean summary statistics"""
        if not stats:
            print("No statistics to display")
            return
        
        print("\n" + "="*80)
        print("CLEAN ACTION TYPE STATISTICS")
        print("="*80)
        
        for action_type, action_stats in stats.items():
            print(f"\n[ACTION] {action_type.upper()}")
            print("-" * 40)
            print(f"Count: {action_stats['count']:,} ({action_stats['percentage']:.1f}%)")
            print(f"P&L: ${action_stats['total_pnl']:,.2f} total, ${action_stats['avg_pnl']:.2f} avg")
            print(f"P&L Range: ${action_stats['min_pnl']:.2f} to ${action_stats['max_pnl']:.2f}")
            print(f"Reward: {action_stats['avg_reward']:.6f} avg, {action_stats['min_reward']:.6f} to {action_stats['max_reward']:.6f}")
            print(f"Win Rate: {action_stats['win_rate']:.1f}% ({action_stats['win_count']} wins, {action_stats['loss_count']} losses)")
            
            if action_stats['profit_factor'] != float('inf'):
                print(f"Profit Factor: {action_stats['profit_factor']:.2f}")
            else:
                print("Profit Factor: ∞ (no losses)")


def main():
    """Main function for standalone execution"""
    analyzer = CleanTradeStatsAnalyzer()
    
    # Analyze the most recent combined trades file
    results = analyzer.analyze_combined_trades()
    
    if results:
        # Display summary
        analyzer.display_clean_summary(results['stats'])
        
        # Show summary table
        print("\n" + "="*120)
        print("SUMMARY TABLE")
        print("="*120)
        print(results['summary_table'].to_string(index=False))
    else:
        print("No analysis results to display")


if __name__ == "__main__":
    main()
