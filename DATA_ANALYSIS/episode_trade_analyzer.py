"""
Episode Trade Log Analysis Tool
==============================

This script analyzes episode trade logs to provide comprehensive insights into:
1. Action type distribution and statistics
2. Reward analysis per action type
3. Top profit/loss trades extraction
4. Anomaly detection (reward-PnL mismatches)

All analysis results are saved in the DATA_ANALYSIS folder structure.
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, track
from rich.panel import Panel
from rich import print as rprint

console = Console()

class EpisodeTradeAnalyzer:
    """Comprehensive trade log analyzer for multi-episode training"""
    
    def __init__(self, analysis_dir: str = "DATA_ANALYSIS"):
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.reports_dir = self.analysis_dir / "reports"
        self.extracts_dir = self.analysis_dir / "extracts"
        self.anomalies_dir = self.analysis_dir / "anomalies"
        self.summaries_dir = self.analysis_dir / "summaries"
        
        for dir_path in [self.reports_dir, self.extracts_dir, self.anomalies_dir, self.summaries_dir]:
            dir_path.mkdir(exist_ok=True)
        
        console.print(f"📊 Trade Analysis initialized in: {self.analysis_dir}")
    
    def find_episode_logs(self, episodes_dir: str = "episodes") -> Dict[str, List[str]]:
        """Find all trade log files across episodes"""
        episodes_path = Path(episodes_dir)
        episode_logs = {}
        
        if not episodes_path.exists():
            console.print(f"[red][ERROR] Episodes directory not found: {episodes_dir}[/red]")
            return {}
        
        for episode_dir in episodes_path.iterdir():
            if episode_dir.is_dir():
                logs_dir = episode_dir / "logs"
                if logs_dir.exists():
                    # Find CSV trade log files
                    log_files = list(logs_dir.glob("trades_*.csv"))
                    if log_files:
                        episode_logs[episode_dir.name] = [str(f) for f in log_files]
        
        console.print(f"📁 Found {len(episode_logs)} episodes with trade logs")
        return episode_logs
    
    def load_trade_data(self, log_file: str) -> Optional[pd.DataFrame]:
        """Load and clean trade data from CSV file"""
        try:
            df = pd.read_csv(log_file)
            
            # Ensure required columns exist
            required_cols = ['side', 'entry_action', 'net_pnl', 'close_reward', 'win_loss']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                console.print(f"[yellow][WARNING]  Missing columns in {log_file}: {missing_cols}[/yellow]")
                return None
            
            # Clean and standardize data
            df['net_pnl'] = pd.to_numeric(df['net_pnl'], errors='coerce').fillna(0)
            df['close_reward'] = pd.to_numeric(df['close_reward'], errors='coerce').fillna(0)
            
            # Standardize action types
            df['action_type'] = df['entry_action'].str.upper()
            
            # Extract file info
            df['log_file'] = Path(log_file).name
            df['episode'] = self._extract_episode_from_filename(log_file)
            
            return df
            
        except Exception as e:
            console.print(f"[red][ERROR] Error loading {log_file}: {str(e)}[/red]")
            return None
    
    def _extract_episode_from_filename(self, filename: str) -> str:
        """Extract episode identifier from log filename"""
        try:
            # Pattern: trades_episode_XX_YYYYMMDD_HHMMSS_envN.csv
            parts = Path(filename).stem.split('_')
            if len(parts) >= 4 and 'episode' in parts:
                episode_idx = parts.index('episode')
                if episode_idx + 1 < len(parts):
                    return f"episode_{parts[episode_idx + 1]}"
            return "unknown_episode"
        except:
            return "unknown_episode"
    
    def analyze_action_types(self, df: pd.DataFrame) -> Dict:
        """Analyze action type distribution and statistics"""
        analysis = {}
        
        total_actions = len(df)
        action_groups = df.groupby('action_type')
        
        for action_type, group in action_groups:
            count = len(group)
            percentage = (count / total_actions) * 100
            
            # Reward statistics
            rewards = group['close_reward']
            reward_stats = {
                'min_reward': float(rewards.min()),
                'max_reward': float(rewards.max()),
                'avg_reward': float(rewards.mean()),
                'percentile_95': float(np.percentile(rewards, 95)),
                'std_reward': float(rewards.std())
            }
            
            # PnL statistics
            pnl = group['net_pnl']
            pnl_stats = {
                'min_pnl': float(pnl.min()),
                'max_pnl': float(pnl.max()),
                'avg_pnl': float(pnl.mean()),
                'total_pnl': float(pnl.sum()),
                'std_pnl': float(pnl.std())
            }
            
            analysis[action_type] = {
                'count': count,
                'percentage': percentage,
                'reward_stats': reward_stats,
                'pnl_stats': pnl_stats
            }
        
        return analysis
    
    def extract_top_trades(self, df: pd.DataFrame, n_top: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract top profit and loss trades"""
        # Sort by PnL
        df_sorted = df.sort_values('net_pnl', ascending=False)
        
        # Top profits (highest PnL)
        top_profits = df_sorted.head(n_top).copy()
        
        # Top losses (lowest PnL)
        top_losses = df_sorted.tail(n_top).copy()
        
        return top_profits, top_losses
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect reward-PnL anomalies"""
        anomalies = []
        
        for idx, row in df.iterrows():
            pnl = row['net_pnl']
            reward = row['close_reward']
            
            # Define anomaly conditions
            is_anomaly = False
            anomaly_type = ""
            
            # Positive reward for loss trade
            if pnl < 0 and reward > 0:
                is_anomaly = True
                anomaly_type = "positive_reward_for_loss"
            
            # Negative reward for profit trade
            elif pnl > 0 and reward < 0:
                is_anomaly = True
                anomaly_type = "negative_reward_for_profit"
            
            # Large reward magnitude mismatch
            elif abs(pnl) > 100 and abs(reward) < 0.1:  # Large PnL, tiny reward
                is_anomaly = True
                anomaly_type = "reward_magnitude_mismatch"
            
            # Very large reward for small PnL
            elif abs(pnl) < 1 and abs(reward) > 100:
                is_anomaly = True
                anomaly_type = "excessive_reward_magnitude"
            
            if is_anomaly:
                anomaly_data = row.to_dict()
                anomaly_data['anomaly_type'] = anomaly_type
                anomaly_data['pnl_reward_ratio'] = pnl / reward if reward != 0 else float('inf')
                anomalies.append(anomaly_data)
        
        return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()
    
    def generate_episode_report(self, episode_name: str, df: pd.DataFrame) -> Dict:
        """Generate comprehensive report for a single episode"""
        report = {
            'episode_name': episode_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_trades': len(df),
            'total_pnl': float(df['net_pnl'].sum()),
            'total_reward': float(df['close_reward'].sum()),
            'avg_pnl_per_trade': float(df['net_pnl'].mean()),
            'avg_reward_per_trade': float(df['close_reward'].mean()),
            'action_analysis': self.analyze_action_types(df)
        }
        
        # Win/Loss analysis
        wins = df[df['net_pnl'] > 0]
        losses = df[df['net_pnl'] < 0]
        
        report['win_loss_stats'] = {
            'total_wins': len(wins),
            'total_losses': len(losses),
            'win_rate': len(wins) / len(df) * 100 if len(df) > 0 else 0,
            'avg_win': float(wins['net_pnl'].mean()) if len(wins) > 0 else 0,
            'avg_loss': float(losses['net_pnl'].mean()) if len(losses) > 0 else 0,
            'profit_factor': abs(wins['net_pnl'].sum() / losses['net_pnl'].sum()) if len(losses) > 0 and losses['net_pnl'].sum() != 0 else float('inf')
        }
        
        return report
    
    def save_analysis_results(self, episode_name: str, df: pd.DataFrame, report: Dict):
        """Save all analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save detailed report
        report_file = self.reports_dir / f"{episode_name}_analysis_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 2. Extract and save top trades
        top_profits, top_losses = self.extract_top_trades(df)
        
        if not top_profits.empty:
            profits_file = self.extracts_dir / f"{episode_name}_top_profits_{timestamp}.csv"
            top_profits.to_csv(profits_file, index=False)
        
        if not top_losses.empty:
            losses_file = self.extracts_dir / f"{episode_name}_top_losses_{timestamp}.csv"
            top_losses.to_csv(losses_file, index=False)
        
        # 3. Detect and save anomalies
        anomalies = self.detect_anomalies(df)
        if not anomalies.empty:
            anomalies_file = self.anomalies_dir / f"{episode_name}_anomalies_{timestamp}.csv"
            anomalies.to_csv(anomalies_file, index=False)
            console.print(f"[yellow][WARNING]  Found {len(anomalies)} anomalies in {episode_name}[/yellow]")
        
        # 4. Save summary statistics
        summary_file = self.summaries_dir / f"{episode_name}_summary_{timestamp}.csv"
        
        # Create summary DataFrame
        action_summary = []
        for action_type, stats in report['action_analysis'].items():
            action_summary.append({
                'episode': episode_name,
                'action_type': action_type,
                'count': stats['count'],
                'percentage': stats['percentage'],
                'min_reward': stats['reward_stats']['min_reward'],
                'max_reward': stats['reward_stats']['max_reward'],
                'avg_reward': stats['reward_stats']['avg_reward'],
                'percentile_95_reward': stats['reward_stats']['percentile_95'],
                'min_pnl': stats['pnl_stats']['min_pnl'],
                'max_pnl': stats['pnl_stats']['max_pnl'],
                'avg_pnl': stats['pnl_stats']['avg_pnl'],
                'total_pnl': stats['pnl_stats']['total_pnl']
            })
        
        summary_df = pd.DataFrame(action_summary)
        summary_df.to_csv(summary_file, index=False)
        
        return {
            'report_file': str(report_file),
            'profits_file': str(profits_file) if not top_profits.empty else None,
            'losses_file': str(losses_file) if not top_losses.empty else None,
            'anomalies_file': str(anomalies_file) if not anomalies.empty else None,
            'summary_file': str(summary_file)
        }
    
    def display_episode_summary(self, episode_name: str, report: Dict):
        """Display episode analysis summary"""
        console.print(f"\n[bold]📊 Episode Analysis: {episode_name}[/bold]")
        
        # Overall statistics table
        overall_table = Table(title="Overall Statistics")
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Value", style="green")
        
        overall_table.add_row("Total Trades", str(report['total_trades']))
        overall_table.add_row("Total PnL", f"{report['total_pnl']:.2f}")
        overall_table.add_row("Total Reward", f"{report['total_reward']:.2f}")
        overall_table.add_row("Avg PnL/Trade", f"{report['avg_pnl_per_trade']:.4f}")
        overall_table.add_row("Win Rate", f"{report['win_loss_stats']['win_rate']:.1f}%")
        overall_table.add_row("Profit Factor", f"{report['win_loss_stats']['profit_factor']:.2f}")
        
        console.print(overall_table)
        
        # Action type analysis table
        action_table = Table(title="Action Type Analysis")
        action_table.add_column("Action", style="cyan")
        action_table.add_column("Count", style="blue")
        action_table.add_column("Percentage", style="green")
        action_table.add_column("Avg Reward", style="yellow")
        action_table.add_column("95th %ile Reward", style="magenta")
        action_table.add_column("Avg PnL", style="red")
        
        for action_type, stats in report['action_analysis'].items():
            action_table.add_row(
                action_type,
                str(stats['count']),
                f"{stats['percentage']:.1f}%",
                f"{stats['reward_stats']['avg_reward']:.4f}",
                f"{stats['reward_stats']['percentile_95']:.4f}",
                f"{stats['pnl_stats']['avg_pnl']:.2f}"
            )
        
        console.print(action_table)
    
    def analyze_all_episodes(self):
        """Analyze all episodes found in the episodes directory"""
        console.print("[bold]🚀 Starting Comprehensive Episode Analysis[/bold]")
        
        episode_logs = self.find_episode_logs()
        
        if not episode_logs:
            console.print("[yellow]No episode logs found to analyze[/yellow]")
            return
        
        all_reports = {}
        combined_data = []
        
        for episode_name, log_files in track(episode_logs.items(), description="Analyzing episodes..."):
            console.print(f"\n[bold blue]🔍 Analyzing {episode_name}[/bold blue]")
            
            # Combine all log files for this episode
            episode_data = []
            for log_file in log_files:
                df = self.load_trade_data(log_file)
                if df is not None:
                    episode_data.append(df)
            
            if not episode_data:
                console.print(f"[yellow][WARNING]  No valid data found for {episode_name}[/yellow]")
                continue
            
            # Combine all data for this episode
            combined_df = pd.concat(episode_data, ignore_index=True)
            
            # Generate analysis report
            report = self.generate_episode_report(episode_name, combined_df)
            all_reports[episode_name] = report
            
            # Save analysis results
            saved_files = self.save_analysis_results(episode_name, combined_df, report)
            
            # Display summary
            self.display_episode_summary(episode_name, report)
            
            # Add to combined dataset
            combined_df['episode_name'] = episode_name
            combined_data.append(combined_df)
            
            console.print(f"[green][SUCCESS] Analysis saved for {episode_name}[/green]")
            for file_type, file_path in saved_files.items():
                if file_path:
                    console.print(f"   {file_type}: {Path(file_path).name}")
        
        # Generate combined analysis
        if combined_data:
            self._generate_combined_analysis(combined_data, all_reports)
        
        console.print(f"\n[bold green]🎉 Analysis Complete![/bold green]")
        console.print(f"📁 All results saved in: {self.analysis_dir}")
    
    def _generate_combined_analysis(self, combined_data: List[pd.DataFrame], all_reports: Dict):
        """Generate combined analysis across all episodes"""
        console.print("\n[bold]📈 Generating Combined Analysis[/bold]")
        
        # Combine all episode data
        all_data = pd.concat(combined_data, ignore_index=True)
        
        # Combined summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save combined dataset
        combined_file = self.analysis_dir / f"combined_trades_{timestamp}.csv"
        all_data.to_csv(combined_file, index=False)
        
        # Generate combined report
        combined_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_episodes': len(all_reports),
            'total_trades_all_episodes': len(all_data),
            'episode_summaries': {}
        }
        
        # Episode comparison table
        comparison_table = Table(title="Episode Comparison")
        comparison_table.add_column("Episode", style="cyan")
        comparison_table.add_column("Total Trades", style="blue")
        comparison_table.add_column("Total PnL", style="green")
        comparison_table.add_column("Win Rate %", style="yellow")
        comparison_table.add_column("Profit Factor", style="magenta")
        
        for episode_name, report in all_reports.items():
            comparison_table.add_row(
                episode_name,
                str(report['total_trades']),
                f"{report['total_pnl']:.2f}",
                f"{report['win_loss_stats']['win_rate']:.1f}",
                f"{report['win_loss_stats']['profit_factor']:.2f}"
            )
            
            combined_report['episode_summaries'][episode_name] = {
                'total_trades': report['total_trades'],
                'total_pnl': report['total_pnl'],
                'win_rate': report['win_loss_stats']['win_rate'],
                'profit_factor': report['win_loss_stats']['profit_factor']
            }
        
        console.print(comparison_table)
        
        # Save combined report
        combined_report_file = self.reports_dir / f"combined_analysis_{timestamp}.json"
        with open(combined_report_file, 'w') as f:
            json.dump(combined_report, f, indent=2)
        
        console.print(f"[green]📊 Combined analysis saved: {combined_report_file.name}[/green]")
        console.print(f"[green]📊 Combined dataset saved: {combined_file.name}[/green]")

def main():
    """Main analysis function"""
    console.print("[bold]📊 Episode Trade Log Analysis Tool[/bold]")
    console.print("=" * 50)
    
    # Initialize analyzer
    analyzer = EpisodeTradeAnalyzer()
    
    # Run comprehensive analysis
    analyzer.analyze_all_episodes()

if __name__ == "__main__":
    main()
