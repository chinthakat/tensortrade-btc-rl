"""
Quick Trade Analysis Tool
========================

A simplified tool for analyzing individual trade log files or specific episodes.
Useful for quick debugging and focused analysis.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
import argparse

console = Console()

def analyze_single_file(file_path: str, output_dir: str = "DATA_ANALYSIS/quick_analysis"):
    """Analyze a single trade log file"""
    console.print(f"[bold]📊 Analyzing: {Path(file_path).name}[/bold]")
    
    try:
        df = pd.read_csv(file_path)
        
        # Basic validation
        if df.empty:
            console.print("[red][ERROR] File is empty[/red]")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Quick statistics
        console.print(f"\n[cyan]📈 Quick Statistics[/cyan]")
        console.print(f"   Total records: {len(df)}")
        console.print(f"   Date range: {df['training_step'].min()} to {df['training_step'].max()}")
        
        if 'net_pnl' in df.columns:
            total_pnl = df['net_pnl'].sum()
            console.print(f"   Total PnL: {total_pnl:.2f}")
            
            wins = len(df[df['net_pnl'] > 0])
            losses = len(df[df['net_pnl'] < 0])
            win_rate = wins / len(df) * 100 if len(df) > 0 else 0
            console.print(f"   Win Rate: {win_rate:.1f}% ({wins} wins, {losses} losses)")
        
        # Action distribution
        if 'entry_action' in df.columns:
            action_dist = df['entry_action'].value_counts()
            
            action_table = Table(title="Action Distribution")
            action_table.add_column("Action", style="cyan")
            action_table.add_column("Count", style="green")
            action_table.add_column("Percentage", style="yellow")
            
            for action, count in action_dist.items():
                percentage = count / len(df) * 100
                action_table.add_row(action, str(count), f"{percentage:.1f}%")
            
            console.print(action_table)
        
        # Quick anomaly check
        if 'net_pnl' in df.columns and 'close_reward' in df.columns:
            # Reward-PnL mismatches
            pos_reward_loss = len(df[(df['net_pnl'] < 0) & (df['close_reward'] > 0)])
            neg_reward_profit = len(df[(df['net_pnl'] > 0) & (df['close_reward'] < 0)])
            
            if pos_reward_loss > 0 or neg_reward_profit > 0:
                console.print(f"\n[yellow][WARNING]  Anomalies Detected:[/yellow]")
                console.print(f"   Positive rewards for losses: {pos_reward_loss}")
                console.print(f"   Negative rewards for profits: {neg_reward_profit}")
                
                # Save anomalies
                anomalies = df[
                    ((df['net_pnl'] < 0) & (df['close_reward'] > 0)) |
                    ((df['net_pnl'] > 0) & (df['close_reward'] < 0))
                ]
                
                if not anomalies.empty:
                    anomaly_file = Path(output_dir) / f"{Path(file_path).stem}_anomalies.csv"
                    anomalies.to_csv(anomaly_file, index=False)
                    console.print(f"   Anomalies saved to: {anomaly_file.name}")
        
        # Top trades
        if 'net_pnl' in df.columns:
            top_5_profits = df.nlargest(5, 'net_pnl')[['training_step', 'entry_action', 'net_pnl', 'close_reward']]
            top_5_losses = df.nsmallest(5, 'net_pnl')[['training_step', 'entry_action', 'net_pnl', 'close_reward']]
            
            console.print(f"\n[green]🏆 Top 5 Profits:[/green]")
            for _, row in top_5_profits.iterrows():
                console.print(f"   Step {row['training_step']}: {row['entry_action']} PnL={row['net_pnl']:.2f} Reward={row['close_reward']:.4f}")
            
            console.print(f"\n[red]📉 Top 5 Losses:[/red]")
            for _, row in top_5_losses.iterrows():
                console.print(f"   Step {row['training_step']}: {row['entry_action']} PnL={row['net_pnl']:.2f} Reward={row['close_reward']:.4f}")
        
        console.print(f"\n[green][SUCCESS] Analysis complete for {Path(file_path).name}[/green]")
        
    except Exception as e:
        console.print(f"[red][ERROR] Error analyzing file: {str(e)}[/red]")

def interactive_file_selector():
    """Interactive file selection"""
    episodes_dir = Path("episodes")
    
    if not episodes_dir.exists():
        console.print("[red][ERROR] Episodes directory not found[/red]")
        return None
    
    # Find all trade log files
    log_files = []
    for episode_dir in episodes_dir.iterdir():
        if episode_dir.is_dir():
            logs_dir = episode_dir / "logs"
            if logs_dir.exists():
                for log_file in logs_dir.glob("trades_*.csv"):
                    log_files.append(log_file)
    
    if not log_files:
        console.print("[yellow]No trade log files found[/yellow]")
        return None
    
    # Display options
    console.print("[bold]📁 Available Trade Log Files:[/bold]")
    for i, log_file in enumerate(log_files):
        episode_name = log_file.parent.parent.name
        console.print(f"   {i+1}. {episode_name}: {log_file.name}")
    
    while True:
        try:
            choice = Prompt.ask("Select file number", default="1")
            idx = int(choice) - 1
            if 0 <= idx < len(log_files):
                return str(log_files[idx])
            else:
                console.print("[red]Invalid selection[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Quick Trade Log Analysis")
    parser.add_argument("--file", "-f", type=str, help="Path to trade log CSV file")
    parser.add_argument("--output", "-o", type=str, default="DATA_ANALYSIS/quick_analysis", 
                       help="Output directory for analysis results")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive file selection mode")
    
    args = parser.parse_args()
    
    console.print("[bold][QUICK] Quick Trade Analysis Tool[/bold]")
    console.print("=" * 40)
    
    if args.interactive or not args.file:
        # Interactive mode
        file_path = interactive_file_selector()
        if not file_path:
            return
    else:
        # Direct file analysis
        file_path = args.file
        if not Path(file_path).exists():
            console.print(f"[red][ERROR] File not found: {file_path}[/red]")
            return
    
    # Run analysis
    analyze_single_file(file_path, args.output)
    
    # Ask if user wants to analyze another file
    if Confirm.ask("Analyze another file?"):
        args.file = None  # Reset for interactive mode
        main()

if __name__ == "__main__":
    main()
