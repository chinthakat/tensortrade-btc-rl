#!/usr/bin/env python3
"""
Enhanced Trade Trace Analyzer
============================

Advanced trade log analysis for reinforcement learning trading episodes.
Analyzes trade trace files with detailed verification and performance metrics.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
from datetime import datetime

# Initialize Rich Console for beautiful output
console = Console()

class EnhancedTradeAnalyzer:
    """
    Advanced analyzer for trade trace CSV files from RL trading episodes.
    Provides detailed verification, performance metrics, and anomaly detection.
    """

    def __init__(self, file_path: str, initial_equity: float = 10000.0, taker_fee_rate: float = 0.0004):
        """
        Initializes the enhanced analyzer.

        Args:
            file_path (str): The path to the trade log CSV file.
            initial_equity (float): The starting equity for the simulation.
            taker_fee_rate (float): The fee rate for taker orders.
        """
        self.file_path = Path(file_path)
        self.initial_equity = initial_equity
        self.taker_fee_rate = taker_fee_rate
        self.df = self._load_and_prepare_data()
        self.analysis_results = {}
        
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Loads and prepares the trade log data from the CSV file."""
        try:
            if not self.file_path.exists():
                console.print(f"[bold red]Error: File not found at '{self.file_path}'[/bold red]")
                return pd.DataFrame()
            
            df = pd.read_csv(self.file_path)
            console.print(f"✅ Successfully loaded [bold green]{self.file_path.name}[/bold green] with {len(df)} records.")

            # Clean and prepare data
            df['entry_datetime'] = pd.to_datetime(df['entry_datetime'], unit='s', errors='coerce')
            df['close_datetime'] = pd.to_datetime(df['close_datetime'], unit='s', errors='coerce')
            
            # Sort by training step to ensure chronological order
            df = df.sort_values(['training_step', 'trade_id']).reset_index(drop=True)
            
            # Fill missing values appropriately
            df['close_price'] = df['close_price'].fillna(0)
            df['trade_duration_hours'] = df['trade_duration_hours'].fillna(0)
            df['fees_paid'] = df['fees_paid'].fillna(0)
            
            return df
            
        except Exception as e:
            console.print(f"[bold red]An error occurred while loading the data: {e}[/bold red]")
            return pd.DataFrame()

    def verify_calculations(self) -> pd.DataFrame:
        """
        Enhanced verification of net_pnl and close_net_worth calculations.
        Includes action-specific validation for OPEN, CLOSE, FLIP, ADJUST actions.
        """
        if self.df.empty:
            return pd.DataFrame()

        console.print("\n[bold blue]🕵️  Verifying Profit/Loss and Net Worth Calculations...[/bold blue]")

        results = []
        verification_issues = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing trades...", total=len(self.df))
            
            for i, row in self.df.iterrows():
                progress.update(task, advance=1)
                
                verification = {
                    'training_step': row['training_step'],
                    'trade_id': row['trade_id'],
                    'entry_action': row['entry_action'],
                    'side': row['side'],
                    'status': row['status'],
                    'original_pnl': row['net_pnl'],
                    'original_close_net_worth': row['close_net_worth'],
                    'original_entry_net_worth': row['entry_net_worth'],
                }

                # Verify based on action type
                if row['entry_action'] in ['CLOSE', 'FLIP'] and row['status'] == 'CLOSED':
                    # Find the opening trade for this trade_id
                    opening_trades = self.df[
                        (self.df['trade_id'] == row['trade_id']) & 
                        (self.df['entry_action'].isin(['OPEN', 'FLIP'])) &
                        (self.df['training_step'] < row['training_step'])
                    ]
                    
                    if not opening_trades.empty:
                        # Get the most recent opening trade
                        entry_row = opening_trades.iloc[-1]
                        
                        position_size = abs(row['position_size'])
                        entry_price = entry_row['entry_price']
                        close_price = row['close_price'] if row['close_price'] > 0 else row['entry_price']
                        side = row['side']

                        # Recalculate PnL based on side
                        if side == 'LONG':
                            raw_pnl = (close_price - entry_price) * position_size
                        else:  # SHORT
                            raw_pnl = (entry_price - close_price) * position_size

                        # Calculate fees (entry + exit)
                        entry_value = entry_price * position_size
                        close_value = close_price * position_size
                        total_fees = (entry_value + close_value) * self.taker_fee_rate

                        calculated_net_pnl = raw_pnl - total_fees
                        
                        # Net worth should be previous net worth + PnL
                        expected_close_net_worth = row['entry_net_worth'] + calculated_net_pnl

                        verification['calculated_pnl'] = calculated_net_pnl
                        verification['calculated_close_net_worth'] = expected_close_net_worth
                        verification['pnl_discrepancy'] = abs(calculated_net_pnl - row['net_pnl'])
                        verification['net_worth_discrepancy'] = abs(expected_close_net_worth - row['close_net_worth'])
                        verification['raw_pnl'] = raw_pnl
                        verification['calculated_fees'] = total_fees
                        
                        # Flag significant discrepancies
                        if verification['pnl_discrepancy'] > 0.01 or verification['net_worth_discrepancy'] > 0.01:
                            verification_issues.append(verification)
                    else:
                        verification['error'] = 'No matching opening trade found'
                        verification_issues.append(verification)
                        
                elif row['entry_action'] == 'ADJUST':
                    # For ADJUST actions, net worth should remain the same
                    verification['calculated_pnl'] = 0
                    verification['calculated_close_net_worth'] = row['entry_net_worth']
                    verification['pnl_discrepancy'] = abs(row['net_pnl'])
                    verification['net_worth_discrepancy'] = abs(row['close_net_worth'] - row['entry_net_worth'])
                    
                elif row['entry_action'] == 'OPEN':
                    # For OPEN actions, PnL should be 0 and net worth should include fees
                    verification['calculated_pnl'] = 0
                    verification['calculated_close_net_worth'] = row['entry_net_worth']
                    verification['pnl_discrepancy'] = abs(row['net_pnl'])
                    verification['net_worth_discrepancy'] = abs(row['close_net_worth'] - row['entry_net_worth'])

                results.append(verification)

        verification_df = pd.DataFrame(results)
        
        # Store issues for reporting
        self.analysis_results['verification_issues'] = verification_issues
        self.analysis_results['total_discrepancies'] = len(verification_issues)
        
        return verification_df

    def analyze_action_performance(self):
        """Analyzes performance by action type (OPEN, CLOSE, FLIP, ADJUST)."""
        if self.df.empty:
            return

        console.print("\n[bold blue]⚡ Analyzing Performance by Action Type...[/bold blue]")

        action_stats = []
        
        for action in ['OPEN', 'CLOSE', 'FLIP', 'ADJUST']:
            action_data = self.df[self.df['entry_action'] == action]
            
            if len(action_data) == 0:
                continue
                
            stats = {
                'action': action,
                'count': len(action_data),
                'percentage': (len(action_data) / len(self.df)) * 100,
                'total_pnl': action_data['net_pnl'].sum(),
                'avg_pnl': action_data['net_pnl'].mean(),
                'avg_reward': action_data['close_reward'].mean(),
                'fees_paid': action_data['fees_paid'].sum(),  # Fees for this action type
            }
            
            # Win/Loss analysis for completed trades
            completed = action_data[action_data['status'] == 'CLOSED']
            if len(completed) > 0:
                wins = completed[completed['win_loss'] == 'WIN']
                losses = completed[completed['win_loss'] == 'LOSS']
                
                stats['completed_trades'] = len(completed)
                stats['win_rate'] = (len(wins) / len(completed)) * 100 if len(completed) > 0 else 0
                stats['avg_win'] = wins['net_pnl'].mean() if len(wins) > 0 else 0
                stats['avg_loss'] = losses['net_pnl'].mean() if len(losses) > 0 else 0
                
                # Profit factor for this action
                gross_profit = wins['net_pnl'].sum() if len(wins) > 0 else 0
                gross_loss = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 0.001
                stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            action_stats.append(stats)

        # Display results
        table = Table(title="Performance by Action Type")
        table.add_column("Action", style="cyan", no_wrap=True)
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        table.add_column("Total PnL", style="magenta")
        table.add_column("Avg PnL", style="blue")
        table.add_column("Win Rate", style="green")
        table.add_column("Profit Factor", style="red")

        for stats in action_stats:
            table.add_row(
                stats['action'],
                str(stats['count']),
                f"{stats['percentage']:.1f}%",
                f"${stats['total_pnl']:.2f}",
                f"${stats['avg_pnl']:.2f}",
                f"{stats.get('win_rate', 0):.1f}%" if 'win_rate' in stats else "N/A",
                f"{stats.get('profit_factor', 0):.2f}" if 'profit_factor' in stats else "N/A"
            )

        console.print(table)
        self.analysis_results['action_performance'] = action_stats

    def analyze_overall_performance(self):
        """
        Calculates and displays comprehensive performance metrics.
        """
        if self.df.empty:
            return

        console.print("\n[bold blue]📈 Analyzing Overall Trading Performance...[/bold blue]")

        # Filter for completed trades only
        closed_trades = self.df[self.df['status'] == 'CLOSED'].copy()
        
        if closed_trades.empty:
            console.print("[yellow]No closed trades found to analyze.[/yellow]")
            return

        # Basic metrics
        total_trades = len(closed_trades)
        winning_trades = closed_trades[closed_trades['win_loss'] == 'WIN']
        losing_trades = closed_trades[closed_trades['win_loss'] == 'LOSS']

        # Performance calculations
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = closed_trades['net_pnl'].sum()
        
        # Calculate actual fees paid by taking the final total_fees value
        # The 'fees_paid' column contains cumulative fees, not per-trade fees
        if len(self.df) > 0:
            # Get the maximum total_fees value as the final cumulative fee
            total_fees = self.df['fees_paid'].max()
        else:
            total_fees = 0
        
        # Profit metrics
        gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average metrics
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Risk metrics
        reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Equity curve analysis
        equity_data = self.df['close_net_worth'].dropna()
        if len(equity_data) > 0:
            final_equity = equity_data.iloc[-1]
            peak_equity = equity_data.max()
            total_return_pct = ((final_equity - self.initial_equity) / self.initial_equity) * 100
            
            # Drawdown calculation
            peak = equity_data.cummax()
            drawdown = (equity_data - peak) / peak
            max_drawdown = drawdown.min() * 100
            
            # Sharpe ratio (simplified)
            if len(closed_trades) > 1:
                returns = closed_trades['net_pnl'] / closed_trades['entry_net_worth']
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            final_equity = self.initial_equity
            total_return_pct = 0
            max_drawdown = 0
            sharpe_ratio = 0

        # Store results
        performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'final_equity': final_equity,
            'total_return_pct': total_return_pct,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
        self.analysis_results['performance_metrics'] = performance_metrics

        # Display results
        summary_panel = Panel(
            Text(f"Final Equity: ${final_equity:,.2f}\n"
                 f"Total Return: {total_return_pct:.2f}%\n"
                 f"Total Net PnL: ${total_pnl:,.2f}\n"
                 f"Total Fees: ${total_fees:,.2f}", justify="center"),
            title="📊 Overall Results",
            border_style="bold green"
        )
        console.print(summary_panel)

        # Main metrics table
        table = Table(title="🎯 Key Performance Indicators")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Total Closed Trades", f"{total_trades:,}")
        table.add_row("Winning Trades", f"{len(winning_trades):,}")
        table.add_row("Losing Trades", f"{len(losing_trades):,}")
        table.add_row("Win Rate", f"{win_rate:.2f}%")
        table.add_row("Profit Factor", f"{profit_factor:.2f}")
        table.add_row("Sharpe Ratio (Annualized)", f"{sharpe_ratio:.3f}")
        table.add_row("Max Drawdown", f"{max_drawdown:.2f}%")
        table.add_row("Average Trade PnL", f"${avg_trade_pnl:,.2f}")
        table.add_row("Average Winning Trade", f"${avg_win:,.2f}")
        table.add_row("Average Losing Trade", f"${avg_loss:,.2f}")
        table.add_row("Reward/Risk Ratio", f"{reward_risk_ratio:.2f}")

        console.print(table)

    def detect_anomalies(self):
        """Detects anomalies in the trade data."""
        console.print("\n[bold blue]🔍 Detecting Trade Anomalies...[/bold blue]")
        
        anomalies = []
        
        # Check for extreme PnL values
        if len(self.df) > 0:
            pnl_std = self.df['net_pnl'].std()
            pnl_mean = self.df['net_pnl'].mean()
            extreme_pnl = self.df[abs(self.df['net_pnl']) > abs(pnl_mean) + 3 * pnl_std]
            
            for _, row in extreme_pnl.iterrows():
                anomalies.append({
                    'type': 'Extreme PnL',
                    'trade_id': self._standardize_trade_id(row['trade_id']),
                    'training_step': row['training_step'],
                    'value': row['net_pnl'],
                    'description': f"PnL of ${row['net_pnl']:.2f} is {abs(row['net_pnl'] - pnl_mean) / pnl_std:.1f}σ from mean"
                })

        # Check for trades with zero position size
        zero_position = self.df[self.df['position_size'] == 0]
        for _, row in zero_position.iterrows():
            anomalies.append({
                'type': 'Zero Position',
                'trade_id': self._standardize_trade_id(row['trade_id']),
                'training_step': row['training_step'],
                'value': 0,
                'description': "Trade with zero position size"
            })

        # Check for missing close prices on closed trades
        closed_no_price = self.df[(self.df['status'] == 'CLOSED') & (self.df['close_price'] <= 0)]
        for _, row in closed_no_price.iterrows():
            anomalies.append({
                'type': 'Missing Close Price',
                'trade_id': self._standardize_trade_id(row['trade_id']),
                'training_step': row['training_step'],
                'value': row['close_price'],
                'description': "Closed trade without close price"
            })

        self.analysis_results['anomalies'] = anomalies
        
        if anomalies:
            console.print(f"[yellow]⚠️  Found {len(anomalies)} anomalies[/yellow]")
            
            anomaly_table = Table(title="🚨 Detected Anomalies")
            anomaly_table.add_column("Type", style="red")
            anomaly_table.add_column("Trade ID", style="yellow")
            anomaly_table.add_column("Step", style="cyan")
            anomaly_table.add_column("Description", style="white")
            
            for anomaly in anomalies[:10]:  # Show first 10
                anomaly_table.add_row(
                    anomaly['type'],
                    str(anomaly['trade_id']),
                    str(anomaly['training_step']),
                    anomaly['description']
                )
            
            console.print(anomaly_table)
            
            if len(anomalies) > 10:
                console.print(f"[dim]... and {len(anomalies) - 10} more anomalies[/dim]")
        else:
            console.print("✅ [bold green]No anomalies detected![/bold green]")

    def save_analysis_report(self, output_dir: str = "DATA_ANALYSIS/reports"):
        """Saves detailed analysis report to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_analysis_{self.file_path.stem}_{timestamp}.json"
        report_file = output_path / filename
        
        # Add metadata
        self.analysis_results['metadata'] = {
            'file_analyzed': str(self.file_path),
            'analysis_timestamp': timestamp,
            'total_records': len(self.df),
            'initial_equity': self.initial_equity,
            'taker_fee_rate': self.taker_fee_rate
        }
        
        with open(report_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        console.print(f"[green]📄 Analysis report saved: {report_file.name}[/green]")

    def run_enhanced_analysis(self, save_report: bool = True):
        """Runs the complete enhanced analysis pipeline."""
        if self.df.empty:
            console.print("[red]❌ No data to analyze[/red]")
            return

        console.print(f"\n[bold cyan]🚀 Enhanced Trade Analysis for {self.file_path.name}[/bold cyan]")
        console.print("=" * 70)

        # Run all analysis components
        verification_df = self.verify_calculations()
        
        # Report verification results
        if 'verification_issues' in self.analysis_results:
            issues = self.analysis_results['verification_issues']
            if len(issues) == 0:
                console.print("✅ [bold green]All calculations verified successfully![/bold green]")
            else:
                console.print(f"[yellow]⚠️  Found {len(issues)} calculation discrepancies[/yellow]")

        # Run performance analyses
        self.analyze_action_performance()
        self.analyze_overall_performance()
        self.detect_anomalies()

        # Save report
        if save_report:
            self.save_analysis_report()

        console.print("\n[bold green]🎉 Enhanced analysis complete![/bold green]")

    def _standardize_trade_id(self, trade_id) -> str:
        """
        Standardizes trade ID format for consistent display.
        Converts both numeric and string trade IDs to consistent format.
        """
        if isinstance(trade_id, (int, float)):
            return f"TRADE_{int(trade_id):05d}"
        elif isinstance(trade_id, str):
            if trade_id.startswith('TRADE_'):
                return trade_id
            else:
                try:
                    # Try to convert string number to standardized format
                    return f"TRADE_{int(trade_id):05d}"
                except ValueError:
                    return str(trade_id)
        else:
            return str(trade_id)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Enhanced trade trace analyzer for RL trading episodes.")
    parser.add_argument("file_path", type=str, help="Path to the trade log CSV file.")
    parser.add_argument("--initial_equity", type=float, default=10000.0, help="Initial equity for the simulation.")
    parser.add_argument("--fee_rate", type=float, default=0.0004, help="Taker fee rate.")
    parser.add_argument("--no-save", action="store_true", help="Don't save analysis report.")
    
    args = parser.parse_args()

    analyzer = EnhancedTradeAnalyzer(
        file_path=args.file_path,
        initial_equity=args.initial_equity,
        taker_fee_rate=args.fee_rate
    )
    
    analyzer.run_enhanced_analysis(save_report=not args.no_save)


if __name__ == '__main__':
    main()
