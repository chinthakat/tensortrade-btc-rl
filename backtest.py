"""
Backtesting and Evaluation Module for Trained Trading Models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint

from stable_baselines3 import PPO, A2C, SAC
from trading_environment import FuturesTradingEnv, TradingMetrics
from model_architectures import CNNLSTMFeatureExtractor, AttentionCNNLSTMExtractor, ResNetLSTMExtractor
from action_space_wrapper import wrap_environment_for_algorithm

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)

console = Console()

class TradingBacktester:
    """Comprehensive backtesting system for trading models"""
    
    def __init__(self, model_path: str, data_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.data_path = data_path
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.env = None
        self.results = {}
        
    def _load_config(self) -> Dict:
        """Load configuration if available"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "training_params": {
                "initial_equity": 10000.0,
                "max_leverage": 25.0,
                "window_size": 60,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04
            }
        }
    
    def load_model_and_data(self):
        """Load the trained model and prepare data"""
        console.print(f"📊 Loading data from: [green]{self.data_path}[/green]")
        
        # Load data
        df = pd.read_csv(self.data_path)
        console.print(f"✅ Loaded {len(df)} data points")
        
        # Create environment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"backtest_logs/backtest_{timestamp}.csv"
        os.makedirs("backtest_logs", exist_ok=True)
        
        self.env = FuturesTradingEnv(
            df=df,
            log_file=log_file,
            use_advanced_action_space=True,  # Enable advanced action space by default
            **self.config["training_params"]
        )
        
        # Wrap environment for model compatibility
        self.env = wrap_environment_for_algorithm(self.env, "PPO")
        
        # Load model
        console.print(f"🤖 Loading model from: [green]{self.model_path}[/green]")
        
        # Determine model type from filename or config
        if "ppo" in self.model_path.lower():
            self.model = PPO.load(self.model_path)
        elif "a2c" in self.model_path.lower():
            self.model = A2C.load(self.model_path)
        elif "sac" in self.model_path.lower():
            self.model = SAC.load(self.model_path)
        else:
            # Default to PPO
            self.model = PPO.load(self.model_path)
        
        console.print("✅ Model and environment loaded successfully")
    
    def run_backtest(self, deterministic: bool = True) -> Dict:
        """Run comprehensive backtest"""
        console.print("\n[bold]🚀 Starting Backtest...[/bold]")
        
        if not self.model or not self.env:
            self.load_model_and_data()
        
        # Reset environment
        obs, info = self.env.reset()
        
        # Track results
        equity_history = []
        action_history = []
        reward_history = []
        trade_count = 0
        
        total_steps = len(self.env.price_data) - self.env.window_size - 1
        
        # Run backtest
        for step in track(range(total_steps), description="Running backtest..."):
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Execute step
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Record data
            equity_history.append(info['equity'])
            action_history.append(action[0])
            reward_history.append(reward)
            
            if info['episode_trades'] > trade_count:
                trade_count = info['episode_trades']
            
            # Check if episode ended
            if terminated or truncated:
                break
        
        # Calculate final results
        final_equity = equity_history[-1] if equity_history else self.config["training_params"]["initial_equity"]
        total_return = (final_equity - self.config["training_params"]["initial_equity"]) / self.config["training_params"]["initial_equity"] * 100
        
        self.results = {
            'initial_equity': self.config["training_params"]["initial_equity"],
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'total_trades': trade_count,
            'equity_history': equity_history,
            'action_history': action_history,
            'reward_history': reward_history,
            'max_drawdown': TradingMetrics.calculate_max_drawdown(np.array(equity_history)),
            'sharpe_ratio': self._calculate_sharpe_ratio(equity_history),
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'avg_trade_duration': self._calculate_avg_trade_duration()
        }
        
        console.print("✅ [bold green]Backtest completed![/bold green]")
        return self.results
    
    def _calculate_sharpe_ratio(self, equity_history: List[float]) -> float:
        """Calculate Sharpe ratio from equity curve"""
        if len(equity_history) < 2:
            return 0.0
        
        returns = np.diff(equity_history) / equity_history[:-1]
        return TradingMetrics.calculate_sharpe_ratio(returns)
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade logs"""
        if not self.env.logger or not self.env.logger.trades:
            return 0.0
        
        trades = self.env.logger.trades
        winning_trades = sum(1 for trade in trades if trade.get('net_pnl', 0) > 0)
        
        return winning_trades / len(trades) * 100 if trades else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.env.logger or not self.env.logger.trades:
            return 0.0
        
        trades = self.env.logger.trades
        gross_profit = sum(trade.get('net_pnl', 0) for trade in trades if trade.get('net_pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('net_pnl', 0) for trade in trades if trade.get('net_pnl', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in hours"""
        if not self.env.logger or not self.env.logger.trades:
            return 0.0
        
        trades = self.env.logger.trades
        durations = [trade.get('trade_duration_hours', 0) for trade in trades]
        
        return np.mean(durations) if durations else 0.0
    
    def display_results(self):
        """Display comprehensive backtest results"""
        if not self.results:
            console.print("[red]❌ No results to display. Run backtest first.[/red]")
            return
        
        # Main results table
        results_table = Table(title="📊 Backtest Results")
        results_table.add_column("Metric", style="cyan", no_wrap=True)
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Initial Equity", f"${self.results['initial_equity']:,.2f}")
        results_table.add_row("Final Equity", f"${self.results['final_equity']:,.2f}")
        results_table.add_row("Total Return", f"{self.results['total_return_pct']:.2f}%")
        results_table.add_row("Total Trades", str(self.results['total_trades']))
        results_table.add_row("Max Drawdown", f"{self.results['max_drawdown']*100:.2f}%")
        results_table.add_row("Sharpe Ratio", f"{self.results['sharpe_ratio']:.3f}")
        results_table.add_row("Win Rate", f"{self.results['win_rate']:.1f}%")
        results_table.add_row("Profit Factor", f"{self.results['profit_factor']:.2f}")
        results_table.add_row("Avg Trade Duration", f"{self.results['avg_trade_duration']:.1f} hours")
        
        console.print(results_table)
        
        # Performance classification
        performance_text = self._classify_performance()
        console.print(Panel(performance_text, title="Performance Analysis", border_style="blue"))
    
    def _classify_performance(self) -> str:
        """Classify strategy performance"""
        total_return = self.results['total_return_pct']
        sharpe_ratio = self.results['sharpe_ratio']
        max_drawdown = self.results['max_drawdown'] * 100
        win_rate = self.results['win_rate']
        
        if total_return > 20 and sharpe_ratio > 1.5 and max_drawdown < 15 and win_rate > 55:
            return "🌟 EXCELLENT: Outstanding performance across all metrics!"
        elif total_return > 10 and sharpe_ratio > 1.0 and max_drawdown < 25 and win_rate > 50:
            return "✅ GOOD: Solid performance with acceptable risk metrics."
        elif total_return > 0 and sharpe_ratio > 0.5 and max_drawdown < 35:
            return "⚠️  MODERATE: Positive returns but with room for improvement."
        elif total_return > -10:
            return "🔴 POOR: Strategy shows limited profitability."
        else:
            return "❌ VERY POOR: Strategy shows significant losses. Review required."
    
    def create_visualizations(self, save_path: str = "backtest_plots"):
        """Create comprehensive visualization plots"""
        if not self.results:
            console.print("[red]❌ No results to visualize. Run backtest first.[/red]")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Equity Curve
        plt.subplot(3, 2, 1)
        equity_history = self.results['equity_history']
        plt.plot(equity_history, linewidth=2, color='blue')
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line for initial equity
        plt.axhline(y=self.results['initial_equity'], color='red', linestyle='--', alpha=0.7, label='Initial Equity')
        plt.legend()
        
        # 2. Drawdown Curve
        plt.subplot(3, 2, 2)
        equity_array = np.array(equity_history)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak * 100
        
        plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown, color='darkred', linewidth=1)
        plt.title('Drawdown Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # 3. Action Distribution
        plt.subplot(3, 2, 3)
        action_history = self.results['action_history']
        plt.hist(action_history, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.title('Action Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Action (Leverage)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 4. Cumulative Reward
        plt.subplot(3, 2, 4)
        reward_history = self.results['reward_history']
        cumulative_rewards = np.cumsum(reward_history)
        plt.plot(cumulative_rewards, color='purple', linewidth=2)
        plt.title('Cumulative Reward', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Reward')
        plt.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
        plt.subplot(3, 2, 5)
        if len(equity_history) > 50:
            rolling_returns = pd.Series(equity_history).pct_change().rolling(50)
            rolling_sharpe = rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)
            plt.plot(rolling_sharpe.dropna(), color='orange', linewidth=2)
            plt.title('Rolling Sharpe Ratio (50-period)', fontsize=14, fontweight='bold')
            plt.xlabel('Time Steps')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True, alpha=0.3)
        
        # 6. Monthly Returns (if data spans multiple months)
        plt.subplot(3, 2, 6)
        if len(equity_history) > 30:
            # Simulate monthly returns
            monthly_returns = []
            month_size = len(equity_history) // 12 if len(equity_history) > 12 else len(equity_history)
            
            for i in range(0, len(equity_history), month_size):
                if i + month_size < len(equity_history):
                    start_equity = equity_history[i]
                    end_equity = equity_history[i + month_size]
                    monthly_return = (end_equity - start_equity) / start_equity * 100
                    monthly_returns.append(monthly_return)
            
            if monthly_returns:
                colors = ['green' if x > 0 else 'red' for x in monthly_returns]
                plt.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
                plt.title('Monthly Returns', fontsize=14, fontweight='bold')
                plt.xlabel('Month')
                plt.ylabel('Return (%)')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{save_path}/backtest_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"📊 Visualizations saved to: [green]{plot_filename}[/green]")
    
    def generate_report(self, output_path: str = "backtest_reports"):
        """Generate comprehensive backtest report"""
        if not self.results:
            console.print("[red]❌ No results to report. Run backtest first.[/red]")
            return
        
        os.makedirs(output_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{output_path}/backtest_report_{timestamp}.html"
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Bot Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .excellent {{ background-color: #d4edda; }}
                .good {{ background-color: #fff3cd; }}
                .poor {{ background-color: #f8d7da; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 Trading Bot Backtest Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📊 Performance Summary</h2>
            <div class="metric">
                <h3>Total Return</h3>
                <p><strong>{self.results['total_return_pct']:.2f}%</strong></p>
            </div>
            <div class="metric">
                <h3>Sharpe Ratio</h3>
                <p><strong>{self.results['sharpe_ratio']:.3f}</strong></p>
            </div>
            <div class="metric">
                <h3>Max Drawdown</h3>
                <p><strong>{self.results['max_drawdown']*100:.2f}%</strong></p>
            </div>
            <div class="metric">
                <h3>Win Rate</h3>
                <p><strong>{self.results['win_rate']:.1f}%</strong></p>
            </div>
            
            <h2>📈 Detailed Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Initial Equity</td><td>${self.results['initial_equity']:,.2f}</td></tr>
                <tr><td>Final Equity</td><td>${self.results['final_equity']:,.2f}</td></tr>
                <tr><td>Total Trades</td><td>{self.results['total_trades']}</td></tr>
                <tr><td>Profit Factor</td><td>{self.results['profit_factor']:.2f}</td></tr>
                <tr><td>Average Trade Duration</td><td>{self.results['avg_trade_duration']:.1f} hours</td></tr>
            </table>
            
            <h2>🎯 Performance Analysis</h2>
            <p>{self._classify_performance()}</p>
            
            <h2>⚙️ Configuration</h2>
            <pre>{json.dumps(self.config, indent=2, cls=NumpyEncoder)}</pre>
        </body>
        </html>
        """
        
        with open(report_filename, 'w') as f:
            f.write(html_content)
        
        console.print(f"📄 Report saved to: [green]{report_filename}[/green]")

def run_backtest_from_cli():
    """CLI interface for running backtests"""
    console.print("[bold]🎯 Trading Bot Backtest System[/bold]")
    
    # Get available models
    models_dir = Path("models")
    if not models_dir.exists():
        console.print("[red]❌ No models directory found![/red]")
        return
    
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        console.print("[red]❌ No trained models found![/red]")
        return
    
    # Display available models
    table = Table(title="Available Models")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Modified", style="yellow")
    
    for i, model_file in enumerate(model_files):
        mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
        table.add_row(str(i+1), model_file.name, mod_time.strftime("%Y-%m-%d %H:%M"))
    
    console.print(table)
    
    # Get user selection
    from rich.prompt import IntPrompt
    choice = IntPrompt.ask("Select model to backtest", default=1)
    
    if 1 <= choice <= len(model_files):
        selected_model = model_files[choice-1]
        
        # Get data file
        data_files = {}
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.rglob("*.csv"):
                data_files[file.name] = str(file)
        
        if not data_files:
            console.print("[red]❌ No data files found![/red]")
            return
        
        # Use first data file found (could be enhanced to let user choose)
        data_file = list(data_files.values())[0]
        
        console.print(f"📊 Using data file: [green]{data_file}[/green]")
        
        # Run backtest
        backtester = TradingBacktester(str(selected_model), data_file)
        results = backtester.run_backtest()
        
        # Display results
        backtester.display_results()
        
        # Create visualizations
        backtester.create_visualizations()
        
        # Generate report
        backtester.generate_report()
        
        console.print("\n[bold green]✅ Backtest completed successfully![/bold green]")

if __name__ == "__main__":
    run_backtest_from_cli()
