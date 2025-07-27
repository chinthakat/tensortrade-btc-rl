"""
Enhanced BTCUSDT Live Trading Launch Script
Features immediate startup with real Binance historical data
Now supports 15m and 5m timeframes
"""

import asyncio
import sys
import json
import os
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm, IntPrompt
from rich.panel import Panel
from rich.table import Table

console = Console()

def get_timeframe_config():
    """Get timeframe configuration from user"""
    
    console.print("\n[bold cyan]⏰ Select Trading Timeframe:[/bold cyan]")
    
    # Create timeframe options table
    timeframe_table = Table(show_header=True, header_style="bold magenta")
    timeframe_table.add_column("Option", style="cyan", width=8)
    timeframe_table.add_column("Timeframe", style="green", width=12)
    timeframe_table.add_column("Strategy", style="yellow", width=20)
    timeframe_table.add_column("Data History", style="blue", width=15)
    
    timeframe_table.add_row("1", "5 minutes", "High-frequency scalping", "5 hours")
    timeframe_table.add_row("2", "15 minutes", "Medium-term swing", "20 hours")
    
    console.print(timeframe_table)
    
    console.print("\n[dim]💡 Recommendations:[/dim]")
    console.print("[dim]• 5m: Better for quick reactions, more trades, higher frequency[/dim]")
    console.print("[dim]• 15m: Better for trend following, fewer but larger moves[/dim]")
    
    choice = IntPrompt.ask(
        "\n[bold yellow]Select timeframe[/bold yellow]",
        choices=["1", "2"],
        default=2
    )
    
    if choice == 1:
        return {
            "interval": "5m",
            "interval_seconds": 300,  # 5 minutes
            "display_name": "5 minutes",
            "history_periods": 60,    # 5 hours of data
            "strategy_type": "scalping"
        }
    else:
        return {
            "interval": "15m", 
            "interval_seconds": 900,  # 15 minutes
            "display_name": "15 minutes",
            "history_periods": 80,    # 20 hours of data
            "strategy_type": "swing"
        }

async def launch_immediate_trading():
    """Launch BTCUSDT trading with immediate startup using real historical data"""
    
    console.print(Panel.fit(
        "[bold green]🚀 BTCUSDT Perpetual Futures Trading System[/bold green]\n"
        "[cyan]✨ Enhanced with Multi-Timeframe Support ✨[/cyan]\n\n"
        "• Choose between 5m and 15m timeframes\n"
        "• Downloads real historical data for immediate startup\n"
        "• Calculates technical indicators immediately\n"
        "• Model predictions ready from first moment\n"
        "• No waiting period for data accumulation",
        title="Live Trading System",
        border_style="green"
    ))
    
    # Get timeframe configuration
    timeframe_config = get_timeframe_config()
    
    # Check requirements
    model_path = "models/best_model.zip"
    config_path = "config.json"
    
    if not os.path.exists(model_path):
        console.print(f"[red]❌ Model file not found: {model_path}[/red]")
        console.print("[cyan]💡 Please ensure you have a trained model in the models/ directory[/cyan]")
        return False
    
    if not os.path.exists(config_path):
        console.print(f"[yellow]⚠️ Configuration file not found: {config_path}[/yellow]")
        console.print("[cyan]💡 The system will run in paper trading mode with default settings[/cyan]")
    
    # Import after path check
    try:
        from binance_integration import LiveBTCUSDTTradingSystem
        from improved_reward_configs import TREND_RIDER_CONFIG
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        return False
    
    # Show configuration summary
    console.print("\n[bold cyan]📋 Trading Configuration Summary:[/bold cyan]")
    
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Setting", style="cyan", width=25)
    config_table.add_column("Value", style="green")
    
    config_table.add_row("📈 Symbol:", "BTCUSDT Perpetual Futures")
    config_table.add_row("⏰ Timeframe:", timeframe_config["display_name"])
    config_table.add_row("🎯 Strategy Type:", timeframe_config["strategy_type"].title())
    config_table.add_row("📊 Data Source:", "Real-time Binance Futures API")
    config_table.add_row("📥 Historical Data:", f"{timeframe_config['history_periods']} periods")
    config_table.add_row("🤖 Model Type:", "PPO trained on market microstructure")
    config_table.add_row("⚡ Update Frequency:", f"Every {timeframe_config['interval_seconds']} seconds")
    
    console.print(config_table)
    
    # Show strategy-specific info
    if timeframe_config["strategy_type"] == "scalping":
        console.print("\n[yellow]⚡ 5-Minute Scalping Mode:[/yellow]")
        console.print("• High-frequency trading with quick entries/exits")
        console.print("• More sensitive to short-term price movements")
        console.print("• Requires faster decision making and execution")
    else:
        console.print("\n[blue]📈 15-Minute Swing Mode:[/blue]")
        console.print("• Medium-term trend following strategy")
        console.print("• Better for capturing larger price movements")
        console.print("• More stable signals with less noise")
    
    # Confirm start
    if not Confirm.ask(f"\n[bold yellow]🔥 Start {timeframe_config['display_name']} trading system?[/bold yellow]", default=True):
        console.print("[yellow]👋 Trading cancelled by user[/yellow]")
        return False
    
    try:
        # Initialize trading system with selected timeframe
        console.print(f"\n[bold green]🚀 Initializing {timeframe_config['display_name']} BTCUSDT Trading System...[/bold green]")
        
        trading_system = LiveBTCUSDTTradingSystem(
            model_path=model_path,
            config_path=config_path,
            reward_config=TREND_RIDER_CONFIG
        )
        
        # Configure for selected timeframe
        trading_system.timeframe = timeframe_config["interval"]
        trading_system.history_periods = timeframe_config["history_periods"]
        trading_system.update_interval = timeframe_config["interval_seconds"]
        
        console.print("[cyan]✅ Trading system initialized[/cyan]")
        
        # Start the system with selected timeframe
        console.print(f"[bold cyan]🎯 Starting {timeframe_config['display_name']} live trading...[/bold cyan]")
        console.print(f"[dim]💡 New candles will arrive every {timeframe_config['interval_seconds']} seconds[/dim]")
        
        await trading_system.start()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Trading interrupted by user[/yellow]")
        console.print("[cyan]👋 Shutting down gracefully...[/cyan]")
        return True
    except Exception as e:
        console.print(f"\n[red]❌ Error during trading: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    console.print("[bold blue]🎯 Multi-Timeframe BTCUSDT Live Trading Launcher[/bold blue]")
    
    try:
        # Run the async trading system
        asyncio.run(launch_immediate_trading())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Fatal error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
