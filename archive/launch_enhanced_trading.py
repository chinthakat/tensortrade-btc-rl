#!/usr/bin/env python3
"""
Enhanced BTCUSDT Perpetual Futures Trading System Launcher
Features: Periodic balance updates, comprehensive risk management, real-time monitoring
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm

# Add current directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

console = Console()

def check_requirements():
    """Check if all requirements are met"""
    console.print("[cyan]🔍 Checking system requirements...[/cyan]")
    
    requirements = []
    
    # Check if model exists
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.zip"))
        if model_files:
            latest_model = max(model_files, key=os.path.getctime)
            requirements.append(("Model File", f"✅ Found: {latest_model.name}", True))
        else:
            requirements.append(("Model File", "❌ No .zip models found in models/", False))
    else:
        requirements.append(("Model File", "❌ models/ directory not found", False))
    
    # Check config file
    config_file = Path("config.json")
    if config_file.exists():
        requirements.append(("Config File", "✅ config.json found", True))
    else:
        requirements.append(("Config File", "❌ config.json not found", False))
    
    # Check Python packages
    # Binance integration (local module)
    try:
        import binance_integration
        requirements.append(("Binance Package", "✅ local binance_integration module found", True))
    except ImportError:
        requirements.append(("Binance Package", "❌ binance_integration module not found", False))
    
    try:
        import stable_baselines3
        requirements.append(("Stable Baselines3", "✅ stable-baselines3 installed", True))
    except ImportError:
        requirements.append(("Stable Baselines3", "❌ pip install stable-baselines3", False))
    
    try:
        import rich
        requirements.append(("Rich Console", "✅ rich installed", True))
    except ImportError:
        requirements.append(("Rich Console", "❌ pip install rich", False))
    
    # Display requirements table
    req_table = Table(title="System Requirements Check")
    req_table.add_column("Component", style="cyan")
    req_table.add_column("Status", style="white")
    
    all_met = True
    for name, status, met in requirements:
        req_table.add_row(name, status)
        if not met:
            all_met = False
    
    console.print(req_table)
    return all_met

def show_configuration_info():
    """Show configuration information"""
    console.print("\n")
    config_panel = Panel.fit(
        """[bold cyan]Configuration Guide[/bold cyan]

[yellow]1. API Keys Setup:[/yellow]
   • Testnet: https://testnet.binancefuture.com/
   • Get API key and secret
   • Add to config.json under binance.testnet

[yellow]2. Risk Management:[/yellow]
   • max_position_size_pct: Max % of balance per trade (default: 10%)
   • max_open_positions: Max simultaneous positions (default: 3)
   • max_daily_loss_pct: Max daily loss % (default: 5%)
   • stop_loss_pct: Stop loss % (default: 2%)
   • take_profit_pct: Take profit % (default: 4%)

[yellow]3. Enhanced Features:[/yellow]
   • Balance updates every 30 seconds
   • Risk checks every 10 seconds  
   • Emergency position closure on risk limits
   • Consecutive loss tracking
   • Real-time risk dashboard

[yellow]4. Safety Features:[/yellow]
   • Automatic trading pause on excessive losses
   • Balance protection mechanisms
   • Position size validation
   • Daily loss limits

[green]Start with paper trading to test the system![/green]
        """,
        title="📋 Setup Information"
    )
    console.print(config_panel)

def get_latest_model():
    """Get the latest model file"""
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.zip"))
        if model_files:
            return max(model_files, key=os.path.getctime)
    return None

def main():
    parser = argparse.ArgumentParser(description="Enhanced BTCUSDT Trading System Launcher")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--check", action="store_true", help="Check system requirements")
    parser.add_argument("--info", action="store_true", help="Show configuration information")
    parser.add_argument("--force", action="store_true", help="Skip safety confirmations")
    
    args = parser.parse_args()
    
    # Show header
    console.print(Panel.fit(
        "[bold green]🚀 Enhanced BTCUSDT Perpetual Futures Trading System[/bold green]\n"
        "[cyan]Features: Real-time balance monitoring, comprehensive risk management[/cyan]",
        title="Trading System Launcher"
    ))
    
    if args.check:
        if check_requirements():
            console.print("\n[green]✅ All requirements met! System ready to launch.[/green]")
        else:
            console.print("\n[red]❌ Some requirements not met. Please install missing components.[/red]")
        return
    
    if args.info:
        show_configuration_info()
        return
    
    # Check requirements
    if not check_requirements():
        console.print("\n[red]❌ Requirements check failed. Use --check for details.[/red]")
        return
    
    # Get model file
    model_file = args.model
    if not model_file:
        latest_model = get_latest_model()
        if latest_model:
            model_file = str(latest_model)
            console.print(f"[green]📁 Auto-detected model: {latest_model.name}[/green]")
        else:
            console.print("[red]❌ No model file specified and none found. Use --model[/red]")
            return
    
    if not os.path.exists(model_file):
        console.print(f"[red]❌ Model file not found: {model_file}[/red]")
        return
    
    # Check config
    if not os.path.exists(args.config):
        console.print(f"[red]❌ Config file not found: {args.config}[/red]")
        return
    
    # Safety confirmation
    if not args.force:
        console.print("\n[yellow]⚠️ Important Safety Information:[/yellow]")
        console.print("• This system includes enhanced risk management")
        console.print("• Balance is monitored every 30 seconds")
        console.print("• Risk checks happen every 10 seconds")
        console.print("• Emergency stops will trigger on risk violations")
        console.print("• Start with paper trading to test")
        
        if not Confirm.ask("\n[bold]Do you want to continue?[/bold]"):
            console.print("[yellow]Launch cancelled by user.[/yellow]")
            return
    
    # Launch system
    console.print(f"\n[green]🚀 Launching Enhanced BTCUSDT Trading System...[/green]")
    console.print(f"Model: {model_file}")
    console.print(f"Config: {args.config}")
    
    try:
        from binance_integration import main as trading_main
        # Override sys.argv for the trading system
        sys.argv = ["binance_integration.py", "--model", model_file, "--config", args.config]
        trading_main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Trading system stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error launching trading system: {e}[/red]")

if __name__ == "__main__":
    main()
