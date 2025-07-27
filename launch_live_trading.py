"""
Enhanced Binance Live Trading System Launcher
Features immediate startup with real historical data preloading
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def get_latest_model():
    """Find the latest trained model"""
    models_dir = Path("models")
    if not models_dir.exists():
        return None
    
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        return None
    
    # Sort by modification time
    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
    return str(latest_model)

def main():
    console.print(Panel.fit(
        "[bold blue]🚀 Enhanced Binance Live Trading System[/bold blue]\n"
        "[cyan]✨ Powered by PPO RL Agent with Real-time Data Processing ✨[/cyan]\n\n"
        "• 📡 Real-time market data integration\n"
        "• 📊 Immediate startup with historical data preloading\n"
        "• 🧠 Advanced technical indicators (MACD, RSI, ATR)\n"
        "• ⚡ Zero-delay model predictions\n"
        "• 🛡️ Robust error handling and fallback systems",
        title="Professional Trading System",
        border_style="blue"
    ))
    
    # Check for model
    latest_model = get_latest_model()
    if latest_model:
        console.print(f"[green]✅ Found latest model: {latest_model}[/green]")
        use_latest = Confirm.ask("Use this model?", default=True)
        if use_latest:
            model_path = latest_model
        else:
            model_path = Prompt.ask("Enter model path")
    else:
        console.print("[red]❌ No trained models found![/red]")
        console.print("Please train a model first using:")
        console.print("  • [cyan]python multi_episode_training.py[/cyan] (Recommended)")
        console.print("  • [cyan]python train_model.py[/cyan] (Basic training)")
        return
    
    # Trading configuration
    console.print("\n[bold]Trading Configuration[/bold]")
    
    # Symbol selection
    symbols = {
        1: "BTCUSDT", 2: "ETHUSDT", 3: "BNBUSDT", 
        4: "SOLUSDT", 5: "ADAUSDT", 6: "DOTUSDT"
    }
    console.print("\nAvailable symbols:")
    for i, sym in symbols.items():
        if sym == "BTCUSDT":
            console.print(f"{i}. {sym} [green](Recommended - Optimized for immediate startup)[/green]")
        else:
            console.print(f"{i}. {sym}")
    
    sym_choice = IntPrompt.ask("Select symbol", default=1)
    symbol = symbols.get(sym_choice, "BTCUSDT")
    
    # Timeframe
    timeframes = {
        1: "1m", 2: "5m", 3: "15m", 4: "30m", 5: "1h", 6: "4h"
    }
    console.print("\nAvailable timeframes:")
    for i, tf in timeframes.items():
        if tf == "15m":
            console.print(f"{i}. {tf} [green](Recommended - Best performance)[/green]")
        else:
            console.print(f"{i}. {tf}")
    
    tf_choice = IntPrompt.ask("Select timeframe", default=3)
    timeframe = timeframes.get(tf_choice, "15m")
    
    # Initial balance
    initial_balance = FloatPrompt.ask("Initial balance (USDT)", default=10000.0)
    
    # Trading mode
    console.print("\n[bold]Trading Mode[/bold]")
    console.print("1. 📝 [green]Paper Trading[/green] (Safe testing with virtual money)")
    console.print("2. 🔴 [yellow]Testnet Trading[/yellow] (Binance testnet with fake money)")
    console.print("3. � [red]Live Trading[/red] (Real money - Use with extreme caution!)")
    
    mode_choice = IntPrompt.ask("Select mode", default=1)
    
    if mode_choice == 1:
        trading_mode = "paper"
        mode_display = "Paper Trading"
        api_key = None
        api_secret = None
    elif mode_choice == 2:
        trading_mode = "testnet"
        mode_display = "Testnet Trading"
        console.print("\n[yellow]⚠️  Testnet requires Binance Futures Testnet API keys[/yellow]")
        console.print("Get them from: [link]https://testnet.binancefuture.com[/link]")
        api_key = Prompt.ask("Enter Testnet API Key")
        api_secret = Prompt.ask("Enter Testnet API Secret", password=True)
    else:
        trading_mode = "live"
        mode_display = "Live Trading"
        console.print("\n[red]🚨 WARNING: LIVE TRADING WITH REAL MONEY![/red]")
        console.print("[red]   This bot is experimental. Use at your own risk![/red]")
        if not Confirm.ask("[red]Do you really want to use live trading?[/red]", default=False):
            console.print("[yellow]Switching to testnet mode for safety[/yellow]")
            trading_mode = "testnet"
            mode_display = "Testnet Trading"
        
        api_key = Prompt.ask("Enter Live API Key")
        api_secret = Prompt.ask("Enter Live API Secret", password=True)
    
    # Advanced options
    console.print("\n[bold]🔧 Advanced Options[/bold]")
    preload_history = Confirm.ask("Preload historical data for immediate startup?", default=True)
    verbose_logging = Confirm.ask("Enable verbose logging?", default=False)
    
    # Summary
    console.print("\n[bold]Configuration Summary[/bold]")
    summary_table = Table()
    summary_table.add_column("Setting", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Model", Path(model_path).name)
    summary_table.add_row("Symbol", symbol)
    summary_table.add_row("Timeframe", timeframe)
    summary_table.add_row("Initial Balance", f"${initial_balance:,.2f}")
    summary_table.add_row("Mode", mode_display)
    summary_table.add_row("Preload Data", "✅ Yes" if preload_history else "❌ No")
    summary_table.add_row("Verbose Logs", "✅ Yes" if verbose_logging else "❌ No")
    
    console.print(summary_table)
    
    # Confirm and launch
    if Confirm.ask("\nStart trading system?", default=True):
        console.print("\n[green]🚀 Launching Enhanced Trading System...[/green]")
        
        # Launch async trading system
        try:
            asyncio.run(launch_trading_system(
                model_path=model_path,
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=initial_balance,
                trading_mode=trading_mode,
                api_key=api_key,
                api_secret=api_secret,
                preload_history=preload_history,
                verbose_logging=verbose_logging
            ))
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Trading interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Error launching trading system: {e}[/red]")
    else:
        console.print("[yellow]Trading system launch cancelled[/yellow]")

async def launch_trading_system(model_path, symbol, timeframe, initial_balance, 
                               trading_mode, api_key, api_secret, preload_history, verbose_logging):
    """Launch the async trading system with enhanced features"""
    
    try:
        from binance_integration import LiveBTCUSDTTradingSystem
        from improved_reward_configs import TREND_RIDER_CONFIG
        
        console.print("[cyan]� Initializing trading system components...[/cyan]")
        
        # Create trading system
        trading_system = LiveBTCUSDTTradingSystem(
            model_path=model_path,
            config_path="config.json",
            reward_config=TREND_RIDER_CONFIG
        )
        
        console.print("[green]✅ Trading system initialized successfully![/green]")
        
        if preload_history:
            console.print("[blue]📊 Preloading historical data for immediate startup...[/blue]")
        
        console.print("[blue]🔄 Starting trading operations...[/blue]")
        
        # Start trading
        await trading_system.start()
        
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("Make sure all required modules are installed")
    except Exception as e:
        console.print(f"[red]❌ Trading system error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()