"""
Launch script for BTCUSDT Perpetual Futures Live Trading
This script provides an easy way to start the live trading system with proper configuration
"""

import os
import sys
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

console = Console()

def check_requirements():
    """Check if all required files and dependencies are available"""
    required_files = [
        "binance_integration.py",
        "trading_environment.py", 
        "action_space_wrapper.py",
        "improved_reward_configs.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        console.print("[red]❌ Missing required files:[/red]")
        for file in missing_files:
            console.print(f"  - {file}")
        return False
    
    return True

def check_config_file(config_path):
    """Check if configuration file exists and is valid"""
    if not os.path.exists(config_path):
        console.print(f"[yellow]⚠️ Configuration file not found: {config_path}[/yellow]")
        console.print("[cyan]💡 The system will run in paper trading mode with default settings[/cyan]")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if API keys are configured
        testnet_config = config.get("binance", {}).get("testnet", {})
        if not testnet_config.get("api_key") or not testnet_config.get("api_secret"):
            console.print("[yellow]⚠️ API keys not configured in config.json[/yellow]")
            console.print("[cyan]💡 The system will run in paper trading mode[/cyan]")
        else:
            console.print("[green]✅ Configuration file loaded successfully[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error reading configuration file: {e}[/red]")
        return False

def find_latest_model():
    """Find the latest trained model"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        return None
    
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        return None
    
    # Sort by modification time, newest first
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(model_files[0])

def display_configuration_info():
    """Display information about configuration"""
    info_text = """
🔧 Configuration Information:

The system uses config.json for secure configuration management:

1. API Keys: Stored securely in config.json (excluded from git)
2. Trading Parameters: Symbol (BTCUSDT), timeframe, balance, etc.
3. Risk Management: Position sizes, stop losses, take profits
4. Training Settings: Continuous learning parameters

To use live trading:
1. Create config.json with your Binance testnet API keys
2. Set "paper_trading": false in the trading section
3. Ensure "use_testnet": true for safe testing

The system will automatically fall back to paper trading if:
- config.json is missing
- API keys are not configured
- Connection to Binance fails
"""
    
    console.print(Panel(info_text, title="[cyan]Configuration Guide[/cyan]", border_style="cyan"))

def main():
    """Main launch function"""
    parser = argparse.ArgumentParser(description="Launch BTCUSDT Perpetual Futures Trading System")
    parser.add_argument("--model", type=str, help="Path to trained model (auto-detect if not provided)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--info", action="store_true", help="Show configuration information")
    parser.add_argument("--check", action="store_true", help="Check system requirements only")
    
    args = parser.parse_args()
    
    # Display banner
    banner = """
╔══════════════════════════════════════════════════════════════╗
║               BTCUSDT Perpetual Futures                       ║
║              Live Trading System v2.0                        ║
║                                                              ║
║  🚀 Binance Integration with Continuous Learning             ║
║  💰 BTCUSDT Perpetual Futures Focus                         ║
║  🔒 Secure Configuration Management                          ║
║  📊 Real-time Performance Monitoring                        ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    console.print(Panel(banner, border_style="green"))
    
    # Show configuration info if requested
    if args.info:
        display_configuration_info()
        return
    
    # Check system requirements
    console.print("[cyan]🔍 Checking system requirements...[/cyan]")
    if not check_requirements():
        console.print("[red]❌ System requirements not met. Please ensure all required files are present.[/red]")
        return
    
    console.print("[green]✅ System requirements check passed[/green]")
    
    if args.check:
        console.print("[green]✅ System check completed successfully[/green]")
        return
    
    # Check configuration
    console.print(f"[cyan]🔍 Checking configuration file: {args.config}[/cyan]")
    check_config_file(args.config)
    
    # Find model file
    model_path = args.model
    if not model_path:
        console.print("[cyan]🔍 Auto-detecting latest model...[/cyan]")
        model_path = find_latest_model()
        
        if not model_path:
            console.print("[red]❌ No trained model found in models/ directory[/red]")
            console.print("[yellow]💡 Please train a model first using train_model.py[/yellow]")
            return
        
        console.print(f"[green]✅ Found latest model: {model_path}[/green]")
    
    # Verify model exists
    if not os.path.exists(model_path):
        console.print(f"[red]❌ Model file not found: {model_path}[/red]")
        return
    
    # Confirm launch
    console.print(f"[cyan]📋 Launch Configuration:[/cyan]")
    console.print(f"  - Model: {model_path}")
    console.print(f"  - Config: {args.config}")
    console.print(f"  - Symbol: BTCUSDT Perpetual Futures")
    
    if not Confirm.ask("\\n[bold]Do you want to start the BTCUSDT trading system?[/bold]"):
        console.print("[yellow]⚠️ Launch cancelled by user[/yellow]")
        return
    
    # Import and run the trading system
    try:
        console.print("[green]🚀 Starting BTCUSDT Trading System...[/green]")
        
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import and run
        from binance_integration import main as run_trading_system
        import sys
        
        # Set command line arguments for the trading system
        sys.argv = ["binance_integration.py", "--model", model_path, "--config", args.config]
        
        # Run the trading system
        run_trading_system()
        
    except KeyboardInterrupt:
        console.print("\\n[yellow]⚠️ Trading system stopped by user[/yellow]")
    except ImportError as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        console.print("[yellow]💡 Please ensure all required dependencies are installed[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error starting trading system: {e}[/red]")

if __name__ == "__main__":
    main()
