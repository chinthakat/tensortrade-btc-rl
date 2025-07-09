"""
Main Entry Point for Binance Futures Trading Bot
Interactive CLI for all trading operations
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

console = Console()

def display_main_menu():
    """Display the main menu"""
    title_text = Text()
    title_text.append("🚀 Binance Futures Trading Bot 🚀\n", style="bold blue")
    title_text.append("Advanced Reinforcement Learning for Cryptocurrency Trading\n", style="cyan")
    title_text.append("Powered by TensorTrade, Stable-Baselines3, and PyTorch", style="green")
    
    console.print(Panel(title_text, title="Welcome", border_style="blue"))
    
    menu_table = Table(title="Main Menu", show_header=False, box=None)
    menu_table.add_column("Option", style="cyan", no_wrap=True, width=4)
    menu_table.add_column("Description", style="white")
    
    menu_options = [
        ("1", "🎯 Train New Model"),
        ("2", "🔄 Multi-Episode Training"),
        ("3", "📊 Backtest Existing Model"),
        ("4", "📈 Live Trading (Testnet/Live)"),
        ("5", "🔧 Data Preprocessing"),
        ("6", "📋 View Training History"),
        ("7", "❓ Help & Documentation"),
        ("8", "❌ Exit")
    ]
    
    for option, description in menu_options:
        menu_table.add_row(option, description)
    
    console.print(menu_table)
    console.print()

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = [
        'stable_baselines3',
        'torch',
        'pandas',
        'numpy',
        'matplotlib',
        'rich',
        'gymnasium',
        'pandas_ta'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        console.print("[red]❌ Missing required dependencies:[/red]")
        for module in missing_modules:
            console.print(f"  - {module}")
        console.print("\n[yellow]Please install requirements:[/yellow]")
        console.print("pip install -r requirements.txt")
        return False
    
    return True

def check_data_availability():
    """Check if data files are available"""
    data_dir = Path("data")
    if not data_dir.exists():
        console.print("[yellow]⚠️  Data directory not found. Creating it...[/yellow]")
        data_dir.mkdir()
        return False
    
    csv_files = list(data_dir.rglob("*.csv"))
    if not csv_files:
        console.print("[yellow]⚠️  No CSV data files found in data directory[/yellow]")
        console.print("Please add your OHLCV data files to the 'data' directory")
        return False
    
    console.print(f"✅ Found {len(csv_files)} data file(s)")
    return True

def train_new_model():
    """Launch new model training"""
    console.print("[bold]🎯 Training New Model[/bold]")
    
    try:
        from train_model import main as train_main
        train_main()
    except ImportError as e:
        console.print(f"[red]❌ Error importing training module: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Training error: {str(e)}[/red]")

def multi_episode_training():
    """Launch multi-episode training"""
    console.print("[bold]🔄 Multi-Episode Training[/bold]")
    
    try:
        from multi_episode_training import setup_multi_episode_training
        setup_multi_episode_training()
    except ImportError as e:
        console.print(f"[red]❌ Error importing multi-episode module: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Multi-episode training error: {str(e)}[/red]")

def backtest_model():
    """Launch model backtesting"""
    console.print("[bold]📊 Backtesting Model[/bold]")
    
    try:
        from backtest import run_backtest_from_cli
        run_backtest_from_cli()
    except ImportError as e:
        console.print(f"[red]❌ Error importing backtest module: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Backtest error: {str(e)}[/red]")

def live_trading():
    """Launch live trading interface"""
    console.print("[bold]📈 Live Trading[/bold]")
    
    # Risk warning
    risk_warning = """
    ⚠️  LIVE TRADING WARNING ⚠️
    
    Live trading involves significant financial risk:
    • You can lose all your invested capital
    • Cryptocurrency markets are highly volatile
    • This is experimental software
    • Always test on testnet first
    • Start with small amounts
    
    The developers are not responsible for any financial losses.
    """
    
    console.print(Panel(risk_warning, title="Risk Warning", border_style="red"))
    
    if not Confirm.ask("Do you understand and accept these risks?"):
        console.print("[yellow]Live trading cancelled[/yellow]")
        return
    
    try:
        from live_trading import setup_live_trading
        setup_live_trading()
    except ImportError as e:
        console.print(f"[red]❌ Error importing live trading module: {str(e)}[/red]")
    except Exception as e:
        console.print(f"[red]❌ Live trading error: {str(e)}[/red]")

def data_preprocessing():
    """Data preprocessing utilities"""
    console.print("[bold]🔧 Data Preprocessing[/bold]")
    
    preprocessing_options = [
        ("1", "📥 Download data from Binance"),
        ("2", "🔄 Convert data format"),
        ("3", "🧹 Clean existing data"),
        ("4", "📊 Analyze data quality"),
        ("5", "🔙 Back to main menu")
    ]
    
    table = Table(title="Preprocessing Options", show_header=False)
    table.add_column("Option", style="cyan", width=4)
    table.add_column("Description", style="white")
    
    for option, description in preprocessing_options:
        table.add_row(option, description)
    
    console.print(table)
    
    choice = IntPrompt.ask("Select preprocessing option", default=5)
    
    if choice == 1:
        download_binance_data()
    elif choice == 2:
        convert_data_format()
    elif choice == 3:
        clean_existing_data()
    elif choice == 4:
        analyze_data_quality()
    else:
        return

def download_binance_data():
    """Download data from Binance API"""
    console.print("[bold]📥 Download Binance Data[/bold]")
    
    try:
        from binance.client import Client
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Get parameters
        symbol = Prompt.ask("Symbol", default="BTCUSDT")
        interval = Prompt.ask("Interval", default="15m")
        days = IntPrompt.ask("Number of days", default=365)
        
        console.print(f"📡 Downloading {days} days of {interval} data for {symbol}...")
        
        client = Client()  # No API key needed for public data
        
        # Calculate start time
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Download data
        klines = client.get_historical_klines(
            symbol, 
            interval, 
            start_time.strftime("%d %b %Y %H:%M:%S"),
            end_time.strftime("%d %b %Y %H:%M:%S")
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Clean and format
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['timestamp'] = df['timestamp'].astype(int) // 1000
        df = df[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        # Save file
        filename = f"data/{symbol}_{interval}_{start_time.strftime('%Y-%m-%d')}_to_{end_time.strftime('%Y-%m-%d')}.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(filename, index=False)
        
        console.print(f"✅ Data saved to: [green]{filename}[/green]")
        console.print(f"📊 Downloaded {len(df)} data points")
        
    except Exception as e:
        console.print(f"[red]❌ Download failed: {str(e)}[/red]")

def convert_data_format():
    """Convert data between different formats"""
    console.print("[bold]🔄 Convert Data Format[/bold]")
    console.print("[yellow]This feature will be implemented in future versions[/yellow]")

def clean_existing_data():
    """Clean existing data files"""
    console.print("[bold]🧹 Clean Existing Data[/bold]")
    console.print("[yellow]This feature will be implemented in future versions[/yellow]")

def analyze_data_quality():
    """Analyze data quality"""
    console.print("[bold]📊 Analyze Data Quality[/bold]")
    
    try:
        import pandas as pd
        from pathlib import Path
        
        # Get data files
        data_dir = Path("data")
        csv_files = list(data_dir.rglob("*.csv"))
        
        if not csv_files:
            console.print("[red]❌ No CSV files found in data directory[/red]")
            return
        
        # Display files
        table = Table(title="Available Data Files")
        table.add_column("Index", style="cyan", width=4)
        table.add_column("File", style="green")
        table.add_column("Size", style="yellow")
        
        for i, file in enumerate(csv_files):
            size = file.stat().st_size / (1024 * 1024)  # MB
            table.add_row(str(i+1), file.name, f"{size:.2f} MB")
        
        console.print(table)
        
        choice = IntPrompt.ask("Select file to analyze", default=1)
        if 1 <= choice <= len(csv_files):
            file_path = csv_files[choice-1]
            
            # Analyze file
            console.print(f"📊 Analyzing: [green]{file_path.name}[/green]")
            
            df = pd.read_csv(file_path)
            
            # Basic info
            info_table = Table(title="Data Quality Analysis")
            info_table.add_column("Metric", style="cyan")
            info_table.add_column("Value", style="green")
            
            info_table.add_row("Total Rows", str(len(df)))
            info_table.add_row("Columns", str(len(df.columns)))
            info_table.add_row("Missing Values", str(df.isnull().sum().sum()))
            info_table.add_row("Duplicate Rows", str(df.duplicated().sum()))
            
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                info_table.add_row("Date Range", f"{df['datetime'].min()} to {df['datetime'].max()}")
            
            if 'close' in df.columns:
                info_table.add_row("Price Range", f"${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            console.print(info_table)
        
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")

def view_training_history():
    """View training history and models"""
    console.print("[bold]📋 Training History[/bold]")
    
    try:
        from multi_episode_training import EpisodeTracker
        
        tracker = EpisodeTracker()
        tracker.display_episode_summary()
        
        # Show available models
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.zip"))
            
            if model_files:
                console.print("\n[bold]Available Models:[/bold]")
                model_table = Table()
                model_table.add_column("Model", style="green")
                model_table.add_column("Size", style="yellow")
                model_table.add_column("Modified", style="blue")
                
                for model_file in model_files:
                    size = model_file.stat().st_size / (1024 * 1024)  # MB
                    mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                    model_table.add_row(
                        model_file.name,
                        f"{size:.2f} MB",
                        mod_time.strftime("%Y-%m-%d %H:%M")
                    )
                
                console.print(model_table)
        
    except Exception as e:
        console.print(f"[red]❌ Error viewing history: {str(e)}[/red]")

def show_help():
    """Show help and documentation"""
    console.print("[bold]❓ Help & Documentation[/bold]")
    
    help_text = """
    🎯 GETTING STARTED:
    1. First, ensure you have data files in the 'data' directory
    2. Train a new model using option 1
    3. Backtest your model using option 3
    4. If satisfied, proceed to live trading (start with testnet!)
    
    📊 DATA FORMAT:
    Your CSV files should contain columns: open, high, low, close, volume, timestamp
    Timestamp should be Unix timestamp in seconds
    
    🤖 MODEL ARCHITECTURES:
    - CNN-LSTM: Standard hybrid architecture
    - Attention CNN-LSTM: Enhanced with attention mechanism
    - ResNet-LSTM: ResNet-style CNN with LSTM
    
    ⚡ ALGORITHMS:
    - PPO: Proximal Policy Optimization (recommended)
    - A2C: Advantage Actor-Critic
    - SAC: Soft Actor-Critic
    
    🔧 CONFIGURATION:
    - Leverage: Up to 25x (be careful!)
    - Stop Loss: Recommended 1-3%
    - Take Profit: Recommended 2-6%
    - Risk Management: Built-in position sizing
    
    📈 LIVE TRADING:
    - Always test on Binance Testnet first
    - Start with small position sizes
    - Monitor performance closely
    - Have risk management rules
    
    ⚠️  RISKS:
    - Cryptocurrency trading involves significant risk
    - You can lose all your invested capital
    - Past performance doesn't guarantee future results
    - This is experimental software
    """
    
    console.print(Panel(help_text, title="Help & Documentation", border_style="blue"))

def main():
    """Main application loop"""
    # Clear screen and show welcome
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check data availability
    data_available = check_data_availability()
    
    while True:
        try:
            display_main_menu()
            
            if not data_available:
                console.print("[yellow]⚠️  No data files found. Consider using option 5 to download data.[/yellow]")
            
            choice = IntPrompt.ask("Select an option", default=8)
            
            if choice == 1:
                if not data_available:
                    console.print("[red]❌ No data available for training[/red]")
                    continue
                train_new_model()
            elif choice == 2:
                if not data_available:
                    console.print("[red]❌ No data available for training[/red]")
                    continue
                multi_episode_training()
            elif choice == 3:
                backtest_model()
            elif choice == 4:
                live_trading()
            elif choice == 5:
                data_preprocessing()
                data_available = check_data_availability()  # Recheck after preprocessing
            elif choice == 6:
                view_training_history()
            elif choice == 7:
                show_help()
            elif choice == 8:
                console.print("[bold green]👋 Thank you for using the Trading Bot![/bold green]")
                console.print("[yellow]⚠️  Remember: Trading involves risk. Trade responsibly![/yellow]")
                break
            else:
                console.print("[red]❌ Invalid option. Please try again.[/red]")
            
            # Pause before showing menu again
            if choice != 8:
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
                os.system('cls' if os.name == 'nt' else 'clear')
                
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
            console.print("[dim]Press Enter to continue...[/dim]")
            input()

if __name__ == "__main__":
    main()
