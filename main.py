"""
Main Entry Point for Binance Futures Trading Bot
Interactive CLI for all trading operations
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import log archiver
from log_archiver import archive_startup_logs

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
        ("7", "🗂️ Archive Old Logs"),
        ("8", "❓ Help & Documentation"),
        ("9", "❌ Exit")
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
    
    # Create menu for risk acceptance
    risk_table = Table(title="Live Trading Options")
    risk_table.add_column("Option", style="cyan", no_wrap=True, width=4)
    risk_table.add_column("Description", style="white")
    
    risk_options = [
        ("1", "✅ I understand and accept the risks - Proceed with Live Trading"),
        ("2", "❌ Cancel - Return to Main Menu")
    ]
    
    for option, description in risk_options:
        risk_table.add_row(option, description)
    
    console.print(risk_table)
    
    choice = IntPrompt.ask("\nSelect option", default=2)
    
    if choice == 1:
        try:
            from live_trading import setup_live_trading
            setup_live_trading()
        except ImportError as e:
            console.print(f"[red]❌ Error importing live trading module: {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]❌ Live trading error: {str(e)}[/red]")
    else:
        console.print("[yellow]Live trading cancelled[/yellow]")
        return

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

def archive_data_analysis_subfolders():
    """Archive DATA_ANALYSIS subfolders to keep only source code"""
    try:
        from pathlib import Path
        import shutil
        from datetime import datetime
        
        data_analysis_dir = Path("DATA_ANALYSIS")
        archive_dir = Path("archive")
        
        if not data_analysis_dir.exists():
            console.print("[yellow]DATA_ANALYSIS directory not found[/yellow]")
            return True
        
        # Create archive directory if it doesn't exist
        archive_dir.mkdir(exist_ok=True)
        
        # Get current timestamp for archive name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"data_analysis_subfolders_{timestamp}.zip"
        archive_path = archive_dir / archive_name
        
        # Find subfolders to archive (ignore Python files and markdown)
        subfolders_to_archive = []
        files_to_archive = []
        
        for item in data_analysis_dir.iterdir():
            if item.is_dir():
                # Archive all subfolders
                subfolders_to_archive.append(item)
            elif item.is_file():
                # Archive non-source files (CSV, JSON, etc. in root)
                if item.suffix.lower() in ['.csv', '.json', '.log', '.tmp', '.cache']:
                    files_to_archive.append(item)
        
        if not subfolders_to_archive and not files_to_archive:
            console.print("[yellow]No DATA_ANALYSIS subfolders or data files to archive[/yellow]")
            return True
        
        # Create the archive
        import zipfile
        archived_count = 0
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Archive subfolders
            for subfolder in subfolders_to_archive:
                for file_path in subfolder.rglob('*'):
                    if file_path.is_file():
                        # Calculate relative path from DATA_ANALYSIS
                        rel_path = file_path.relative_to(data_analysis_dir)
                        zipf.write(file_path, f"DATA_ANALYSIS/{rel_path}")
                        archived_count += 1
            
            # Archive data files in root
            for file_path in files_to_archive:
                rel_path = file_path.relative_to(data_analysis_dir)
                zipf.write(file_path, f"DATA_ANALYSIS/{rel_path}")
                archived_count += 1
        
        if archived_count > 0:
            # Remove archived subfolders and files
            for subfolder in subfolders_to_archive:
                if subfolder.exists():
                    shutil.rmtree(subfolder)
                    console.print(f"  🗂️ Archived and removed: {subfolder.name}/")
            
            for file_path in files_to_archive:
                if file_path.exists():
                    file_path.unlink()
                    console.print(f"  📄 Archived and removed: {file_path.name}")
            
            # Show archive info
            archive_size = archive_path.stat().st_size / 1024  # KB
            console.print(f"[green]✅ Created archive: {archive_name}[/green]")
            console.print(f"[green]   📦 Size: {archive_size:.1f} KB | Files: {archived_count}[/green]")
            
            return True
        else:
            # Remove empty archive if no files were added
            if archive_path.exists():
                archive_path.unlink()
            console.print("[yellow]No files found to archive[/yellow]")
            return True
            
    except Exception as e:
        console.print(f"[red]❌ Error archiving DATA_ANALYSIS subfolders: {str(e)}[/red]")
        return False

def archive_logs_menu():
    """Interactive log archiving menu"""
    console.print("\n[bold]🗂️ Log Archiving Options[/bold]")
    
    # Show current log status
    from pathlib import Path
    logs_dir = Path("logs")
    models_dir = Path("models") 
    tensorboard_dir = Path("tensorboard_logs")
    archive_dir = Path("archive")
    data_analysis_dir = Path("DATA_ANALYSIS")
    
    # Count current files
    log_files = list(logs_dir.glob("*.csv")) + list(logs_dir.glob("*.log")) + list(logs_dir.glob("*.npz")) if logs_dir.exists() else []
    model_files = list(models_dir.glob("*.zip")) + list(models_dir.glob("*.pkl")) if models_dir.exists() else []
    tb_dirs = [d for d in tensorboard_dir.iterdir() if d.is_dir()] if tensorboard_dir.exists() else []
    archive_files = list(archive_dir.glob("*.zip")) if archive_dir.exists() else []
    
    # Count episode logs
    episodes_dir = Path("episodes")
    episode_log_files = []
    episode_dirs = []
    if episodes_dir.exists():
        episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
        for episode_dir in episode_dirs:
            episode_logs_dir = episode_dir / "logs"
            if episode_logs_dir.exists():
                for pattern in ["*.csv", "*.log", "*.npz"]:
                    episode_log_files.extend(episode_logs_dir.glob(pattern))
    
    # Count DATA_ANALYSIS files and subfolders
    data_analysis_files = []
    data_analysis_subfolders = []
    if data_analysis_dir.exists():
        # Count subfolders that contain files
        data_analysis_subfolders = [d for d in data_analysis_dir.iterdir() if d.is_dir()]
        for subfolder in data_analysis_subfolders:
            for pattern in ["*.csv", "*.json", "*.pdf", "*.png", "*.log"]:
                data_analysis_files.extend(subfolder.rglob(pattern))
    
    # Display current status
    status_table = Table(title="Current File Status")
    status_table.add_column("Type", style="cyan")
    status_table.add_column("Count", style="green")
    status_table.add_column("Location", style="yellow")
    
    status_table.add_row("Log Files", str(len(log_files)), "logs/")
    status_table.add_row("Episode Logs", str(len(episode_log_files)), f"episodes/ ({len(episode_dirs)} dirs)")
    status_table.add_row("Model Files", str(len(model_files)), "models/")
    status_table.add_row("TensorBoard Dirs", str(len(tb_dirs)), "tensorboard_logs/")
    status_table.add_row("Analysis Files", str(len(data_analysis_files)), f"DATA_ANALYSIS/ ({len(data_analysis_subfolders)} subdirs)")
    status_table.add_row("Archives", str(len(archive_files)), "archive/")
    
    console.print(status_table)
    
    # Archive options
    archive_table = Table(title="Archive Options")
    archive_table.add_column("Option", style="cyan", no_wrap=True, width=4)
    archive_table.add_column("Description", style="white")
    
    archive_options = [
        ("1", "🗂️ Archive All (Automatic Settings + Analysis)"),
        ("2", "📊 Archive Logs Only (includes episode logs)"),
        ("3", "🤖 Archive Models Only"),
        ("4", "📈 Archive TensorBoard Only"),
        ("5", "🔬 Archive DATA_ANALYSIS Subfolders Only"),
        ("6", "⚡ Quick Archive (1 day old files)"),
        ("7", "🗑️ Archive Everything (All files including episodes & analysis)"),
        ("8", "📋 View Archive Contents"),
        ("9", "❌ Cancel"),
        ("10", "🔙 Back to Main Menu")
    ]
    
    for option, description in archive_options:
        archive_table.add_row(option, description)
    
    console.print(archive_table)
    
    choice = IntPrompt.ask("\nSelect archive option", default=10)
    
    try:
        if choice == 1:
            # Standard archiving with config settings + DATA_ANALYSIS
            console.print("[bold]🗂️ Running standard archiving (including DATA_ANALYSIS)...[/bold]")
            success = archive_startup_logs(base_dir=".")
            # Also archive DATA_ANALYSIS subfolders
            success_analysis = archive_data_analysis_subfolders()
            if success and success_analysis:
                console.print("[green]✅ Archiving completed successfully![/green]")
            else:
                console.print("[red]❌ Archiving had some issues[/red]")
                
        elif choice == 2:
            # Archive logs only
            console.print("[bold]📊 Archiving logs only...[/bold]")
            from log_archiver import LogArchiver
            archiver = LogArchiver(".")
            success = archiver.archive_logs()
            if success:
                console.print("[green]✅ Log archiving completed![/green]")
            
        elif choice == 3:
            # Archive models only
            console.print("[bold]🤖 Archiving models only...[/bold]")
            from log_archiver import LogArchiver
            archiver = LogArchiver(".")
            success = archiver.archive_old_models()
            if success:
                console.print("[green]✅ Model archiving completed![/green]")
            
        elif choice == 4:
            # Archive TensorBoard only
            console.print("[bold]📈 Archiving TensorBoard logs only...[/bold]")
            from log_archiver import LogArchiver
            archiver = LogArchiver(".")
            success = archiver.archive_tensorboard_logs()
            if success:
                console.print("[green]✅ TensorBoard archiving completed![/green]")
            
        elif choice == 5:
            # Archive DATA_ANALYSIS subfolders only
            console.print("[bold]🔬 Archiving DATA_ANALYSIS subfolders only...[/bold]")
            success = archive_data_analysis_subfolders()
            if success:
                console.print("[green]✅ DATA_ANALYSIS archiving completed![/green]")
            else:
                console.print("[red]❌ DATA_ANALYSIS archiving failed![/red]")
            
        elif choice == 6:
            # Quick archive (1 day old)
            console.print("[bold]⚡ Quick archive (1 day old files)...[/bold]")
            success = archive_startup_logs(
                base_dir=".",
                log_age_days=1,
                model_age_days=1, 
                tensorboard_age_days=1
            )
            if success:
                console.print("[green]✅ Quick archiving completed![/green]")
            
        elif choice == 7:
            # Archive everything (all files immediately)
            console.print("[bold]🗑️ Archiving all files (including DATA_ANALYSIS)...[/bold]")
            from log_archiver import LogArchiver
            archiver = LogArchiver(".")
            success = archiver.archive_everything_now()
            # Also archive DATA_ANALYSIS subfolders
            success_analysis = archive_data_analysis_subfolders()
            if success and success_analysis:
                console.print("[green]✅ Complete archiving finished![/green]")
            else:
                console.print("[red]❌ Some archiving operations failed![/red]")
                
        elif choice == 8:
            # View archive contents
            console.print("\n[bold]📋 Archive Contents:[/bold]")
            if archive_files:
                for archive_file in sorted(archive_files, reverse=True):  # Newest first
                    file_size = archive_file.stat().st_size / 1024  # KB
                    mod_time = datetime.fromtimestamp(archive_file.stat().st_mtime)
                    console.print(f"  📦 {archive_file.name}")
                    console.print(f"     Size: {file_size:.1f} KB | Created: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                console.print("  [yellow]No archive files found[/yellow]")
                
        elif choice == 9:
            # Cancel - do nothing
            console.print("[yellow]Archive operation cancelled[/yellow]")
            return
            
        elif choice == 10:
            return
            
        # Pause before returning to menu
        if choice not in [9, 10]:  # Don't pause for Cancel or Back options
            input("\nPress Enter to continue...")
            
    except Exception as e:
        console.print(f"[red]❌ Error during archiving: {str(e)}[/red]")
        input("\nPress Enter to continue...")

def main():
    """Main application loop"""
    # Clear screen and show welcome
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Archive old logs before starting
    console.print("[bold]🗂️  Checking for old logs to archive...[/bold]")
    try:
        archive_startup_logs(base_dir=".")
    except Exception as e:
        console.print(f"[yellow]⚠️  Log archiving failed: {str(e)}[/yellow]")
        console.print("[yellow]Continuing with normal startup...[/yellow]")
    
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
            
            choice = IntPrompt.ask("Select an option", default=9)
            
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
                archive_logs_menu()
            elif choice == 8:
                show_help()
            elif choice == 9:
                console.print("[bold green]👋 Thank you for using the Trading Bot![/bold green]")
                console.print("[yellow]⚠️  Remember: Trading involves risk. Trade responsibly![/yellow]")
                break
            else:
                console.print("[red]❌ Invalid option. Please try again.[/red]")
            
            # Pause before showing menu again
            if choice != 9:
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
