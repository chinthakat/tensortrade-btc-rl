#!/usr/bin/env python3
"""
Interactive CoinAPI Data Downloader
Supports flexible date ranges and intervals with user-friendly prompts
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from coinapi_downloader import CoinAPIDownloader
import logging

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_date(date_str):
    """Parse date string in YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")

def get_date_range(args):
    """Get start and end dates based on user input"""
    today = datetime.now().date()
    
    if args.last_days:
        # Option 3: Last N days
        end_date = today
        start_date = today - timedelta(days=args.last_days)
        print(f"📅 Downloading last {args.last_days} days: {start_date} to {end_date}")
        
    elif args.since_date:
        # Option 1: Since date to today
        start_date = parse_date(args.since_date)
        end_date = today
        print(f"📅 Downloading since {start_date} to {end_date}")
        
    elif args.from_date and args.to_date:
        # Option 2: From date to date
        start_date = parse_date(args.from_date)
        end_date = parse_date(args.to_date)
        
        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")
        
        print(f"📅 Downloading from {start_date} to {end_date}")
        
    else:
        # Default: Last 30 days
        end_date = today
        start_date = today - timedelta(days=30)
        print(f"📅 No date range specified, using default: last 30 days ({start_date} to {end_date})")
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def get_available_intervals():
    """Get list of available intervals"""
    return {
        '1m': '1 minute',
        '5m': '5 minutes', 
        '15m': '15 minutes',
        '30m': '30 minutes',
        '1h': '1 hour',
        '4h': '4 hours',
        '1d': '1 day'
    }

def get_available_symbols():
    """Get list of available symbols"""
    return {
        'BTCUSDT': 'Bitcoin/USDT',
        'ETHUSDT': 'Ethereum/USDT',
        'ADAUSDT': 'Cardano/USDT',
        'SOLUSDT': 'Solana/USDT',
        'BNBUSDT': 'Binance Coin/USDT',
        'DOGEUSDT': 'Dogecoin/USDT'
    }

def get_available_data_types():
    """Get list of available data types"""
    return {
        'ohlcv': 'OHLC Volume data',
        'funding_rates': 'Funding rates (futures)',
        'both': 'Both OHLCV and funding rates'
    }

def prompt_with_default(prompt, default_value, validator=None):
    """Prompt user with a default value"""
    while True:
        try:
            user_input = input(f"{prompt} (default: {default_value}): ").strip()
            if not user_input:
                return default_value
            
            if validator:
                return validator(user_input)
            return user_input
        except (ValueError, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            print(f"❌ Invalid input: {e}. Please try again.")

def prompt_choice(prompt, options, default_key=None):
    """Prompt user to choose from options with default"""
    print(f"\n{prompt}")
    for i, (key, desc) in enumerate(options.items(), 1):
        marker = " (default)" if key == default_key else ""
        print(f"   {i}. {key} - {desc}{marker}")
    
    while True:
        try:
            choice = input(f"\nSelect option (1-{len(options)}) or press Enter for default: ").strip()
            
            if not choice and default_key:
                return default_key
            elif choice.isdigit() and 1 <= int(choice) <= len(options):
                return list(options.keys())[int(choice) - 1]
            elif choice.upper() in [k.upper() for k in options.keys()]:
                return choice.upper()
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid selection. Please try again.")

def prompt_date_range():
    """Prompt user for date range with defaults"""
    today = datetime.now().date()
    
    print(f"\n📅 Date range options:")
    print("   1. Last N days (default: last 7 days)")
    print("   2. Since specific date")
    print("   3. From date to date")
    
    while True:
        try:
            choice = input("\nSelect date range option (1-3) or press Enter for default: ").strip()
            
            if not choice or choice == "1":
                # Default: Last 7 days
                days = prompt_with_default("Enter number of days", "7", lambda x: int(x))
                return {'last_days': days}
            elif choice == "2":
                since_date = prompt_with_default("Enter since date", "2025-01-01", parse_date)
                return {'since_date': since_date.strftime('%Y-%m-%d')}
            elif choice == "3":
                from_date = prompt_with_default("Enter from date", "2025-01-01", parse_date)
                to_date = prompt_with_default("Enter to date", "2025-01-31", parse_date)
                
                if from_date > to_date:
                    print("❌ Start date cannot be after end date. Please try again.")
                    continue
                    
                return {'from_date': from_date.strftime('%Y-%m-%d'), 'to_date': to_date.strftime('%Y-%m-%d')}
            else:
                print("Invalid selection. Please try again.")
        except ValueError as e:
            print(f"❌ Error: {e}")

def interactive_mode():
    """Interactive mode for selecting options with defaults"""
    print("🚀 CoinAPI Interactive Data Downloader")
    print("=" * 50)
    print("💡 Press Enter to use default values")
    
    # Symbol selection
    symbols = get_available_symbols()
    symbol = prompt_choice("📊 Available symbols:", symbols, "BTCUSDT")
    
    # Interval selection
    intervals = get_available_intervals()
    interval = prompt_choice("⏱️  Available intervals:", intervals, "15m")
    
    # Data type selection
    data_types = get_available_data_types()
    data_type = prompt_choice("📈 Available data types:", data_types, "ohlcv")
    
    # Date range selection
    date_args = prompt_date_range()
    
    # Output directory
    output_dir = prompt_with_default("\n📁 Output directory", "data/coinapi")
    
    return symbol, interval, data_type, date_args, output_dir

def create_arg_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Download cryptocurrency data from CoinAPI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python download_script.py --interactive
  
  # Last 30 days of BTC 15m data
  python download_script.py --symbol BTCUSDT --interval 15m --last-days 30
  
  # Since specific date
  python download_script.py --symbol ETHUSDT --interval 1h --since-date 2025-01-01
  
  # Date range
  python download_script.py --symbol BTCUSDT --interval 4h --from-date 2025-01-01 --to-date 2025-01-31
  
  # Download both OHLCV and funding rates
  python download_script.py --symbol BTCUSDT --interval 15m --data-type both --last-days 7
        """
    )
    
    # Mode selection
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    
    # Symbol and interval
    parser.add_argument('--symbol', '-s', type=str, default='BTCUSDT',
                       help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='15m',
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                       help='Data interval (default: 15m)')
    
    # Data type
    parser.add_argument('--data-type', type=str, default='ohlcv',
                       choices=['ohlcv', 'funding_rates', 'both'],
                       help='Type of data to download (default: ohlcv)')
    
    # Date range options (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument('--last-days', type=int, metavar='N',
                           help='Download last N days of data')
    date_group.add_argument('--since-date', type=str, metavar='YYYY-MM-DD',
                           help='Download from specific date to today')
    
    # From/to dates (used together)
    parser.add_argument('--from-date', type=str, metavar='YYYY-MM-DD',
                       help='Start date (use with --to-date)')
    parser.add_argument('--to-date', type=str, metavar='YYYY-MM-DD',
                       help='End date (use with --from-date)')
    
    # Other options
    parser.add_argument('--output-dir', type=str, default='data/coinapi',
                       help='Output directory (default: data/coinapi)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser

def download_data(symbol, interval, data_type, start_date, end_date, output_dir, verbose=False):
    """Download the specified data"""
    print(f"\n🚀 Starting download...")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Data type: {data_type}")
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Output directory: {output_dir}")
    
    # Initialize downloader
    downloader = CoinAPIDownloader(data_dir=output_dir)
    
    # Determine which data types to download
    if data_type == 'both':
        data_types = ['ohlcv', 'funding_rates']
    else:
        data_types = [data_type]
    
    try:
        # Download using daily batches
        result = downloader.download_daily_data_batch(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types
        )
        
        if result:
            print(f"\n✅ Download completed successfully!")
            print(f"   📁 Data saved to: {output_dir}")
            
            # Show file locations
            for dtype in data_types:
                data_path = Path(output_dir) / symbol / interval / dtype
                if data_path.exists():
                    files = list(data_path.glob(f"*{dtype}*.csv"))
                    print(f"   📄 {dtype}: {len(files)} files in {data_path}")
        else:
            print("❌ Download failed. Check the logs above for details.")
            return False
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False
    
    return True

def main():
    """Main function - Interactive mode with command line fallback"""
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Check if any arguments were provided (command line mode)
        has_args = any([
            args.symbol != 'BTCUSDT',  # Non-default symbol
            args.interval != '15m',    # Non-default interval
            args.data_type != 'ohlcv', # Non-default data type
            args.last_days,
            args.since_date,
            args.from_date,
            args.to_date,
            args.output_dir != 'data/coinapi'
        ])
        
        if has_args and not args.interactive:
            # Command line mode - use provided arguments
            print("🚀 CoinAPI Data Downloader (Command Line Mode)")
            
            symbol = args.symbol
            interval = args.interval
            data_type = args.data_type
            output_dir = args.output_dir
            
            # Validate from/to date combination
            if (args.from_date and not args.to_date) or (args.to_date and not args.from_date):
                parser.error("--from-date and --to-date must be used together")
            
            start_date, end_date = get_date_range(args)
            
        else:
            # Interactive mode with defaults
            print("🚀 CoinAPI Data Downloader (Interactive Mode)")
            print("Welcome! This script will guide you through downloading cryptocurrency data.")
            print("💡 You can press Enter to use default values for quick setup.\n")
            
            symbol, interval, data_type, date_args, output_dir = interactive_mode()
            
            # Create a mock args object for get_date_range
            class MockArgs:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                    # Set defaults for missing attributes
                    for attr in ['last_days', 'since_date', 'from_date', 'to_date']:
                        if not hasattr(self, attr):
                            setattr(self, attr, None)
            
            mock_args = MockArgs(**date_args)
            start_date, end_date = get_date_range(mock_args)
        
        # Show summary before downloading
        print(f"\n📋 Download Summary:")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {interval}")
        print(f"   Data type: {data_type}")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Output directory: {output_dir}")
        
        # Confirm before proceeding (only in interactive mode)
        if not has_args or args.interactive:
            confirm = input("\n❓ Proceed with download? (Y/n): ").strip().lower()
            if confirm and confirm not in ['y', 'yes']:
                print("⏹️  Download cancelled by user")
                return
        
        # Download the data
        success = download_data(symbol, interval, data_type, start_date, end_date, output_dir, args.verbose)
        
        if success:
            print("\n🎉 All done!")
            if not has_args or args.interactive:
                print("💡 You can run the script again to download more data.")
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
