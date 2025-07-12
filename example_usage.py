#!/usr/bin/env python3
"""
Example usage of the CoinAPI multi-data downloader
"""

import sys
from pathlib import Path

# Add the DataDownload directory to path
sys.path.append(str(Path(__file__).parent / "DataDownload"))

from DataDownload.coinapi_downloader import download_crypto_data, load_combined_data, CoinAPIDownloader

def download_btc_data():
    """Download comprehensive BTC data"""
    print("🚀 Downloading BTC data with funding rates and open interest...")
    
    # Download 7 days of data with all data types
    success = download_crypto_data(
        symbol='BTCUSDT',
        interval='15m',
        days=7,
        data_types=['ohlcv', 'funding_rates', 'open_interest']
    )
    
    if success:
        print("✅ Download completed successfully!")
        
        # Load and examine the combined data
        df = load_combined_data('BTCUSDT', '15m', days=7)
        
        if df is not None:
            print(f"\n📊 Data Summary:")
            print(f"   Records: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Show some statistics
            print(f"\n📈 Price Statistics:")
            print(f"   Open range: ${df['open'].min():.2f} - ${df['open'].max():.2f}")
            print(f"   Close range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"   Volume range: {df['volume'].min():.2f} - {df['volume'].max():,.2f}")
            
            # Check for funding rates and open interest
            if 'funding_rate' in df.columns:
                funding_mean = df['funding_rate'].mean()
                print(f"   Average funding rate: {funding_mean:.6f} ({funding_mean*100:.4f}%)")
            
            if 'open_interest' in df.columns:
                oi_mean = df['open_interest'].mean()
                print(f"   Average open interest: {oi_mean:,.0f}")
            
            # Save to file for later use
            output_file = f"btc_combined_data_{df.index.min().strftime('%Y%m%d')}_{df.index.max().strftime('%Y%m%d')}.csv"
            df.to_csv(output_file)
            print(f"\n💾 Data saved to: {output_file}")
            
            return df
        else:
            print("❌ Failed to load combined data")
            return None
    else:
        print("❌ Download failed")
        return None

def check_data_availability():
    """Check what data is available locally"""
    print("\n🔍 Checking available data...")
    
    downloader = CoinAPIDownloader()
    summary = downloader.get_available_data_summary('BTCUSDT', '15m')
    
    print("📊 Local Data Availability:")
    for data_type, info in summary['data_types'].items():
        if info['available'] and info['file_count'] > 0:
            print(f"   ✅ {data_type.upper()}: {info['file_count']} files")
            if info['date_range']:
                print(f"      Range: {info['date_range']['start']} to {info['date_range']['end']}")
        else:
            print(f"   ❌ {data_type.upper()}: No data available")

def download_multiple_symbols():
    """Download data for multiple symbols"""
    print("\n🔄 Downloading data for multiple symbols...")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    for symbol in symbols:
        print(f"\n📊 Downloading {symbol}...")
        success = download_crypto_data(
            symbol=symbol,
            interval='1h',  # Use 1h for faster download
            days=3,
            data_types=['ohlcv', 'funding_rates']  # Skip open interest for speed
        )
        
        if success:
            print(f"   ✅ {symbol} download completed")
        else:
            print(f"   ❌ {symbol} download failed")

def main():
    """Main example function"""
    print("🚀 CoinAPI Multi-Data Downloader Examples")
    print("=" * 50)
    
    # Check current data availability
    check_data_availability()
    
    # Download comprehensive BTC data
    btc_data = download_btc_data()
    
    # Download multiple symbols (optional)
    choice = input("\n❓ Download multiple symbols? (y/n): ").lower().strip()
    if choice == 'y':
        download_multiple_symbols()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 Example completed!")
    
    if btc_data is not None:
        print("✅ You now have comprehensive BTC trading data including:")
        print("   📈 OHLCV price data")
        if 'funding_rate' in btc_data.columns:
            print("   💰 Funding rates")
        if 'open_interest' in btc_data.columns:
            print("   📊 Open interest")
        print("\n🚀 Ready for advanced trading strategy development!")
    else:
        print("❌ No data was successfully downloaded")
        print("   Check your API key and network connection")

if __name__ == "__main__":
    main()
