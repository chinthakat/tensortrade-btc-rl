#!/usr/bin/env python3
"""
Test script for the new CoinAPI multi-data downloader
"""

import sys
import os
from pathlib import Path

# Add the DataDownload directory to path
sys.path.append(str(Path(__file__).parent))

from coinapi_downloader import CoinAPIDownloader, download_crypto_data, load_combined_data

def test_basic_download():
    """Test basic OHLCV download"""
    print("🔄 Testing basic OHLCV download...")
    
    downloader = CoinAPIDownloader()
    
    # Test single data type download
    df = downloader.download_single_data_type(
        symbol='BTCUSDT',
        data_type='ohlcv',
        interval='15m',
        start_date='2024-01-01',
        end_date='2024-01-02',
        exchange='BINANCE'
    )
    
    if df is not None:
        print(f"✅ Downloaded {len(df)} OHLCV records")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        return True
    else:
        print("❌ OHLCV download failed")
        return False

def test_funding_rates():
    """Test funding rates download"""
    print("\n🔄 Testing funding rates download...")
    
    downloader = CoinAPIDownloader()
    
    df = downloader.download_single_data_type(
        symbol='BTCUSDT',
        data_type='funding_rates',
        interval='15m',
        start_date='2024-01-01',
        end_date='2024-01-02',
        exchange='BINANCE'
    )
    
    if df is not None:
        print(f"✅ Downloaded {len(df)} funding rate records")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        return True
    else:
        print("⚠️  Funding rates download failed (may not be available)")
        return False

def test_open_interest():
    """Test open interest download"""
    print("\n🔄 Testing open interest download...")
    
    downloader = CoinAPIDownloader()
    
    df = downloader.download_single_data_type(
        symbol='BTCUSDT',
        data_type='open_interest',
        interval='15m',
        start_date='2024-01-01',
        end_date='2024-01-02',
        exchange='BINANCE'
    )
    
    if df is not None:
        print(f"✅ Downloaded {len(df)} open interest records")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        return True
    else:
        print("⚠️  Open interest download failed (may not be available)")
        return False

def test_multi_data_batch():
    """Test multi-data batch download"""
    print("\n🔄 Testing multi-data batch download...")
    
    success = download_crypto_data(
        symbol='BTCUSDT',
        interval='15m',
        days=3,
        data_types=['ohlcv', 'funding_rates', 'open_interest']
    )
    
    if success:
        print("✅ Multi-data batch download completed")
        return True
    else:
        print("❌ Multi-data batch download failed")
        return False

def test_data_combination():
    """Test combining daily files"""
    print("\n🔄 Testing data combination...")
    
    df = load_combined_data('BTCUSDT', '15m', days=3)
    
    if df is not None:
        print(f"✅ Combined data loaded successfully")
        print(f"📊 Records: {len(df)}")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        # Show sample data
        print("\n📋 Sample data:")
        print(df.head())
        
        return True
    else:
        print("❌ Data combination failed")
        return False

def test_data_summary():
    """Test data availability summary"""
    print("\n🔄 Testing data summary...")
    
    downloader = CoinAPIDownloader()
    summary = downloader.get_available_data_summary('BTCUSDT', '15m')
    
    print("📊 Data Summary:")
    print(f"   Symbol: {summary['symbol']}")
    print(f"   Interval: {summary['interval']}")
    
    for data_type, info in summary['data_types'].items():
        print(f"   📈 {data_type.upper()}:")
        print(f"      Available: {info['available']}")
        print(f"      Files: {info['file_count']}")
        if info['date_range']:
            print(f"      Range: {info['date_range']['start']} to {info['date_range']['end']}")

def main():
    """Run all tests"""
    print("🚀 CoinAPI Multi-Data Downloader Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic OHLCV Download", test_basic_download),
        ("Funding Rates Download", test_funding_rates),
        ("Open Interest Download", test_open_interest),
        ("Multi-Data Batch Download", test_multi_data_batch),
        ("Data Combination", test_data_combination),
        ("Data Summary", test_data_summary)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed!")
    elif passed > 0:
        print("⚠️  Some tests failed - check API key and network connection")
    else:
        print("❌ All tests failed - check API key and CoinAPI service")

if __name__ == "__main__":
    main()
