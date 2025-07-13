#!/usr/bin/env python3
"""
Demo script showing the interactive features
"""

print("🚀 CoinAPI Data Downloader - Interactive Demo")
print("=" * 50)
print()

print("✨ New Features:")
print("   📊 Symbol selection with defaults (BTCUSDT)")
print("   ⏱️  Interval selection with defaults (15m)")
print("   📈 Data type selection with defaults (ohlcv)")  
print("   📅 Date range with defaults (last 7 days)")
print("   📁 Output directory with defaults (data/coinapi)")
print()

print("💡 How to use:")
print("   • Just press Enter to use default values")
print("   • Or type a number to select an option")
print("   • Or type custom values")
print()

print("📋 Example with all defaults:")
print("   Symbol: BTCUSDT (just press Enter)")
print("   Interval: 15m (just press Enter)")
print("   Data type: ohlcv (just press Enter)")
print("   Date range: last 7 days (just press Enter)")
print("   Output: data/coinapi (just press Enter)")
print()

print("🎯 Quick examples:")
print("   # Use all defaults (fastest):")
print("   python download_script.py")
print("   (then press Enter 5 times)")
print()
print("   # Command line mode still works:")
print("   python download_script.py --symbol ETHUSDT --last-days 3")
print()

print("🎉 Ready to try? Run: python download_script.py")
