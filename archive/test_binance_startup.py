"""
Test the new Binance historical data loading for immediate trading startup
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from binance_integration import LiveBTCUSDTTradingSystem

async def test_historical_data_loading():
    """Test loading 30 periods of real BTCUSDT 15m data from Binance"""
    
    print("=== Testing Historical Data Loading for Immediate Trading ===")
    
    try:
        # Initialize the trading system
        print("1. Initializing trading system...")
        trading_system = LiveBTCUSDTTradingSystem(
            model_path="models/best_model.zip",
            config_path="config.json"
        )
        
        print(f"   Symbol: {trading_system.symbol}")
        print(f"   Timeframe: {trading_system.timeframe}")
        print(f"   Initial buffer size: {len(trading_system.market_data_buffer)}")
        
        # Test the startup historical data loading
        print("\n2. Testing startup historical data loading...")
        success = await trading_system.fetch_startup_historical_data(periods=30)
        
        if success:
            print(f"   ✅ Successfully loaded {len(trading_system.market_data_buffer)} candles")
            
            # Show some sample data
            if len(trading_system.market_data_buffer) > 0:
                latest = trading_system.market_data_buffer[-1]
                oldest = trading_system.market_data_buffer[0]
                
                print(f"   📊 Latest candle: ${latest['close']:.2f} (Volume: {latest['volume']:.2f})")
                print(f"   📊 Oldest candle: ${oldest['close']:.2f}")
                print(f"   📅 Time span: {len(trading_system.market_data_buffer)} x 15-minute periods")
                
                # Test model input preparation
                print("\n3. Testing model input preparation...")
                model_data = trading_system.prepare_model_input()
                
                if model_data is not None:
                    print(f"   ✅ Model input prepared: {model_data.shape}")
                    print(f"   📋 Columns: {list(model_data.columns)}")
                    
                    # Test model signal generation
                    print("\n4. Testing model signal generation...")
                    signal = await trading_system.get_model_signal(model_data)
                    
                    if signal is not None:
                        print(f"   ✅ Model signal generated successfully!")
                        print(f"   🎯 Action type: {signal.action_type}")
                        print(f"   📊 Leverage: {signal.leverage:.2f}")
                        print(f"   📈 Risk percentage: {signal.risk_percentage:.2f}")
                        print(f"   🎯 Confidence: {signal.confidence:.2f}")
                        
                        # Interpret the action
                        action_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}
                        action_name = action_names.get(signal.action_type, "UNKNOWN")
                        print(f"   💡 Interpretation: {action_name} signal")
                        
                        print("\n🎉 SUCCESS: Ready for immediate live trading!")
                        return True
                        
                    else:
                        print("   ❌ Failed to generate model signal")
                        return False
                else:
                    print("   ❌ Failed to prepare model input")
                    return False
            else:
                print("   ❌ No data in buffer after loading")
                return False
        else:
            print("   ❌ Failed to load historical data")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_historical_data_loading()
    
    if success:
        print("\n✅ ALL TESTS PASSED")
        print("🚀 The system is ready to start live trading immediately!")
        print("💡 Technical indicators will be calculated from real Binance data")
        print("🎯 Model predictions will work from the first moment")
    else:
        print("\n❌ TESTS FAILED")
        print("🔧 Check your Binance API connection and model files")

if __name__ == "__main__":
    asyncio.run(main())
