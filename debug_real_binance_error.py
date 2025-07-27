"""
Debug the live trading NoneType error with real Binance data
"""

import asyncio
import traceback
from binance_integration import LiveBTCUSDTTradingSystem

async def debug_live_trading_error():
    """Debug the actual live trading system to find the NoneType error"""
    
    print("=== Debugging Live Trading NoneType Error ===")
    
    try:
        # Create the actual trading system
        print("1. Creating LiveBTCUSDTTradingSystem...")
        trading_system = LiveBTCUSDTTradingSystem(
            model_path="models/best_model.zip",
            config_path="config.json"
        )
        
        print("2. Loading real Binance historical data...")
        # Use the actual startup method that loads real Binance data
        success = await trading_system.fetch_startup_historical_data(periods=30)
        
        if not success:
            print("❌ Failed to load real Binance data")
            return
            
        print(f"✅ Loaded {len(trading_system.market_data_buffer)} real Binance candles")
        
        # Now test the exact sequence that fails in live trading
        print("3. Testing prepare_model_input with real data...")
        model_data = trading_system.prepare_model_input()
        
        if model_data is None:
            print("❌ prepare_model_input returned None with real data")
            return
            
        print(f"✅ Model input prepared: {model_data.shape}")
        
        # Check the real data for any issues
        print("4. Analyzing real Binance data quality...")
        for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
            null_count = model_data[col].isnull().sum()
            inf_count = model_data[col].isin([float('inf'), float('-inf')]).sum()
            zero_count = (model_data[col] == 0).sum()
            
            print(f"   {col}: {null_count} nulls, {inf_count} infs, {zero_count} zeros")
            
            if null_count > 0 or inf_count > 0:
                print(f"   ⚠️ Data quality issue in {col}")
        
        # Test model signal generation with real data
        print("5. Testing get_model_signal with real Binance data...")
        try:
            signal = await trading_system.get_model_signal(model_data)
            
            if signal is not None:
                print(f"✅ Model signal SUCCESS with real data!")
                print(f"   Action: {signal.action_type}")
                print(f"   Leverage: {signal.leverage}")
                print(f"   Risk: {signal.risk_percentage}")
            else:
                print("❌ Model signal returned None with real data")
                
        except Exception as e:
            print(f"❌ Model signal FAILED with real data: {e}")
            print("Full traceback:")
            traceback.print_exc()
            
            # This is the exact error we're trying to fix
            if "unsupported operand type(s) for -: 'float' and 'NoneType'" in str(e):
                print("\n🎯 FOUND THE ERROR! This is the NoneType arithmetic issue.")
                print("The error occurs when real Binance data hits the trading environment.")
                
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_live_trading_error())
