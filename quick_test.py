#!/usr/bin/env python3
"""Simple test to check if the None type error is fixed"""

try:
    print("Testing signal generation...")
    import asyncio
    from binance_integration import LiveBTCUSDTTradingSystem
    
    async def quick_test():
        system = LiveBTCUSDTTradingSystem('models/best_model.zip')
        
        # Add test data
        for i in range(35):
            system.market_data_buffer.append({
                'timestamp': 1672531200000 + i * 900000,
                'open': 50000 + i,
                'high': 50100 + i,
                'low': 49900 + i,
                'close': 50050 + i,
                'volume': 1000
            })
        
        data = system.prepare_model_input()
        if data is not None:
            signal = await system.get_model_signal(data)
            if signal:
                print(f"✅ SUCCESS: Signal generated with action_type={signal.action_type}")
                return True
            else:
                print("❌ No signal generated")
                return False
        else:
            print("❌ Failed to prepare data")
            return False
    
    result = asyncio.run(quick_test())
    if result:
        print("🎉 All tests passed! NoneType error is fixed.")
    else:
        print("❌ Tests failed.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
