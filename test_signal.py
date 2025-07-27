#!/usr/bin/env python3
"""Test signal generation to debug the NoneType error"""

import asyncio
import sys
import traceback
from binance_integration import LiveBTCUSDTTradingSystem

async def test_signal():
    try:
        print("Initializing system...")
        system = LiveBTCUSDTTradingSystem('models/best_model.zip')
        print('✅ System initialized successfully')
        
        print("Adding test market data...")
        # Test data preparation
        for i in range(35):
            system.market_data_buffer.append({
                'timestamp': 1672531200000 + i * 900000,
                'open': 50000 + i,
                'high': 50100 + i,
                'low': 49900 + i,
                'close': 50050 + i,
                'volume': 1000
            })
        
        print(f"Market data buffer size: {len(system.market_data_buffer)}")
        
        print("Preparing model input...")
        data = system.prepare_model_input()
        if data is not None:
            print(f'✅ Data prepared successfully: {len(data)} rows')
            print(f"Data columns: {list(data.columns)}")
            print(f"Data shape: {data.shape}")
            
            print("Getting model signal...")
            signal = await system.get_model_signal(data)
            if signal:
                print(f'✅ Signal generated: action_type={signal.action_type}, leverage={signal.leverage}, risk={signal.risk_percentage}')
            else:
                print('❌ No signal generated')
        else:
            print('❌ Failed to prepare data')
            
    except Exception as e:
        print(f'❌ Error: {e}')
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_signal())
