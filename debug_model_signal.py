"""
Debug the specific NoneType error in model signal generation
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from binance_integration import LiveBTCUSDTTradingSystem

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def debug_model_signal():
    """Debug the model signal generation to find the NoneType error"""
    
    print("=== Debugging Model Signal Generation ===")
    
    try:
        # Initialize trading system
        trading_system = LiveBTCUSDTTradingSystem(
            model_path="models/best_model.zip",
            config_path="config.json"
        )
        
        # Create sample market data (simulate what would come from Binance)
        print("Creating test market data...")
        dates = pd.date_range(start='2025-01-01', periods=50, freq='15min')
        np.random.seed(42)
        
        initial_price = 45000
        returns = np.random.normal(0, 0.001, 50)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Fill market data buffer with proper Binance-style data
        for i in range(50):
            timestamp = int(dates[i].timestamp() * 1000)  # Milliseconds like Binance
            
            candle = {
                'timestamp': timestamp,
                'open': float(prices[i] * (1 + np.random.uniform(-0.001, 0.001))),
                'high': float(prices[i] * (1 + abs(np.random.uniform(0, 0.002)))),
                'low': float(prices[i] * (1 - abs(np.random.uniform(0, 0.002)))),
                'close': float(prices[i]),
                'volume': float(np.random.uniform(1000, 5000))
            }
            
            trading_system.market_data_buffer.append(candle)
        
        print(f"Filled buffer with {len(trading_system.market_data_buffer)} candles")
        
        # Test prepare_model_input
        print("\n1. Testing prepare_model_input...")
        model_data = trading_system.prepare_model_input()
        
        if model_data is not None:
            print(f"✅ Model input prepared: {model_data.shape}")
            print(f"Columns: {list(model_data.columns)}")
            print(f"Data types:\n{model_data.dtypes}")
            
            # Check for any None/NaN values
            print("\nChecking for null values:")
            for col in model_data.columns:
                null_count = model_data[col].isnull().sum()
                none_count = (model_data[col] == None).sum() if col != 'timestamp' else 0
                print(f"  {col}: {null_count} nulls, {none_count} Nones")
            
            # Test get_model_signal with detailed debugging
            print("\n2. Testing get_model_signal with debugging...")
            signal = await trading_system.get_model_signal(model_data)
            
            if signal is not None:
                print(f"✅ Model signal generated successfully!")
                print(f"   Action type: {signal.action_type}")
                print(f"   Leverage: {signal.leverage}")
                print(f"   Risk percentage: {signal.risk_percentage}")
            else:
                print("❌ Model signal generation failed")
                
        else:
            print("❌ Failed to prepare model input")
            
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_model_signal())
