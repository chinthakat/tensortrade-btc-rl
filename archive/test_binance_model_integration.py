"""
Test the Binance integration with the fixed observation space
"""

import asyncio
import pandas as pd
import numpy as np
from binance_integration import LiveBTCUSDTTradingSystem

async def test_binance_model_integration():
    """Test the model prediction with Binance integration"""
    
    print("=== Testing Binance Integration Model Prediction ===")
    
    try:
        # Initialize Binance integration
        print("Initializing Binance integration...")
        binance = LiveBTCUSDTTradingSystem(
            testnet=True,
            initial_balance=10000.0,
            leverage=1.0,
            risk_per_trade=0.02
        )
        
        # Create sample market data (simulate having enough data in buffer)
        print("Creating sample market data...")
        dates = pd.date_range(start='2025-01-01', periods=50, freq='15min')
        np.random.seed(42)
        
        initial_price = 45000
        returns = np.random.normal(0, 0.001, 50)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Fill the market data buffer with sample data
        for i in range(50):
            timestamp = int(dates[i].timestamp() * 1000)
            open_price = prices[i] * (1 + np.random.uniform(-0.001, 0.001))
            high_price = prices[i] * (1 + abs(np.random.uniform(0, 0.002)))
            low_price = prices[i] * (1 - abs(np.random.uniform(0, 0.002)))
            close_price = prices[i]
            volume = np.random.uniform(1000, 5000)
            
            binance.market_data_buffer.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        print(f"Market data buffer filled with {len(binance.market_data_buffer)} records")
        
        # Test prepare_model_input
        print("\nTesting prepare_model_input...")
        model_data = binance.prepare_model_input()
        
        if model_data is not None:
            print(f"✅ Model input prepared successfully: {len(model_data)} rows")
            print(f"   Columns: {list(model_data.columns)}")
            print(f"   Shape: {model_data.shape}")
        else:
            print("❌ Failed to prepare model input")
            return
        
        # Test get_model_signal
        print("\nTesting get_model_signal...")
        signal = await binance.get_model_signal(model_data)
        
        if signal is not None:
            print(f"✅ Model signal generated successfully!")
            print(f"   Action type: {signal.action_type}")
            print(f"   Leverage: {signal.leverage}")
            print(f"   Risk percentage: {signal.risk_percentage}")
            print(f"   Confidence: {signal.confidence}")
            print(f"   Timestamp: {signal.timestamp}")
        else:
            print("❌ Failed to generate model signal")
            return
        
        print("\n🎉 SUCCESS: Binance integration with model prediction is working!")
        
    except Exception as e:
        print(f"❌ Error in Binance integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_binance_model_integration())
