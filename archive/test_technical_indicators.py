"""Test technical indicator calculations to find the NoneType error"""

import pandas as pd
import numpy as np
import ta

def test_technical_indicators():
    """Test technical indicator calculations"""
    
    # Create sample data
    data = {
        'close': [50000 + i * 10 for i in range(50)],
        'high': [50000 + i * 10 + 5 for i in range(50)],
        'low': [50000 + i * 10 - 5 for i in range(50)],
        'volume': [1000] * 50
    }
    df = pd.DataFrame(data)
    
    print("Testing technical indicators...")
    
    try:
        # RSI
        print("\n1. Testing RSI...")
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        print(f"RSI nulls: {df['rsi'].isnull().sum()}")
        
        # MACD
        print("\n2. Testing MACD...")
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        print(f"MACD nulls: {df['macd'].isnull().sum()}")
        
        # Bollinger Bands
        print("\n3. Testing Bollinger Bands...")
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        print(f"BB upper nulls: {df['bb_upper'].isnull().sum()}")
        print(f"BB middle nulls: {df['bb_middle'].isnull().sum()}")
        print(f"BB lower nulls: {df['bb_lower'].isnull().sum()}")
        
        # The problematic calculation
        print("\n4. Testing BB width calculation...")
        print("First few BB values:")
        print(f"BB upper: {df['bb_upper'].head(25).values}")
        print(f"BB middle: {df['bb_middle'].head(25).values}")
        print(f"BB lower: {df['bb_lower'].head(25).values}")
        
        # Try the calculation that might be failing
        try:
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            print(f"BB width calculation successful")
            print(f"BB width nulls: {df['bb_width'].isnull().sum()}")
        except Exception as e:
            print(f"BB width calculation failed: {e}")
            
            # Debug each part
            print("\nDebugging BB width calculation:")
            diff = df['bb_upper'] - df['bb_lower']
            print(f"Upper - Lower: {diff.head()}")
            print(f"Middle values: {df['bb_middle'].head()}")
            
            # Check for zero values
            zero_middle = (df['bb_middle'] == 0).sum()
            print(f"Zero values in bb_middle: {zero_middle}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_technical_indicators()