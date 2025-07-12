"""
Simple ATR debugging script
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

# Create simple test data
np.random.seed(42)
n = 100

# Generate realistic price data
prices = np.random.uniform(45000, 55000, n)
highs = prices * np.random.uniform(1.005, 1.02, n)
lows = prices * np.random.uniform(0.98, 0.995, n)
closes = prices
opens = np.roll(closes, 1)
opens[0] = closes[0]

df = pd.DataFrame({
    'open': opens,
    'high': highs,
    'low': lows,
    'close': closes,
    'volume': np.random.uniform(100, 1000, n),
    'timestamp': range(n)
})

print("Original data shape:", df.shape)
print("Data types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())

# Test ATR calculation directly
print("\n" + "="*50)
print("DIRECT ATR CALCULATION TEST")
print("="*50)

try:
    atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
    print(f"ATR result type: {type(atr_result)}")
    print(f"ATR result shape: {atr_result.shape if hasattr(atr_result, 'shape') else 'No shape'}")
    print(f"ATR first 20 values:")
    print(atr_result.head(20))
    
    # Add to dataframe
    df['atr'] = atr_result
    print(f"\nDataFrame after adding ATR:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"ATR column stats:")
    print(df['atr'].describe())
    
    # Check for NaN values
    print(f"NaN values in ATR: {df['atr'].isna().sum()}")
    print(f"Valid ATR values: {df['atr'].notna().sum()}")
    
    # Show some valid ATR values
    valid_atr = df['atr'].dropna()
    if len(valid_atr) > 0:
        print(f"\nFirst 10 valid ATR values:")
        for i, val in enumerate(valid_atr.head(10)):
            price = df.loc[valid_atr.index[i], 'close']
            pct = (val / price) * 100
            print(f"Index {valid_atr.index[i]:2d}: ATR = ${val:.2f}, Price = ${price:.0f}, ATR% = {pct:.3f}%")
    
except Exception as e:
    print(f"Error calculating ATR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("TESTING TRADING ENVIRONMENT")
print("="*50)

try:
    # Test with trading environment
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from trading_environment import FuturesTradingEnv
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        use_dynamic_stops=True,
        window_size=20
    )
    
    print("Environment created successfully")
    print(f"Price data shape: {env.price_data.shape}")
    print(f"Price data columns: {list(env.price_data.columns)}")
    
    # Check if ATR exists
    if 'atr' in env.price_data.columns:
        print("ATR column found in price_data")
        atr_values = env.price_data['atr'].dropna()
        print(f"Valid ATR values: {len(atr_values)}")
        if len(atr_values) > 0:
            print("Sample ATR values:")
            for i in range(min(5, len(atr_values))):
                idx = atr_values.index[i]
                atr_val = atr_values.iloc[i]
                price = env.price_data.loc[idx, 'close']
                pct = (atr_val / price) * 100
                print(f"Step {idx}: ATR = ${atr_val:.2f}, Price = ${price:.0f}, ATR% = {pct:.3f}%")
    else:
        print("ATR column NOT found in price_data")
        print("Available columns:", list(env.price_data.columns))

except Exception as e:
    print(f"Error with trading environment: {e}")
    import traceback
    traceback.print_exc()
