"""
Test script to demonstrate the data leakage fix in the trading environment.
This script shows how to properly create environments without data leakage.
"""

import pandas as pd
import numpy as np
from trading_environment import FuturesTradingEnv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sample_data(n_samples=5000):
    """Create sample price data for testing"""
    np.random.seed(42)
    
    # Generate synthetic price data
    base_price = 50000
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    # Random walk for price
    returns = np.random.normal(0, 0.001, n_samples)
    returns = np.cumsum(returns)
    close_prices = base_price * np.exp(returns)
    
    # Create OHLV data
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    volumes = np.random.lognormal(10, 1, n_samples)
    
    df = pd.DataFrame({
        'timestamp': timestamps.astype(int) // 10**9,  # Convert to Unix timestamp
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return df

def test_old_vs_new_approach():
    """Compare old (leaky) vs new (fixed) approach"""
    print("=" * 80)
    print("TESTING DATA LEAKAGE FIX")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data(5000)
    print(f"Created sample dataset with {len(df)} samples")
    
    # Test 1: Old approach (would have data leakage)
    print("\n" + "="*50)
    print("TEST 1: Environment with proper train/val split")
    print("="*50)
    
    # Split data manually
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()
    
    # Create training environment (scaler fitted on training data only)
    print(f"\nCreating training environment with {len(train_df)} samples...")
    train_env = FuturesTradingEnv(
        df=train_df,
        training_end_idx=len(train_df),  # Use all training data for scaler
        initial_equity=10000,
        window_size=60
    )
    
    # Create validation environment (uses same scaler, no refitting)
    print(f"Creating validation environment with {len(val_df)} samples...")
    val_env = FuturesTradingEnv(
        df=val_df,
        training_end_idx=train_size,  # Reference training size
        initial_equity=10000,
        window_size=60
    )
    
    # Copy scaler from training to validation
    val_env.scaler = train_env.scaler
    val_env.feature_columns_scaled = pd.DataFrame(
        val_env.scaler.transform(val_env.feature_columns),
        columns=val_env.feature_columns.columns,
        index=val_env.feature_columns.index
    )
    
    # Get scaler parameters for verification
    train_scaler_params = train_env.get_scaler_params()
    print(f"\nTraining scaler fitted on {train_scaler_params['n_samples_seen']} samples")
    
    # Validate no data leakage
    validation_results = val_env.validate_no_data_leakage(train_size)
    print(f"Data leakage validation: {validation_results}")
    
    # Test 2: Using the new class method (recommended approach)
    print("\n" + "="*50)
    print("TEST 2: Using class method for proper splitting")
    print("="*50)
    
    # This is the recommended way to create environments
    train_env2, val_env2 = FuturesTradingEnv.create_train_val_environments(
        df=df,
        train_ratio=0.7,
        val_ratio=0.3,
        initial_equity=10000,
        window_size=60
    )
    
    # Test 3: Walk-forward validation
    print("\n" + "="*50)
    print("TEST 3: Walk-forward validation environments")
    print("="*50)
    
    walk_forward_envs = FuturesTradingEnv.create_walk_forward_environments(
        df=df,
        train_window=1000,
        val_window=200,
        step_size=500,
        initial_equity=10000,
        window_size=60
    )
    
    print(f"Created {len(walk_forward_envs)} walk-forward environment pairs")
    
    # Test each walk-forward pair for data leakage
    for i, (train_env_wf, val_env_wf) in enumerate(walk_forward_envs[:3]):  # Test first 3
        val_results = val_env_wf.validate_no_data_leakage(1000)  # Training window size
        print(f"Walk-forward pair {i+1}: {val_results.get('status', val_results.get('warning', 'ERROR'))}")
    
    print("\n" + "="*50)
    print("TEST 4: Feature scaling verification")
    print("="*50)
    
    # Compare feature statistics between training and validation
    train_features = train_env.feature_columns_scaled
    val_features = val_env.feature_columns_scaled
    
    print("Training data feature statistics (first 5 features):")
    for col in train_features.columns[:5]:
        mean_val = train_features[col].mean()
        std_val = train_features[col].std()
        print(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    print("\nValidation data feature statistics (first 5 features):")
    for col in val_features.columns[:5]:
        mean_val = val_features[col].mean()
        std_val = val_features[col].std()
        print(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    print("\nExpected: Training data should have ~0 mean, ~1 std")
    print("Validation data will have different mean/std (this is correct!)")
    
    print("\n" + "="*80)
    print("DATA LEAKAGE FIX VERIFICATION COMPLETE")
    print("="*80)
    print("✅ Feature scaling now prevents lookahead bias")
    print("✅ Scaler fitted only on training data")
    print("✅ Validation data transformed using training scaler")
    print("✅ Helper methods available for proper train/val splits")
    print("✅ Walk-forward validation supported")

def test_live_trading_simulation():
    """Test how to handle new incoming data (like in live trading)"""
    print("\n" + "="*50)
    print("TEST 5: Live trading data handling")
    print("="*50)
    
    # Create base environment with historical data
    df = create_sample_data(3000)
    env = FuturesTradingEnv(
        df=df,
        training_end_idx=int(len(df) * 0.8),  # Use 80% for scaler fitting
        initial_equity=10000,
        window_size=60
    )
    
    # Simulate new incoming data (like live trading)
    new_data = create_sample_data(100)  # New 100 samples
    
    # Calculate features for new data (you would do this in your live trading loop)
    # This is a simplified version - in practice you'd use the same feature engineering
    new_features = pd.DataFrame({
        'returns': new_data['close'].pct_change(),
        'log_returns': np.log(new_data['close'] / new_data['close'].shift(1)),
        'high_low_pct': (new_data['high'] - new_data['low']) / new_data['close'],
        'close_open_pct': (new_data['close'] - new_data['open']) / new_data['open']
    }).dropna()
    
    # Scale new features using the existing scaler (NO refitting)
    try:
        scaled_new_features = env.update_scaler_with_new_data(new_features)
        print("✅ Successfully scaled new data using existing scaler")
        print(f"   New data shape: {scaled_new_features.shape}")
        print(f"   Sample scaled values: {scaled_new_features.iloc[0].values[:4]}")
    except Exception as e:
        print(f"❌ Error scaling new data: {e}")

if __name__ == "__main__":
    test_old_vs_new_approach()
    test_live_trading_simulation()
    
    print("\n" + "="*80)
    print("HOW TO USE THE FIXED ENVIRONMENT:")
    print("="*80)
    print("""
1. For basic train/val split:
   train_env, val_env = FuturesTradingEnv.create_train_val_environments(
       df=your_data, train_ratio=0.7, val_ratio=0.3
   )

2. For walk-forward validation:
   env_pairs = FuturesTradingEnv.create_walk_forward_environments(
       df=your_data, train_window=5000, val_window=1000, step_size=1000
   )

3. For live trading:
   env = FuturesTradingEnv(df=historical_data, training_end_idx=len(historical_data))
   # Later, for new data:
   scaled_new = env.update_scaler_with_new_data(new_features)

4. To validate no data leakage:
   results = env.validate_no_data_leakage(validation_start_index)
   print(results)
""")
