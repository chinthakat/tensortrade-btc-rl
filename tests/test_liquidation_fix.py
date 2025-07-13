"""
Test script to demonstrate the enhanced liquidation logic in the trading environment.

This script tests the realistic liquidation implementation compared to the old simplified version.
"""

import pandas as pd
import numpy as np
import sys
import os
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv


def create_test_data(n_samples: int = 1000, initial_price: float = 50000.0) -> pd.DataFrame:
    """Create synthetic price data for testing liquidation scenarios"""
    
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15T')
    
    # Generate realistic OHLCV data with volatility
    returns = np.random.normal(0.0, 0.02, n_samples)  # 2% volatility
    
    # Create price series
    prices = [initial_price]
    for i in range(1, n_samples):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    # Generate OHLC from close prices
    highs = prices * (1 + np.random.uniform(0.001, 0.01, n_samples))
    lows = prices * (1 - np.random.uniform(0.001, 0.01, n_samples))
    opens = np.roll(prices, 1)  # Open is previous close
    opens[0] = prices[0]
    
    # Generate volume
    volumes = np.random.uniform(100, 1000, n_samples)
    
    df = pd.DataFrame({
        'timestamp': [int(ts.timestamp()) for ts in timestamps],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    return df


def test_liquidation_scenarios():
    """Test different liquidation scenarios"""
    
    print("Testing Enhanced Liquidation Logic")
    print("=" * 60)
    
    # Create test data
    df = create_test_data(500, 50000.0)
    
    # Test Scenario 1: High leverage long position with declining price
    print("\nScenario 1: High Leverage Long Position (20x)")
    print("-" * 40)
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=25.0,
        maintenance_margin_rate=0.004,  # 0.4%
        liquidation_fee_rate=0.005,     # 0.5%
        window_size=60
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Debug: Check initial state
    current_price = env.price_data.iloc[env.current_step]['close']
    print(f"Initial step: {env.current_step}")
    print(f"Initial price: ${current_price:.2f}")
    print(f"Initial equity: ${env.equity:.2f}")
    
    # Take a high leverage long position (20x instead of 25x to avoid equity issues)
    action = np.array([20.0])  # High leverage long
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Debug information
    current_price = env.price_data.iloc[env.current_step]['close']
    print(f"Current price: ${current_price:.2f}")
    print(f"Position opened at: ${env.entry_price:.2f}")
    print(f"Position size: {env.position_size:.6f} BTC")
    print(f"Leverage: {env.leverage:.2f}x")
    
    if env.liquidation_price is not None:
        print(f"Liquidation price: ${env.liquidation_price:.2f}")
    else:
        print("Liquidation price: None (no position opened)")
        return  # Exit early if no position opened
    
    liquidation_info = env.get_liquidation_info()
    if liquidation_info.get('margin_ratio'):
        print(f"Initial margin ratio: {liquidation_info['margin_ratio']:.2f}")
        print(f"Distance to liquidation: {liquidation_info['liquidation_distance_pct']}")
        print(f"Liquidation risk: {liquidation_info['liquidation_risk']}")
    else:
        print("No position information available")
    
    # Simulate price movements
    steps_taken = 0
    liquidated = False
    
    for i in range(100):  # Run for up to 100 steps
        if env.position_size == 0:
            liquidated = True
            break
            
        obs, reward, terminated, truncated, info = env.step(np.array([20.0]))  # Hold position
        steps_taken += 1
        
        if i % 20 == 0:  # Print updates every 20 steps
            current_price = env.price_data.iloc[env.current_step]['close']
            liquidation_info = env.get_liquidation_info()
            
            print(f"\nStep {env.current_step}: Price ${current_price:.2f}")
            if env.liquidation_price:
                distance = abs(current_price - env.liquidation_price) / current_price
                print(f"Distance to liquidation: {distance:.2%}")
                if liquidation_info.get('margin_ratio'):
                    print(f"Margin ratio: {liquidation_info['margin_ratio']:.2f}")
                print(f"Unrealized PnL: ${env.unrealized_pnl:.2f}")
        
        if terminated or truncated:
            break
    
    if liquidated:
        print(f"\nLIQUIDATION OCCURRED after {steps_taken} steps")
        print(f"Final balance: ${env.balance:.2f}")
        print(f"Total realized PnL: ${env.total_realized_pnl:.2f}")
    else:
        print(f"\nPosition survived {steps_taken} steps without liquidation")
    
    # Test Scenario 2: Compare maintenance margin rates
    print("\n\nScenario 2: Maintenance Margin Rate Comparison")
    print("-" * 50)
    
    test_maintenance_rates = [0.002, 0.004, 0.008, 0.01]  # 0.2%, 0.4%, 0.8%, 1.0%
    
    for mm_rate in test_maintenance_rates:
        test_env = FuturesTradingEnv(
            df=df.iloc[:100].copy(),  # Use smaller dataset
            initial_equity=10000.0,
            max_leverage=25.0,
            maintenance_margin_rate=mm_rate,
            liquidation_fee_rate=0.005,
            window_size=60
        )
        
        test_env.reset()
        
        # Open 20x long position
        action = np.array([20.0])
        test_env.step(action)
        
        liquidation_price = test_env.liquidation_price
        entry_price = test_env.entry_price
        if liquidation_price and entry_price:
            liquidation_distance = abs(entry_price - liquidation_price) / entry_price
            print(f"MM Rate: {mm_rate:.1%} | Liquidation Distance: {liquidation_distance:.2%} | Liq Price: ${liquidation_price:.2f}")
        else:
            print(f"MM Rate: {mm_rate:.1%} | Position not opened")
    
    # Test Scenario 3: Different leverage levels
    print("\n\nScenario 3: Leverage Impact on Liquidation")
    print("-" * 45)
    
    test_leverages = [5, 10, 15, 20, 25]
    
    for leverage in test_leverages:
        test_env = FuturesTradingEnv(
            df=df.iloc[:100].copy(),
            initial_equity=10000.0,
            max_leverage=25.0,
            maintenance_margin_rate=0.004,
            liquidation_fee_rate=0.005,
            window_size=60
        )
        
        test_env.reset()
        
        # Open position with specific leverage
        action = np.array([float(leverage)])
        test_env.step(action)
        
        if test_env.liquidation_price and test_env.entry_price:
            liquidation_distance = abs(test_env.entry_price - test_env.liquidation_price) / test_env.entry_price
            print(f"Leverage: {leverage:2d}x | Liquidation Distance: {liquidation_distance:.2%} | Liq Price: ${test_env.liquidation_price:.2f}")
        else:
            print(f"Leverage: {leverage:2d}x | Position not opened")
    
    print("\n" + "=" * 60)
    print("Enhanced liquidation testing complete!")
    print("\nKey improvements:")
    print("- Realistic maintenance margin calculations")
    print("- Accurate liquidation price determination")
    print("- Proper liquidation fee handling")
    print("- Real-time margin ratio monitoring")


def demonstrate_liquidation_info():
    """Demonstrate the liquidation information system"""
    
    print("\nLiquidation Information System Demo")
    print("=" * 50)
    
    df = create_test_data(200, 45000.0)
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=5000.0,
        max_leverage=20.0,
        maintenance_margin_rate=0.005,  # 0.5% (higher risk)
        liquidation_fee_rate=0.005,
        window_size=60
    )
    
    env.reset()
    
    # Open a risky 18x long position
    action = np.array([18.0])
    env.step(action)
    
    print("Position Details:")
    liquidation_info = env.get_liquidation_info()
    
    for key, value in liquidation_info.items():
        print(f"  {key}: {value}")
    
    print("\nReal-time monitoring during price movements:")
    
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(np.array([18.0]))
        
        current_price = env.price_data.iloc[env.current_step]['close']
        liquidation_info = env.get_liquidation_info()
        
        margin_ratio = liquidation_info.get('margin_ratio')
        liquidation_risk = liquidation_info.get('liquidation_risk', 'UNKNOWN')
        liquidation_distance_pct = liquidation_info.get('liquidation_distance_pct', 'N/A')
        
        # Handle None values gracefully
        margin_ratio_str = f"{margin_ratio:5.2f}" if margin_ratio is not None else " None"
        
        print(f"Step {i+1:2d}: Price ${current_price:6.0f} | "
              f"Margin Ratio: {margin_ratio_str} | "
              f"Risk: {liquidation_risk:6s} | "
              f"Distance: {liquidation_distance_pct:>7s}")
        
        if env.position_size == 0:  # Liquidated
            print("LIQUIDATED!")
            break
        
        if terminated or truncated:
            break


if __name__ == "__main__":
    try:
        test_liquidation_scenarios()
        demonstrate_liquidation_info()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
