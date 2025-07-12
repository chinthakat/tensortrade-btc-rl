"""
Test script for the Advanced Action Space implementation (Phase 1).

This script demonstrates the new Dict action space with leverage and risk_percentage controls,
comparing it with the legacy simple action space.
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


def create_test_data(n_samples: int = 200, initial_price: float = 50000.0) -> pd.DataFrame:
    """Create synthetic price data for testing action spaces"""
    
    np.random.seed(42)
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15T')
    
    # Generate realistic OHLCV data with volatility
    returns = np.random.normal(0.0, 0.015, n_samples)  # 1.5% volatility
    
    # Create price series
    prices = [initial_price]
    for i in range(1, n_samples):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    # Generate OHLC from close prices
    highs = prices * (1 + np.random.uniform(0.001, 0.008, n_samples))
    lows = prices * (1 - np.random.uniform(0.001, 0.008, n_samples))
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


def test_legacy_action_space():
    """Test the legacy simple action space (single leverage value)"""
    
    print("Testing Legacy Action Space (Simple Leverage)")
    print("=" * 60)
    
    # Create test data
    df = create_test_data(150, 50000.0)
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=10.0,
        taker_fee=0.0004,
        window_size=50,
        use_advanced_action_space=False  # Use legacy action space
    )
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"Action space: {env.action_space}")
    print(f"Initial equity: ${env.equity:.2f}")
    print(f"Initial balance: ${env.balance:.2f}")
    
    # Test different leverage actions
    test_actions = [5.0, -3.0, 0.0, 8.0, -10.0]
    
    for i, leverage in enumerate(test_actions):
        action = np.array([leverage])
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_price = env.price_data.iloc[env.current_step]['close']
        
        print(f"\\nStep {i+1}: Leverage {leverage:.1f}x")
        print(f"  Price: ${current_price:.2f}")
        print(f"  Position: {env.position_size:.6f} BTC")
        print(f"  Equity: ${env.equity:.2f}")
        print(f"  Balance: ${env.balance:.2f}")
        print(f"  Reward: {reward:.4f}")
        
        if terminated or truncated:
            break
    
    return env.equity


def test_advanced_action_space():
    """Test the new advanced Dict action space (leverage + risk_percentage)"""
    
    print("\\n\\nTesting Advanced Action Space (Dict: Leverage + Risk %)")
    print("=" * 60)
    
    # Create test data
    df = create_test_data(150, 50000.0)
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=10.0,
        taker_fee=0.0004,
        window_size=50,
        use_advanced_action_space=True  # Use advanced action space
    )
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"Action space: {env.action_space}")
    print(f"Initial equity: ${env.equity:.2f}")
    print(f"Initial balance: ${env.balance:.2f}")
    
    # Test different combinations of leverage and risk percentage
    test_actions = [
        {'leverage': 5.0, 'risk_percentage': 0.5},   # 5x leverage, 50% equity risk
        {'leverage': -3.0, 'risk_percentage': 0.3},  # -3x leverage, 30% equity risk
        {'leverage': 0.0, 'risk_percentage': 0.0},   # Close position
        {'leverage': 8.0, 'risk_percentage': 0.8},   # 8x leverage, 80% equity risk
        {'leverage': -10.0, 'risk_percentage': 0.2}  # -10x leverage, 20% equity risk
    ]
    
    for i, action_dict in enumerate(test_actions):
        action = {
            'leverage': np.array([action_dict['leverage']]),
            'risk_percentage': np.array([action_dict['risk_percentage']])
        }
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_price = env.price_data.iloc[env.current_step]['close']
        
        print(f"\\nStep {i+1}: Leverage {action_dict['leverage']:.1f}x, Risk {action_dict['risk_percentage']:.1%}")
        print(f"  Price: ${current_price:.2f}")
        print(f"  Position: {env.position_size:.6f} BTC")
        print(f"  Equity: ${env.equity:.2f}")
        print(f"  Balance: ${env.balance:.2f}")
        print(f"  Reward: {reward:.4f}")
        
        if terminated or truncated:
            break
    
    return env.equity


def test_risk_management_scenarios():
    """Test different risk management scenarios with advanced action space"""
    
    print("\\n\\nTesting Risk Management Scenarios")
    print("=" * 50)
    
    df = create_test_data(100, 50000.0)
    
    # Scenario 1: Conservative trader (low risk percentages)
    print("\\nScenario 1: Conservative Trader")
    print("-" * 30)
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=10.0,
        use_advanced_action_space=True
    )
    
    env.reset()
    
    # Conservative: Low leverage, low risk percentage
    conservative_actions = [
        {'leverage': 2.0, 'risk_percentage': 0.1},   # 2x leverage, 10% risk
        {'leverage': -1.5, 'risk_percentage': 0.15}, # -1.5x leverage, 15% risk
        {'leverage': 3.0, 'risk_percentage': 0.12}   # 3x leverage, 12% risk
    ]
    
    for action_dict in conservative_actions:
        action = {
            'leverage': np.array([action_dict['leverage']]),
            'risk_percentage': np.array([action_dict['risk_percentage']])
        }
        env.step(action)
        
        print(f"  Leverage: {action_dict['leverage']:.1f}x, Risk: {action_dict['risk_percentage']:.1%}")
        print(f"  Position: {env.position_size:.6f} BTC, Equity: ${env.equity:.2f}")
    
    conservative_final_equity = env.equity
    
    # Scenario 2: Aggressive trader (high risk percentages)
    print("\\nScenario 2: Aggressive Trader")
    print("-" * 30)
    
    env.reset()
    
    # Aggressive: High leverage, high risk percentage
    aggressive_actions = [
        {'leverage': 8.0, 'risk_percentage': 0.8},   # 8x leverage, 80% risk
        {'leverage': -6.0, 'risk_percentage': 0.7},  # -6x leverage, 70% risk
        {'leverage': 10.0, 'risk_percentage': 0.9}   # 10x leverage, 90% risk
    ]
    
    for action_dict in aggressive_actions:
        action = {
            'leverage': np.array([action_dict['leverage']]),
            'risk_percentage': np.array([action_dict['risk_percentage']])
        }
        env.step(action)
        
        print(f"  Leverage: {action_dict['leverage']:.1f}x, Risk: {action_dict['risk_percentage']:.1%}")
        print(f"  Position: {env.position_size:.6f} BTC, Equity: ${env.equity:.2f}")
    
    aggressive_final_equity = env.equity
    
    print(f"\\nComparison:")
    print(f"  Conservative final equity: ${conservative_final_equity:.2f}")
    print(f"  Aggressive final equity: ${aggressive_final_equity:.2f}")


def demonstrate_action_space_flexibility():
    """Demonstrate the flexibility of the new action space"""
    
    print("\\n\\nDemonstrating Action Space Flexibility")
    print("=" * 50)
    
    df = create_test_data(120, 50000.0)  # Larger dataset
    
    env = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=15.0,
        use_advanced_action_space=True
    )
    
    env.reset()
    
    # Demonstrate different trading strategies in sequence
    strategies = [
        {
            'name': 'Market Testing',
            'action': {'leverage': 1.0, 'risk_percentage': 0.05},
            'description': 'Small position to test market direction'
        },
        {
            'name': 'Trend Following',
            'action': {'leverage': 5.0, 'risk_percentage': 0.4},
            'description': 'Medium risk trend following'
        },
        {
            'name': 'Counter Trend',
            'action': {'leverage': -3.0, 'risk_percentage': 0.2},
            'description': 'Counter-trend with controlled risk'
        },
        {
            'name': 'High Conviction',
            'action': {'leverage': 10.0, 'risk_percentage': 0.6},
            'description': 'High leverage with substantial risk'
        },
        {
            'name': 'Risk Reduction',
            'action': {'leverage': 2.0, 'risk_percentage': 0.1},
            'description': 'Reduce exposure and risk'
        }
    ]
    
    for i, strategy in enumerate(strategies):
        action = {
            'leverage': np.array([strategy['action']['leverage']]),
            'risk_percentage': np.array([strategy['action']['risk_percentage']])
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_price = env.price_data.iloc[env.current_step]['close']
        
        print(f"\\nStrategy {i+1}: {strategy['name']}")
        print(f"  Description: {strategy['description']}")
        print(f"  Action: {strategy['action']['leverage']:.1f}x leverage, {strategy['action']['risk_percentage']:.1%} risk")
        print(f"  Result: Position {env.position_size:.6f} BTC, Equity ${env.equity:.2f}")
        print(f"  Price: ${current_price:.2f}, Reward: {reward:.4f}")
        
        if terminated or truncated:
            break


def compare_action_spaces():
    """Compare performance between legacy and advanced action spaces"""
    
    print("\\n\\nComparing Action Space Performance")
    print("=" * 50)
    
    # Use the same data for fair comparison
    df = create_test_data(100, 50000.0)
    
    # Test legacy action space
    legacy_equity = test_legacy_action_space_simple(df)
    
    # Test advanced action space
    advanced_equity = test_advanced_action_space_simple(df)
    
    print(f"\\nPerformance Comparison:")
    print(f"  Legacy action space final equity: ${legacy_equity:.2f}")
    print(f"  Advanced action space final equity: ${advanced_equity:.2f}")
    print(f"  Difference: ${advanced_equity - legacy_equity:.2f}")
    
    if advanced_equity > legacy_equity:
        improvement = ((advanced_equity - legacy_equity) / legacy_equity) * 100
        print(f"  Advanced action space performed {improvement:.2f}% better")
    else:
        decline = ((legacy_equity - advanced_equity) / legacy_equity) * 100
        print(f"  Legacy action space performed {decline:.2f}% better")


def test_legacy_action_space_simple(df):
    """Simple test for legacy action space"""
    env = FuturesTradingEnv(df=df, initial_equity=10000.0, max_leverage=10.0, use_advanced_action_space=False, window_size=50)
    env.reset()
    
    actions = [5.0, -3.0, 8.0, -2.0, 0.0]
    for action in actions:
        env.step(np.array([action]))
    
    return env.equity


def test_advanced_action_space_simple(df):
    """Simple test for advanced action space"""
    env = FuturesTradingEnv(df=df, initial_equity=10000.0, max_leverage=10.0, use_advanced_action_space=True, window_size=50)
    env.reset()
    
    actions = [
        {'leverage': 5.0, 'risk_percentage': 0.5},
        {'leverage': -3.0, 'risk_percentage': 0.3},
        {'leverage': 8.0, 'risk_percentage': 0.6},
        {'leverage': -2.0, 'risk_percentage': 0.2},
        {'leverage': 0.0, 'risk_percentage': 0.0}
    ]
    
    for action_dict in actions:
        action = {
            'leverage': np.array([action_dict['leverage']]),
            'risk_percentage': np.array([action_dict['risk_percentage']])
        }
        env.step(action)
    
    return env.equity


if __name__ == "__main__":
    try:
        # Test legacy action space
        legacy_final_equity = test_legacy_action_space()
        
        # Test advanced action space
        advanced_final_equity = test_advanced_action_space()
        
        # Test risk management scenarios
        test_risk_management_scenarios()
        
        # Demonstrate flexibility
        demonstrate_action_space_flexibility()
        
        # Compare performance
        compare_action_spaces()
        
        print("\\n" + "=" * 60)
        print("Advanced Action Space Testing Complete!")
        print("\\n🎯 Key Features Demonstrated:")
        print("- Backward compatibility with legacy action space")
        print("- Flexible risk percentage control (1% to 100% of equity)")
        print("- Independent leverage and risk management")
        print("- Enhanced trading strategy possibilities")
        print("- Professional-grade position sizing")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
