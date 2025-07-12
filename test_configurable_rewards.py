"""
Test script to demonstrate configurable reward function parameters.

This script shows how to customize reward function behavior by adjusting parameters
instead of hardcoded "magic numbers".
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv


def create_test_data(n_samples: int = 500, initial_price: float = 50000.0) -> pd.DataFrame:
    """Create synthetic price data for testing reward configurations"""
    
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


def test_default_vs_custom_rewards():
    """Compare default reward configuration vs custom configurations"""
    
    print("Testing Configurable Reward Function")
    print("=" * 60)
    
    # Create test data
    df = create_test_data(200, 50000.0)
    
    # Test 1: Default reward configuration
    print("\n1. Default Reward Configuration")
    print("-" * 40)
    
    env_default = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=20.0,
        window_size=60,
        use_advanced_action_space=True  # Use advanced action space
    )
    
    # Test 2: Conservative reward configuration (lower penalties)
    print("\n2. Conservative Reward Configuration")
    print("-" * 40)
    
    conservative_config = {
        # Reduce all penalties by 50%
        'severe_drawdown_penalty': 10.0,     # Default: 20.0
        'major_drawdown_penalty': 5.0,       # Default: 10.0
        'moderate_drawdown_penalty': 2.5,    # Default: 5.0
        'critical_equity_penalty': 25.0,     # Default: 50.0
        'severe_equity_penalty': 15.0,       # Default: 30.0
        'liquidation_penalty': 12.5,         # Default: 25.0
        'consecutive_loss_cap': 7.5,         # Default: 15.0
        
        # Increase positive bonuses
        'position_hold_bonus': 1.0,          # Default: 0.5
        'consecutive_wins_multiplier': 0.4,  # Default: 0.2
        'recovery_multiplier': 30,           # Default: 20
    }
    
    env_conservative = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=20.0,
        window_size=60,
        reward_config=conservative_config,
        use_advanced_action_space=True
    )
    
    # Test 3: Aggressive reward configuration (higher penalties)
    print("\n3. Aggressive Reward Configuration")
    print("-" * 40)
    
    aggressive_config = {
        # Increase all penalties by 50%
        'severe_drawdown_penalty': 30.0,     # Default: 20.0
        'major_drawdown_penalty': 15.0,      # Default: 10.0
        'moderate_drawdown_penalty': 7.5,    # Default: 5.0
        'critical_equity_penalty': 75.0,     # Default: 50.0
        'severe_equity_penalty': 45.0,       # Default: 30.0
        'liquidation_penalty': 37.5,         # Default: 25.0
        'consecutive_loss_cap': 22.5,        # Default: 15.0
        
        # Tighter thresholds
        'severe_drawdown_threshold': 0.3,    # Default: 0.5 (trigger at 30% instead of 50%)
        'major_drawdown_threshold': 0.2,     # Default: 0.3 (trigger at 20% instead of 30%)
        'excessive_leverage_threshold': 15,  # Default: 20 (penalize 15x+ instead of 20x+)
        
        # Reduce positive bonuses
        'position_hold_bonus': 0.25,         # Default: 0.5
        'consecutive_wins_multiplier': 0.1,  # Default: 0.2
    }
    
    env_aggressive = FuturesTradingEnv(
        df=df,
        initial_equity=10000.0,
        max_leverage=20.0,
        window_size=60,
        reward_config=aggressive_config,
        use_advanced_action_space=True
    )
    
    # Run simulation with each configuration
    environments = [
        ("Default", env_default),
        ("Conservative", env_conservative),
        ("Aggressive", env_aggressive)
    ]
    
    for config_name, env in environments:
        print(f"\nTesting {config_name} Configuration:")
        
        # Reset environment
        obs, info = env.reset()
        
        # Take some risky trades to trigger different reward components
        actions = [
            {'leverage': 15.0, 'risk_percentage': 0.8},  # High leverage, high risk
            {'leverage': -10.0, 'risk_percentage': 0.5}, # Moderate short
            {'leverage': 8.0, 'risk_percentage': 0.3},   # Conservative long
        ]
        
        total_reward = 0.0
        equity_history = [env.equity]
        
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            equity_history.append(env.equity)
            
            print(f"  Step {i+1}: Action={action}, Reward={reward:.3f}, Equity=${env.equity:.2f}")
            
            if terminated or truncated:
                break
        
        # Run a few more steps to see longer-term effects
        for i in range(5):
            if env.position_size != 0:  # Hold position if we have one
                action = {'leverage': 0.0, 'risk_percentage': 0.1}  # Neutral action
            else:  # Take a small position
                action = {'leverage': 5.0, 'risk_percentage': 0.2}
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            equity_history.append(env.equity)
            
            if terminated or truncated:
                break
        
        # Summary
        final_equity = env.equity
        total_return = (final_equity - env.initial_equity) / env.initial_equity
        
        # Calculate max drawdown from equity history
        if len(equity_history) > 1:
            equity_array = np.array(equity_history)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0.0
        
        print(f"  Summary:")
        print(f"    Total Reward: {total_reward:.3f}")
        print(f"    Final Equity: ${final_equity:.2f}")
        print(f"    Total Return: {total_return:.2%}")
        print(f"    Max Drawdown: {max_drawdown:.2%}")


def test_reward_parameter_sensitivity():
    """Test sensitivity to specific reward parameters"""
    
    print("\n\nReward Parameter Sensitivity Analysis")
    print("=" * 60)
    
    df = create_test_data(100, 50000.0)
    
    # Test different liquidation penalty values
    print("\nLiquidation Penalty Sensitivity:")
    print("-" * 35)
    
    liquidation_penalties = [10.0, 25.0, 50.0, 100.0]  # Default is 25.0
    
    for penalty in liquidation_penalties:
        config = {'liquidation_penalty': penalty}
        
        env = FuturesTradingEnv(
            df=df,
            initial_equity=5000.0,  # Smaller equity for easier liquidation
            max_leverage=25.0,
            window_size=60,
            reward_config=config,
            use_advanced_action_space=True
        )
        
        env.reset()
        
        # Force a liquidation scenario with very high leverage
        risky_action = {'leverage': 25.0, 'risk_percentage': 0.95}
        
        reward_history = []
        for i in range(20):  # Run until liquidation or end
            obs, reward, terminated, truncated, info = env.step(risky_action)
            reward_history.append(reward)
            
            if env.liquidated:
                liquidation_reward = reward
                print(f"  Penalty {penalty:5.1f}: Liquidated at step {i+1}, "
                      f"Final reward: {liquidation_reward:.2f}")
                break
            
            if terminated or truncated:
                print(f"  Penalty {penalty:5.1f}: Completed without liquidation, "
                      f"Total reward: {sum(reward_history):.2f}")
                break
    
    # Test different drawdown thresholds
    print("\nDrawdown Threshold Sensitivity:")
    print("-" * 35)
    
    severe_thresholds = [0.3, 0.4, 0.5, 0.6]  # Default is 0.5 (50%)
    
    for threshold in severe_thresholds:
        config = {'severe_drawdown_threshold': threshold}
        
        env = FuturesTradingEnv(
            df=df,
            initial_equity=10000.0,
            max_leverage=20.0,
            window_size=60,
            reward_config=config,
            use_advanced_action_space=True
        )
        
        env.reset()
        
        # Simulate losses to trigger drawdown
        losing_action = {'leverage': -15.0, 'risk_percentage': 0.7}  # Aggressive short
        
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(losing_action)
            
            if len(env.equity_history) > 1:
                current_drawdown = (env.max_equity - env.equity) / env.max_equity
                if current_drawdown > threshold * 0.8:  # Close to threshold
                    print(f"  Threshold {threshold:.1f}: Drawdown {current_drawdown:.2%}, "
                          f"Reward: {reward:.2f}")
                    break
            
            if terminated or truncated:
                break


def demonstrate_reward_config_usage():
    """Show practical examples of reward configuration usage"""
    
    print("\n\nPractical Reward Configuration Examples")
    print("=" * 60)
    
    df = create_test_data(150, 50000.0)
    
    # Example 1: Day Trading Configuration (fast trades, lower hold bonuses)
    print("\n1. Day Trading Configuration:")
    print("-" * 30)
    
    day_trading_config = {
        'optimal_hold_min': 1,           # Very short holds OK
        'optimal_hold_max': 8,           # Don't hold too long
        'position_hold_bonus': 0.2,     # Lower hold bonus
        'position_hold_penalty': 0.8,   # Higher penalty for long holds
        'excessive_hold_threshold': 12,  # Penalize holds > 3 hours
        'volatility_penalty_cap': 2.0,  # Lower volatility penalty (day traders expect volatility)
    }
    
    env_day = FuturesTradingEnv(df=df, initial_equity=10000.0, reward_config=day_trading_config,
                               use_advanced_action_space=True, window_size=60)
    env_day.reset()
    print("  Day trading environment created with fast-trade optimized rewards")
    
    # Example 2: Swing Trading Configuration (longer holds encouraged)
    print("\n2. Swing Trading Configuration:")
    print("-" * 32)
    
    swing_trading_config = {
        'optimal_hold_min': 12,          # Encourage longer holds (3+ hours)
        'optimal_hold_max': 48,          # Up to 12 hours optimal
        'position_hold_bonus': 1.0,     # Higher hold bonus
        'position_hold_penalty': 0.1,   # Lower penalty for long holds
        'excessive_hold_threshold': 72,  # Don't penalize until 18+ hours
        'recovery_bonus_cap': 5.0,      # Higher recovery bonus cap
    }
    
    env_swing = FuturesTradingEnv(df=df, initial_equity=10000.0, reward_config=swing_trading_config,
                                 use_advanced_action_space=True, window_size=60)
    env_swing.reset()
    print("  Swing trading environment created with long-hold optimized rewards")
    
    # Example 3: Risk-Averse Configuration (heavy penalties for losses)
    print("\n3. Risk-Averse Configuration:")
    print("-" * 31)
    
    risk_averse_config = {
        'excessive_leverage_threshold': 5,    # Penalize leverage > 5x
        'excessive_leverage_multiplier': 2.0, # Heavy leverage penalty
        'severe_drawdown_threshold': 0.2,     # Trigger severe penalty at 20% drawdown
        'moderate_drawdown_threshold': 0.05,  # Trigger moderate penalty at 5% drawdown
        'liquidation_penalty': 100.0,         # Very heavy liquidation penalty
        'final_reward_negative_cap': -50.0,   # Allow more negative rewards
    }
    
    env_safe = FuturesTradingEnv(df=df, initial_equity=10000.0, reward_config=risk_averse_config,
                                use_advanced_action_space=True, window_size=60)
    env_safe.reset()
    print("  Risk-averse environment created with heavy loss penalties")
    
    print("\n4. How to Use Custom Configurations:")
    print("-" * 37)
    print("""
  # Create custom reward configuration
  my_config = {
      'liquidation_penalty': 50.0,      # Increase liquidation penalty
      'position_hold_bonus': 2.0,       # Increase hold bonus
      'recovery_multiplier': 25,        # Increase recovery bonus
  }
  
  # Apply to environment
  env = FuturesTradingEnv(
      df=your_data,
      initial_equity=10000.0,
      reward_config=my_config,
      use_advanced_action_space=True
  )
  
  # The environment will use your custom values and defaults for unspecified parameters
    """)


if __name__ == "__main__":
    try:
        test_default_vs_custom_rewards()
        test_reward_parameter_sensitivity()
        demonstrate_reward_config_usage()
        
        print("\n" + "=" * 60)
        print("Configurable reward function testing complete!")
        print("\nKey benefits:")
        print("- All 'magic numbers' are now configurable parameters")
        print("- Easy hyperparameter optimization")
        print("- Support for different trading strategies (day/swing/safe)")
        print("- Backward compatible (defaults match original behavior)")
        print("- Clear parameter names and documentation")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
