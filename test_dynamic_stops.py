"""
Test script to demonstrate Dynamic Stop-Loss and Take-Profit based on ATR
This system adapts risk management to market volatility conditions.
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_environment import FuturesTradingEnv


def create_volatility_scenarios():
    """Create datasets with different volatility patterns"""
    np.random.seed(42)
    base_price = 50000.0
    n_samples = 200
    
    scenarios = {}
    
    # Scenario 1: Low Volatility (Calm Market)
    low_vol_returns = np.random.normal(0.0, 0.005, n_samples)  # 0.5% volatility
    scenarios['low_volatility'] = generate_price_data(base_price, low_vol_returns, "Low Volatility")
    
    # Scenario 2: Medium Volatility (Normal Market)
    med_vol_returns = np.random.normal(0.0, 0.015, n_samples)  # 1.5% volatility
    scenarios['medium_volatility'] = generate_price_data(base_price, med_vol_returns, "Medium Volatility")
    
    # Scenario 3: High Volatility (Turbulent Market)
    high_vol_returns = np.random.normal(0.0, 0.035, n_samples)  # 3.5% volatility
    scenarios['high_volatility'] = generate_price_data(base_price, high_vol_returns, "High Volatility")
    
    # Scenario 4: Mixed Volatility (Changing Conditions)
    mixed_returns = []
    for i in range(n_samples):
        if i < 50:  # Low vol period
            vol = 0.005
        elif i < 100:  # High vol period
            vol = 0.04
        elif i < 150:  # Medium vol period
            vol = 0.02
        else:  # Low vol again
            vol = 0.008
        mixed_returns.append(np.random.normal(0.0, vol))
    
    scenarios['mixed_volatility'] = generate_price_data(base_price, mixed_returns, "Mixed Volatility")
    
    return scenarios


def generate_price_data(initial_price: float, returns: list, scenario_name: str) -> pd.DataFrame:
    """Generate OHLCV data from returns"""
    
    n_samples = len(returns)
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15T')
    
    # Create price series
    prices = [initial_price]
    for i in range(1, n_samples):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    # Generate OHLC from close prices with realistic spreads
    highs = prices * (1 + np.random.uniform(0.001, 0.005, n_samples))
    lows = prices * (1 - np.random.uniform(0.001, 0.005, n_samples))
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


def test_dynamic_vs_fixed_stops():
    """Compare dynamic vs fixed stop-loss/take-profit performance"""
    
    print("Dynamic vs Fixed Stop-Loss/Take-Profit Comparison")
    print("=" * 60)
    
    scenarios = create_volatility_scenarios()
    results = {}
    
    for scenario_name, df in scenarios.items():
        print(f"\n📊 Testing {scenario_name.replace('_', ' ').title()} Market")
        print("-" * 50)
        
        # Test 1: Fixed stops (traditional approach)
        env_fixed = FuturesTradingEnv(
            df=df,
            initial_equity=10000.0,
            use_dynamic_stops=False,  # Use fixed percentages
            stop_loss_pct=0.02,       # Fixed 2%
            take_profit_pct=0.04,     # Fixed 4%
            use_advanced_action_space=True,
            window_size=50
        )
        
        # Test 2: Dynamic stops (ATR-based)
        env_dynamic = FuturesTradingEnv(
            df=df,
            initial_equity=10000.0,
            use_dynamic_stops=True,         # Enable dynamic stops
            atr_stop_loss_multiplier=2.0,   # Stop-loss = 2 * ATR
            atr_take_profit_multiplier=3.0, # Take-profit = 3 * ATR
            min_stop_loss_pct=0.005,        # Min 0.5%
            max_stop_loss_pct=0.08,         # Max 8%
            min_take_profit_pct=0.01,       # Min 1%
            max_take_profit_pct=0.15,       # Max 15%
            use_advanced_action_space=True,
            window_size=50
        )
        
        # Run simulations
        fixed_results = run_trading_simulation(env_fixed, "Fixed Stops")
        dynamic_results = run_trading_simulation(env_dynamic, "Dynamic Stops")
        
        results[scenario_name] = {
            'fixed': fixed_results,
            'dynamic': dynamic_results
        }
        
        # Print comparison
        print(f"\n  Fixed Stops Performance:")
        print(f"    Final Equity: ${fixed_results['final_equity']:,.2f}")
        print(f"    Total Return: {fixed_results['total_return']:.2%}")
        print(f"    Avg Stop Loss: {fixed_results['avg_stop_pct']:.2%}")
        print(f"    Avg Take Profit: {fixed_results['avg_tp_pct']:.2%}")
        print(f"    Trades Executed: {fixed_results['trades_count']}")
        
        print(f"\n  Dynamic Stops Performance:")
        print(f"    Final Equity: ${dynamic_results['final_equity']:,.2f}")
        print(f"    Total Return: {dynamic_results['total_return']:.2%}")
        print(f"    Avg Stop Loss: {dynamic_results['avg_stop_pct']:.2%}")
        print(f"    Avg Take Profit: {dynamic_results['avg_tp_pct']:.2%}")
        print(f"    Trades Executed: {dynamic_results['trades_count']}")
        
        # Calculate improvement
        performance_improvement = dynamic_results['total_return'] - fixed_results['total_return']
        print(f"\n  💡 Dynamic vs Fixed Improvement: {performance_improvement:.2%}")
    
    return results


def run_trading_simulation(env, method_name: str) -> dict:
    """Run a trading simulation and collect statistics"""
    
    env.reset()
    
    equity_history = [env.equity]
    stop_percentages = []
    tp_percentages = []
    trades_count = 0
    
    # Simple trading strategy: alternating long/short positions
    for i in range(min(100, len(env.price_data) - env.window_size - 10)):
        
        # Alternate between long and short positions with varying risk
        if i % 20 < 10:  # Long position
            action = {'leverage': 8.0 + (i % 3) * 2, 'risk_percentage': 0.3 + (i % 3) * 0.1}
        else:  # Short position
            action = {'leverage': -(6.0 + (i % 3) * 2), 'risk_percentage': 0.25 + (i % 3) * 0.1}
        
        obs, reward, terminated, truncated, info = env.step(action)
        equity_history.append(env.equity)
        
        # Collect stop-loss and take-profit information
        if abs(env.position_size) > 0.001:
            stops_info = env.get_dynamic_stops_info()
            if stops_info.get('mode') == 'dynamic':
                stop_percentages.append(stops_info.get('dynamic_stop_loss_pct', 0) * 100)
                tp_percentages.append(stops_info.get('dynamic_take_profit_pct', 0) * 100)
            else:
                stop_percentages.append(env.stop_loss_pct * 100)
                tp_percentages.append(env.take_profit_pct * 100)
            
            trades_count += 1
        
        if terminated or truncated:
            break
    
    final_equity = env.equity
    total_return = (final_equity - env.initial_equity) / env.initial_equity
    
    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'equity_history': equity_history,
        'avg_stop_pct': np.mean(stop_percentages) if stop_percentages else 0,
        'avg_tp_pct': np.mean(tp_percentages) if tp_percentages else 0,
        'stop_percentages': stop_percentages,
        'tp_percentages': tp_percentages,
        'trades_count': trades_count
    }


def demonstrate_atr_adaptation():
    """Demonstrate how ATR-based stops adapt to market conditions"""
    
    print("\n\nATR-Based Stop Adaptation Demonstration")
    print("=" * 60)
    
    # Create a dataset with changing volatility
    scenarios = create_volatility_scenarios()
    mixed_vol_data = scenarios['mixed_volatility']
    
    env = FuturesTradingEnv(
        df=mixed_vol_data,
        initial_equity=10000.0,
        use_dynamic_stops=True,
        atr_stop_loss_multiplier=2.5,   # Slightly more conservative
        atr_take_profit_multiplier=3.5,
        min_stop_loss_pct=0.003,        # 0.3% minimum
        max_stop_loss_pct=0.12,         # 12% maximum
        use_advanced_action_space=True,
        window_size=50
    )
    
    env.reset()
    
    print("\nReal-time ATR adaptation during different market phases:")
    print("-" * 55)
    
    adaptation_data = []
    
    for i in range(0, min(80, len(env.price_data) - env.window_size - 10), 10):
        # Take a position
        action = {'leverage': 10.0, 'risk_percentage': 0.4}
        obs, reward, terminated, truncated, info = env.step(action)
        
        if abs(env.position_size) > 0.001:
            current_price = env.price_data.iloc[env.current_step]['close']
            stops_info = env.get_dynamic_stops_info()
            
            atr_value = stops_info.get('current_atr', 0)
            atr_pct = stops_info.get('atr_percentage', 0) * 100
            stop_pct = stops_info.get('dynamic_stop_loss_pct', 0) * 100
            tp_pct = stops_info.get('dynamic_take_profit_pct', 0) * 100
            
            # Determine market phase based on step
            if i < 20:
                phase = "Low Vol"
            elif i < 40:
                phase = "High Vol"
            elif i < 60:
                phase = "Medium Vol"
            else:
                phase = "Calm"
            
            print(f"Step {i:2d} | {phase:9s} | Price: ${current_price:6.0f} | "
                  f"ATR: ${atr_value:4.0f} ({atr_pct:4.1f}%) | "
                  f"Stop: {stop_pct:4.1f}% | TP: {tp_pct:4.1f}%")
            
            adaptation_data.append({
                'step': i,
                'phase': phase,
                'price': current_price,
                'atr_value': atr_value,
                'atr_percentage': atr_pct,
                'stop_loss_pct': stop_pct,
                'take_profit_pct': tp_pct
            })
        
        if terminated or truncated:
            break
    
    return adaptation_data


def test_different_atr_multipliers():
    """Test different ATR multiplier configurations"""
    
    print("\n\nATR Multiplier Configuration Testing")
    print("=" * 60)
    
    # Use medium volatility scenario for consistent comparison
    scenarios = create_volatility_scenarios()
    test_data = scenarios['medium_volatility']
    
    multiplier_configs = [
        (1.5, 2.5, "Conservative"),  # Tight stops
        (2.0, 3.0, "Balanced"),     # Moderate stops
        (2.5, 4.0, "Aggressive"),   # Wide stops
        (3.0, 5.0, "Very Wide"),    # Very wide stops
    ]
    
    print("\nTesting different ATR multiplier configurations:")
    print("-" * 50)
    
    for stop_mult, tp_mult, config_name in multiplier_configs:
        env = FuturesTradingEnv(
            df=test_data,
            initial_equity=10000.0,
            use_dynamic_stops=True,
            atr_stop_loss_multiplier=stop_mult,
            atr_take_profit_multiplier=tp_mult,
            use_advanced_action_space=True,
            window_size=50
        )
        
        results = run_trading_simulation(env, config_name)
        
        print(f"\n{config_name:12s} (SL: {stop_mult}x, TP: {tp_mult}x ATR):")
        print(f"  Final Equity: ${results['final_equity']:8,.2f}")
        print(f"  Total Return: {results['total_return']:8.2%}")
        print(f"  Avg Stop:     {results['avg_stop_pct']:8.2f}%")
        print(f"  Avg TP:       {results['avg_tp_pct']:8.2f}%")
        print(f"  Trades:       {results['trades_count']:8d}")


def demonstrate_risk_bounds():
    """Demonstrate how min/max bounds protect against extreme ATR values"""
    
    print("\n\nRisk Bounds Protection Demonstration")
    print("=" * 60)
    
    # Create extreme volatility scenario
    np.random.seed(123)
    n_samples = 100
    
    # Simulate extreme volatility spikes
    extreme_returns = []
    for i in range(n_samples):
        if 30 <= i <= 40:  # Extreme volatility spike
            vol = 0.15  # 15% volatility - extremely high
        elif 60 <= i <= 70:  # Another spike
            vol = 0.08  # 8% volatility
        else:
            vol = 0.01  # Normal low volatility
        extreme_returns.append(np.random.normal(0.0, vol))
    
    extreme_data = generate_price_data(50000.0, extreme_returns, "Extreme Volatility")
    
    # Test with and without bounds
    print("\nComparing bounded vs unbounded ATR-based stops:")
    print("-" * 50)
    
    # Environment with protective bounds
    env_bounded = FuturesTradingEnv(
        df=extreme_data,
        initial_equity=10000.0,
        use_dynamic_stops=True,
        atr_stop_loss_multiplier=2.0,
        atr_take_profit_multiplier=3.0,
        min_stop_loss_pct=0.005,    # 0.5% minimum
        max_stop_loss_pct=0.06,     # 6% maximum protection
        min_take_profit_pct=0.01,   # 1% minimum  
        max_take_profit_pct=0.12,   # 12% maximum
        use_advanced_action_space=True,
        window_size=30
    )
    
    # Environment with very loose bounds (simulates unbounded)
    env_loose = FuturesTradingEnv(
        df=extreme_data,
        initial_equity=10000.0,
        use_dynamic_stops=True,
        atr_stop_loss_multiplier=2.0,
        atr_take_profit_multiplier=3.0,
        min_stop_loss_pct=0.001,    # 0.1% minimum
        max_stop_loss_pct=0.30,     # 30% maximum (very loose)
        min_take_profit_pct=0.005,  # 0.5% minimum
        max_take_profit_pct=0.50,   # 50% maximum (very loose)
        use_advanced_action_space=True,
        window_size=30
    )
    
    bounded_results = run_trading_simulation(env_bounded, "Bounded")
    loose_results = run_trading_simulation(env_loose, "Loose Bounds")
    
    print(f"\nBounded Stops (Protected):")
    print(f"  Final Equity: ${bounded_results['final_equity']:,.2f}")
    print(f"  Total Return: {bounded_results['total_return']:.2%}")
    print(f"  Avg Stop:     {bounded_results['avg_stop_pct']:.2f}%")
    print(f"  Max Stop:     {max(bounded_results['stop_percentages']) if bounded_results['stop_percentages'] else 0:.2f}%")
    
    print(f"\nLoose Bounds (Risky):")
    print(f"  Final Equity: ${loose_results['final_equity']:,.2f}")
    print(f"  Total Return: {loose_results['total_return']:.2%}")
    print(f"  Avg Stop:     {loose_results['avg_stop_pct']:.2f}%")
    print(f"  Max Stop:     {max(loose_results['stop_percentages']) if loose_results['stop_percentages'] else 0:.2f}%")
    
    protection_benefit = bounded_results['total_return'] - loose_results['total_return']
    print(f"\n💡 Protection Benefit: {protection_benefit:.2%}")


if __name__ == "__main__":
    try:
        # Run comprehensive dynamic stop-loss testing
        print("🎯 Dynamic Stop-Loss and Take-Profit Testing Suite")
        print("Using ATR (Average True Range) for Market-Adaptive Risk Management")
        print("=" * 80)
        
        # Test 1: Compare dynamic vs fixed stops across different market conditions
        results = test_dynamic_vs_fixed_stops()
        
        # Test 2: Demonstrate real-time ATR adaptation
        adaptation_data = demonstrate_atr_adaptation()
        
        # Test 3: Test different ATR multiplier configurations
        test_different_atr_multipliers()
        
        # Test 4: Demonstrate protective bounds
        demonstrate_risk_bounds()
        
        print("\n" + "=" * 80)
        print("✅ Dynamic Stop-Loss and Take-Profit Testing Complete!")
        print("\n🎯 Key Benefits of Dynamic ATR-Based Stops:")
        print("- Adapts to market volatility automatically")
        print("- Tighter stops in calm markets (less noise)")
        print("- Wider stops in volatile markets (avoid premature exits)")
        print("- Configurable multipliers for different strategies")
        print("- Protective bounds prevent extreme values")
        print("- Better risk-reward ratio adaptation")
        
        print("\n📊 Configuration Guidelines:")
        print("- Conservative: 1.5x-2.5x ATR (tight stops)")
        print("- Balanced: 2.0x-3.0x ATR (moderate stops)")
        print("- Aggressive: 2.5x-4.0x ATR (wide stops)")
        print("- Always use protective min/max bounds!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
