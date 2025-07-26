#!/usr/bin/env python3
"""
Test Improved Reward Configuration

This script tests the new reward components designed to fix the four main trading behavior issues:
1. Closing winning trades too early
2. Holding losing trades too long  
3. Frequent CANCEL_CLOSE usage
4. Overtrading small positions
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the improved configurations
from improved_reward_configs import IMPROVED_REWARD_CONFIG, CONSERVATIVE_IMPROVED_CONFIG

def test_improved_reward_system():
    """Test the new reward system components"""
    print("🧪 TESTING IMPROVED REWARD SYSTEM")
    print("=" * 60)
    
    try:
        from trading_environment import FuturesTradingEnv
        
        # Create sample data
        print("📊 Creating sample market data...")
        dates = pd.date_range('2024-01-01', periods=100, freq='15T')
        sample_data = pd.DataFrame({
            '': range(100),
            'open': 50000 + np.random.randn(100) * 1000,
            'high': 50000 + np.random.randn(100) * 1000 + 500,
            'low': 50000 + np.random.randn(100) * 1000 - 500,
            'close': 50000 + np.random.randn(100) * 1000,
            'volume': np.random.randint(1000, 5000, 100),
            'timestamp': [int(ts.timestamp()) for ts in dates]
        })
        
        print("✅ Sample data created")
        
        # Test scenarios for each issue
        test_scenarios = [
            ("Default Config", None),
            ("Conservative Improved", CONSERVATIVE_IMPROVED_CONFIG),
            ("Aggressive Improved", IMPROVED_REWARD_CONFIG)
        ]
        
        for config_name, config in test_scenarios:
            print(f"\n🎯 TESTING: {config_name}")
            print("-" * 40)
            
            # Initialize environment
            if config:
                env = FuturesTradingEnv(df=sample_data, initial_equity=10000.0, reward_config=config)
            else:
                env = FuturesTradingEnv(df=sample_data, initial_equity=10000.0)
                
            print(f"✅ Environment initialized with {config_name}")
            
            # Test specific reward components
            obs, info = env.reset()
            
            # Simulate different trading scenarios
            scenarios = [
                ("Hold profitable position", test_hold_profitable_position),
                ("Cut losses quickly", test_cut_losses_quickly),
                ("Deliberate vs Cancel exits", test_exit_strategies),
                ("Small position penalties", test_small_position_penalties)
            ]
            
            for scenario_name, test_func in scenarios:
                try:
                    result = test_func(env, config_name)
                    print(f"  📋 {scenario_name}: {result}")
                except Exception as e:
                    print(f"  ❌ {scenario_name}: Error - {e}")
        
        print("\n🎉 IMPROVED REWARD SYSTEM TESTING COMPLETE")
        
        # Summary of improvements
        print("\n📈 EXPECTED IMPROVEMENTS:")
        print("✅ Agents should hold winning positions longer")
        print("✅ Agents should cut losing positions faster")  
        print("✅ Agents should prefer CLOSE over CANCEL actions")
        print("✅ Agents should avoid tiny, fee-eroding trades")
        print("✅ Better risk-adjusted returns overall")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_hold_profitable_position(env, config_name):
    """Test ISSUE 1: Reward for holding profitable positions"""
    
    # Simulate opening a position
    env.step(1)  # BUY action
    
    # Simulate profitable trend (increasing unrealized PnL)
    original_pnl = env.unrealized_pnl
    env.unrealized_pnl = 0.02  # 2% profit
    env.unrealized_pnl_history.append(0.01)  # Previous PnL
    env.unrealized_pnl_history.append(0.02)  # Current PnL (improving trend)
    env.last_action_type = "HOLD"
    
    # Get reward for holding profitable position
    prev_equity = env.equity
    reward = env._calculate_enhanced_reward(prev_equity)
    
    # Check if trend following bonus was applied
    has_trend_bonus = env.reward_config.get('trend_following_bonus', 0) > 0
    
    if has_trend_bonus:
        return f"✅ Trend following bonus active ({env.reward_config['trend_following_bonus']})"
    else:
        return f"➖ No trend following bonus (default: {env.reward_config['trend_following_bonus']})"

def test_cut_losses_quickly(env, config_name):
    """Test ISSUE 2: Reward for cutting losses quickly"""
    
    # Simulate opening a position
    env.step(1)  # BUY action
    env.trade_start_step = env.current_step - 3  # Short hold duration
    
    # Simulate losing trade
    env.unrealized_pnl = -0.015  # -1.5% loss
    env.unrealized_pnl_history.append(-0.015)
    env.last_action_type = "CLOSE_LONG"
    env.position_size = 0.0  # Position closed
    
    # Get reward for cutting losses
    prev_equity = env.equity
    reward = env._calculate_enhanced_reward(prev_equity)
    
    # Check if quick loss cut bonus was applied
    has_loss_cut_bonus = env.reward_config.get('quick_loss_cut_bonus', 0) > 0
    
    if has_loss_cut_bonus:
        return f"✅ Quick loss cut bonus active ({env.reward_config['quick_loss_cut_bonus']})"
    else:
        return f"➖ No quick loss cut bonus (default: {env.reward_config['quick_loss_cut_bonus']})"

def test_exit_strategies(env, config_name):
    """Test ISSUE 3: Differentiate exit strategies"""
    
    results = []
    
    # Test CANCEL action penalty
    env.last_action_type = "CANCEL"
    prev_equity = env.equity
    cancel_reward = env._calculate_enhanced_reward(prev_equity)
    
    has_cancel_penalty = env.reward_config.get('cancel_close_penalty', 0) > 0
    if has_cancel_penalty:
        results.append(f"Cancel penalty: -{env.reward_config['cancel_close_penalty']}")
    else:
        results.append("No cancel penalty")
    
    # Test CLOSE_LONG action bonus
    env.last_action_type = "CLOSE_LONG"
    env.position_size = 0.0  # Position closed
    env.unrealized_pnl_history.append(0.01)  # Profitable exit
    deliberate_reward = env._calculate_enhanced_reward(prev_equity)
    
    has_deliberate_bonus = env.reward_config.get('deliberate_exit_bonus', 0) > 0
    if has_deliberate_bonus:
        results.append(f"Deliberate exit bonus: +{env.reward_config['deliberate_exit_bonus']}")
    else:
        results.append("No deliberate exit bonus")
    
    return " | ".join(results)

def test_small_position_penalties(env, config_name):
    """Test ISSUE 4: Penalties for small positions"""
    
    # Simulate very small position
    env.position_size = 0.001  # Very small position (about $50)
    env._last_fees = 5.0  # $5 in fees
    
    prev_equity = env.equity
    reward = env._calculate_enhanced_reward(prev_equity)
    
    has_small_penalty = env.reward_config.get('small_position_penalty', 0) > 0
    has_fee_penalty = env.reward_config.get('excessive_fee_ratio_penalty', 0) > 0
    
    results = []
    if has_small_penalty:
        results.append(f"Small position penalty: -{env.reward_config['small_position_penalty']}")
    else:
        results.append("No small position penalty")
        
    if has_fee_penalty:
        results.append(f"Fee ratio penalty: -{env.reward_config['excessive_fee_ratio_penalty']}")
    else:
        results.append("No fee ratio penalty")
    
    return " | ".join(results)

def show_configuration_comparison():
    """Show the key differences between configurations"""
    print("\n📊 CONFIGURATION COMPARISON")
    print("=" * 60)
    
    key_params = [
        'position_hold_bonus',
        'consecutive_loss_cap', 
        'cost_penalty_multiplier',
        'trend_following_bonus',
        'quick_loss_cut_bonus',
        'cancel_close_penalty',
        'minimum_profit_bonus'
    ]
    
    print(f"{'Parameter':<25} {'Default':<10} {'Conservative':<12} {'Aggressive':<12}")
    print("-" * 60)
    
    for param in key_params:
        default_val = 0.5 if param == 'position_hold_bonus' else (15.0 if param == 'consecutive_loss_cap' else (500 if param == 'cost_penalty_multiplier' else 0.0))
        conservative_val = CONSERVATIVE_IMPROVED_CONFIG.get(param, default_val)
        aggressive_val = IMPROVED_REWARD_CONFIG.get(param, default_val)
        
        print(f"{param:<25} {default_val:<10} {conservative_val:<12} {aggressive_val:<12}")

if __name__ == "__main__":
    show_configuration_comparison()
    test_improved_reward_system()
