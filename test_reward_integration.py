"""
Test script to verify reward configuration integration
"""
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from improved_reward_configs import TREND_RIDER_CONFIG, MAX_PROFIT_CONFIG
from trading_environment import FuturesTradingEnv
import pandas as pd

def test_reward_config_integration():
    """Test that reward configurations are properly passed to the environment"""
    
    # Create dummy data for testing
    dummy_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
        'open': [50000 + i for i in range(100)],
        'high': [50100 + i for i in range(100)],
        'low': [49900 + i for i in range(100)],
        'close': [50050 + i for i in range(100)],
        'volume': [1000] * 100
    })
    
    print("🧪 Testing Reward Configuration Integration...")
    
    # Test 1: Default configuration (no reward_config passed)
    print("\n1️⃣ Testing DEFAULT configuration...")
    try:
        env_default = FuturesTradingEnv(
            df=dummy_data,
            initial_equity=10000,
            max_leverage=10
        )
        print("✅ Default environment created successfully")
        default_config = env_default.reward_config
        print(f"   - position_hold_bonus: {default_config.get('position_hold_bonus', 'Not found')}")
        print(f"   - cost_penalty_multiplier: {default_config.get('cost_penalty_multiplier', 'Not found')}")
        print(f"   - cancel_close_penalty: {default_config.get('cancel_close_penalty', 'Not found')}")
    except Exception as e:
        print(f"❌ Default environment failed: {e}")
    
    # Test 2: TREND_RIDER configuration
    print("\n2️⃣ Testing TREND_RIDER configuration...")
    try:
        env_trend_rider = FuturesTradingEnv(
            df=dummy_data,
            initial_equity=10000,
            max_leverage=10,
            reward_config=TREND_RIDER_CONFIG
        )
        print("✅ TREND_RIDER environment created successfully")
        trend_rider_config = env_trend_rider.reward_config
        print(f"   - position_hold_bonus: {trend_rider_config.get('position_hold_bonus', 'Not found')}")
        print(f"   - cost_penalty_multiplier: {trend_rider_config.get('cost_penalty_multiplier', 'Not found')}")
        print(f"   - profit_milestone_bonuses: {trend_rider_config.get('profit_milestone_bonuses', 'Not found')}")
    except Exception as e:
        print(f"❌ TREND_RIDER environment failed: {e}")
    
    # Test 3: MAX_PROFIT configuration
    print("\n3️⃣ Testing MAX_PROFIT configuration...")
    try:
        env_max_profit = FuturesTradingEnv(
            df=dummy_data,
            initial_equity=10000,
            max_leverage=10,
            reward_config=MAX_PROFIT_CONFIG
        )
        print("✅ MAX_PROFIT environment created successfully")
        max_profit_config = env_max_profit.reward_config
        print(f"   - position_hold_bonus: {max_profit_config.get('position_hold_bonus', 'Not found')}")
        print(f"   - cost_penalty_multiplier: {max_profit_config.get('cost_penalty_multiplier', 'Not found')}")
        print(f"   - profit_milestone_bonuses: {max_profit_config.get('profit_milestone_bonuses', 'Not found')}")
    except Exception as e:
        print(f"❌ MAX_PROFIT environment failed: {e}")
    
    print("\n🎯 Reward Configuration Integration Test Complete!")
    print("\nTo use in training:")
    print("1. Run main.py")
    print("2. Select option 1 (Train New Model) or 2 (Multi-Episode Training)")
    print("3. Choose your preferred reward configuration")
    print("4. The improved reward system will guide better trading behavior!")

if __name__ == "__main__":
    test_reward_config_integration()
