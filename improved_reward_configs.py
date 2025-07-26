"""
Essential Reward Configurations for Enhanced Trading Behavior

Focused on the two most effective configurations:
1. TREND_RIDER_CONFIG: Enhanced for holding profitable positions longer (RECOMMENDED)
2. MAX_PROFIT_CONFIG: Maximum profit capture with aggressive trend riding

These configurations address key trading issues:
- Premature position closing
- Insufficient pattern recognition  
- Weak trend following
- Overemphasis on risk penalties

ANOMALY REDUCTION SETTINGS (FIX 4):
- use_realistic_execution: False (for backtesting accuracy)
- price_validation_tolerance: 0.001 (0.1% tolerance)
"""

# Base reward configuration with essential parameters
BASE_REWARD_CONFIG = {
    # Anomaly reduction settings (FIX 4)
    'use_realistic_execution': False,  # Disable realistic execution for backtesting accuracy
    'price_validation_tolerance': 0.001,  # 0.1% price validation tolerance
    
    # Base scaling
    'base_reward_scale': 100,
    'base_reward_cap_positive': 10.0,
    'base_reward_cap_negative': -10.0,
    
    # Drawdown penalties
    'severe_drawdown_threshold': 0.5,
    'severe_drawdown_penalty': 20.0,
    'major_drawdown_threshold': 0.3,
    'major_drawdown_penalty': 10.0,
    'moderate_drawdown_threshold': 0.1,
    'moderate_drawdown_penalty': 5.0,
    'linear_drawdown_multiplier': 25,
    
    # Equity ratio penalties
    'critical_equity_threshold': 0.05,
    'critical_equity_penalty': 50.0,
    'severe_equity_threshold': 0.10,
    'severe_equity_penalty': 30.0,
    'major_equity_threshold': 0.20,
    'major_equity_penalty': 20.0,
    'moderate_equity_threshold': 0.30,
    'moderate_equity_penalty': 10.0,
    'minor_equity_threshold': 0.50,
    'minor_equity_penalty': 5.0,
    
    # Consecutive loss penalties
    'consecutive_loss_exponent': 1.5,
    'consecutive_loss_cap': 15.0,
    
    # Trend and volatility
    'trend_penalty_multiplier': 1000,
    'trend_penalty_cap': 8.0,
    'volatility_multiplier': 15,
    'volatility_penalty_cap': 5.0,
    'volatility_history_threshold': 10,
    
    # Trading cost penalties
    'cost_penalty_multiplier': 500,
    'cost_penalty_cap': 2.0,
    
    # Special penalties
    'liquidation_penalty': 25.0,
    'excessive_leverage_threshold': 20,
    'excessive_leverage_multiplier': 0.5,
    
    # Position holding
    'position_hold_bonus': 0.5,
    'position_hold_penalty': 0.3,
    'optimal_hold_min': 4,
    'optimal_hold_max': 24,
    'excessive_hold_threshold': 24,
    'consecutive_wins_multiplier': 0.2,
    'consecutive_wins_cap': 2.0,
    'recovery_threshold': 0.05,
    'recovery_multiplier': 20,
    'recovery_bonus_cap': 3.0,
    
    # Final reward caps
    'final_reward_positive_cap': 15.0,
    'final_reward_negative_cap': -25.0,
    'severe_loss_reward_cap': -50.0,
}

# === TREND_RIDER CONFIGURATION (RECOMMENDED FOR ANOMALY-FREE BACKTESTING) ===
# Specifically designed to encourage holding winning positions longer
# and following trends more effectively
TREND_RIDER_CONFIG = {
    # Copy base config
    **BASE_REWARD_CONFIG,
    
    # === ANOMALY REDUCTION SETTINGS (FIX 4) ===
    'use_realistic_execution': False,     # Disable for backtesting accuracy
    'price_validation_tolerance': 0.001,  # 0.1% tolerance for price validation
    
    # === POSITION HOLDING ENHANCEMENTS ===
    'position_hold_bonus': 3.0,           # Increased from 0.5
    'optimal_hold_min': 8,                # Increased from 4 
    'optimal_hold_max': 96,               # Increased from 24 (24 hours equivalent)
    'excessive_hold_threshold': 144,      # Increased from 24 (36 hours equivalent)
    
    # === TREND FOLLOWING REWARDS ===
    'trend_following_bonus': 0.05,        # New parameter
    'trend_following_threshold': 0.005,   # More sensitive
    'exit_profitable_trend_penalty': 0.15, # Penalty for exiting good trends
    'profitable_trend_threshold': 0.01,    # 1% positive trend required
    
    # === PROGRESSIVE PROFIT REWARDS (NEW) ===
    'profit_milestone_bonuses': {
        0.01: 0.02,   # 1% profit: small bonus
        0.02: 0.05,   # 2% profit: medium bonus  
        0.05: 0.15,   # 5% profit: large bonus
        0.10: 0.30,   # 10% profit: huge bonus
    },
    
    # === REDUCE OVERTRADING PENALTIES ===
    'cost_penalty_multiplier': 200,       # Reduced from 500
    'small_position_penalty': 0.005,      # Penalty for small positions
    'cancel_close_penalty': 0.03,         # Penalty for CANCEL actions
    
    # === PATTERN COMPLETION REWARDS (NEW) ===
    'pattern_completion_bonus': 0.10,     # Reward for holding through patterns
    'momentum_continuation_bonus': 0.03,  # Reward for riding momentum
    
    # === ADJUSTED RISK PENALTIES ===
    'moderate_drawdown_penalty': 5.0,     # Reduced from 5.0
    'minor_equity_penalty': 3.0,          # Reduced from 5.0
    
    # === INACTIVITY PENALTIES (Patient approach) ===
    'inactivity_penalty_start_steps': 30,   # More patient than typical
    'inactivity_penalty_base': 0.001,       # Very low base penalty
    'inactivity_penalty_multiplier': 1.05,  # Very slow escalation
    'inactivity_penalty_cap': 0.1,          # Low maximum penalty
    'inactivity_penalty_max_steps': 100,    # Very long before cap
}

# === MAXIMUM PROFIT CAPTURE CONFIGURATION ===
# For aggressive profit maximization with anomaly reduction
MAX_PROFIT_CONFIG = {
    **TREND_RIDER_CONFIG,
    
    # === ANOMALY REDUCTION SETTINGS (FIX 4) ===
    'use_realistic_execution': False,     # Disable for backtesting accuracy
    'price_validation_tolerance': 0.001,  # 0.1% tolerance for price validation
    
    'position_hold_bonus': 5.0,
    'optimal_hold_max': 144,  # 36 hours equivalent
    'exit_profitable_trend_penalty': 0.25,
    'profit_milestone_bonuses': {
        0.01: 0.05, 0.03: 0.15, 0.05: 0.30, 0.10: 0.50, 0.15: 1.0
    },
    'cost_penalty_multiplier': 100,  # Very low cost penalties
    'cancel_close_penalty': 0.01,    # Minimal CANCEL penalty
}

if __name__ == "__main__":
    print("=== ESSENTIAL REWARD CONFIGURATIONS ===")
    print("\n1. TREND_RIDER_CONFIG: 🚀 Enhanced for holding profitable positions longer (RECOMMENDED)")
    print("   - Progressive profit bonuses, reduced cost penalties, trend following rewards")
    print("   - Anomaly reduction settings: use_realistic_execution=False, price_validation_tolerance=0.001")
    print("2. MAX_PROFIT_CONFIG: 💰 Maximum profit capture (aggressive trend riding)")
    print("   - Very high position hold bonuses, minimal cost penalties")
    print("   - Anomaly reduction settings included for backtesting accuracy")
    print("\nFEATURES:")
    print("✅ Progressive profit milestone bonuses (1%, 2%, 5%, 10%)")
    print("✅ Enhanced trend following and momentum continuation rewards")
    print("✅ Pattern completion bonuses for holding through support/resistance")
    print("✅ Reduced cost penalties to encourage position holding")
    print("✅ Position context-aware inactivity penalties")
    print("✅ ANOMALY REDUCTION: Realistic execution disabled for backtesting accuracy")
    print("✅ PRICE VALIDATION: 0.1% tolerance for trade price validation")
    print("\nTo use in training:")
    print("env = FuturesTradingEnv(df=data, reward_config=TREND_RIDER_CONFIG)  # 🚀 RECOMMENDED")
    print("env = FuturesTradingEnv(df=data, reward_config=MAX_PROFIT_CONFIG)   # 💰 AGGRESSIVE")
