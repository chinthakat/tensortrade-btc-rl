"""
Improved Reward Configuration for Better Trading Behavior

This configuration addresses the four main issues identified in the trading analysis:
1. Agent closing winning trades too early
2. Agent holding losing trades too long  
3. Frequent CANCEL_CLOSE usage instead of deliberate exits
4. Overtrading small, insignificant positions

Key Changes Made:
================

ISSUE 1: Closing Winners Too Early
----------------------------------
✅ Increased position_hold_bonus: 0.5 → 2.0 (4x increase)
✅ Added trend_following_bonus for growing unrealized PnL
✅ Penalty for exiting profitable trends (exit_profitable_trend_penalty)
✅ Extended optimal_hold_max: 24 → 48 steps to encourage longer holds

ISSUE 2: Holding Losers Too Long  
--------------------------------
✅ Increased consecutive_loss_penalty: 15.0 → 25.0 (67% increase)
✅ Steeper drawdown penalties (2x increases across all levels)
✅ Added quick_loss_cut_bonus for cutting losses fast
✅ Enhanced loss escalation (consecutive_loss_exponent: 1.5 → 2.0)

ISSUE 3: Frequent CANCEL_CLOSE Usage
------------------------------------
✅ Added cancel_close_penalty: 0.1 penalty for CANCEL actions
✅ Added deliberate_exit_bonus: 0.05 reward for CLOSE_LONG/CLOSE_SHORT
✅ Added profit_target_achievement_bonus for hitting targets

ISSUE 4: Overtrading Small Positions
------------------------------------
✅ Increased cost_penalty_multiplier: 500 → 1500 (3x increase)
✅ Added minimum_profit_threshold_bonus: Rewards only if net_pnl > 0.5%
✅ Added small_position_penalty: Discourages tiny position sizes
✅ Enhanced fee sensitivity
"""

# Enhanced reward configuration to fix identified trading behavior issues
IMPROVED_REWARD_CONFIG = {
    # === BASE SCALING (unchanged) ===
    'base_reward_scale': 100,
    'base_reward_cap_positive': 10.0,
    'base_reward_cap_negative': -10.0,
    
    # === ISSUE 2: HOLDING LOSERS TOO LONG - Steeper drawdown penalties ===
    'severe_drawdown_threshold': 0.5,
    'severe_drawdown_penalty': 40.0,      # 20.0 → 40.0 (2x increase)
    'major_drawdown_threshold': 0.3,
    'major_drawdown_penalty': 20.0,       # 10.0 → 20.0 (2x increase)
    'moderate_drawdown_threshold': 0.1,
    'moderate_drawdown_penalty': 10.0,    # 5.0 → 10.0 (2x increase)
    'linear_drawdown_multiplier': 50,     # 25 → 50 (2x increase)
    
    # === ISSUE 2: Balance ratio penalties (enhanced) ===
    'critical_equity_threshold': 0.05,
    'critical_equity_penalty': 100.0,     # 50.0 → 100.0 (2x increase)
    'severe_equity_threshold': 0.10,
    'severe_equity_penalty': 60.0,        # 30.0 → 60.0 (2x increase)
    'major_equity_threshold': 0.20,
    'major_equity_penalty': 40.0,         # 20.0 → 40.0 (2x increase)
    'moderate_equity_threshold': 0.30,
    'moderate_equity_penalty': 20.0,      # 10.0 → 20.0 (2x increase)
    'minor_equity_threshold': 0.50,
    'minor_equity_penalty': 10.0,         # 5.0 → 10.0 (2x increase)
    
    # === ISSUE 2: Enhanced consecutive loss penalties ===
    'consecutive_loss_exponent': 2.0,     # 1.5 → 2.0 (steeper escalation)
    'consecutive_loss_cap': 25.0,         # 15.0 → 25.0 (67% increase)
    
    # Trend and volatility (moderate changes)
    'trend_penalty_multiplier': 1000,
    'trend_penalty_cap': 8.0,
    'volatility_multiplier': 15,
    'volatility_penalty_cap': 5.0,
    'volatility_history_threshold': 10,
    
    # === ISSUE 4: OVERTRADING - Enhanced trading cost penalties ===
    'cost_penalty_multiplier': 1500,      # 500 → 1500 (3x increase)
    'cost_penalty_cap': 5.0,              # 2.0 → 5.0 (2.5x increase)
    
    # Special penalties (enhanced)
    'liquidation_penalty': 25.0,
    'excessive_leverage_threshold': 20,
    'excessive_leverage_multiplier': 0.5,
    
    # === ISSUE 1: CLOSING WINNERS TOO EARLY - Enhanced position holding ===
    'position_hold_bonus': 2.0,           # 0.5 → 2.0 (4x increase)
    'position_hold_penalty': 0.3,
    'optimal_hold_min': 4,
    'optimal_hold_max': 48,               # 24 → 48 (2x increase for longer holds)
    'excessive_hold_threshold': 24,
    'consecutive_wins_multiplier': 0.4,   # 0.2 → 0.4 (2x increase)
    'consecutive_wins_cap': 3.0,          # 2.0 → 3.0 (50% increase)
    'recovery_threshold': 0.05,
    'recovery_multiplier': 20,
    'recovery_bonus_cap': 3.0,
    
    # Final reward caps (slightly expanded for larger bonuses)
    'final_reward_positive_cap': 20.0,    # 15.0 → 20.0 (33% increase)
    'final_reward_negative_cap': -35.0,   # -25.0 → -35.0 (40% increase)
    'severe_loss_reward_cap': -75.0,      # -50.0 → -75.0 (50% increase)
    
    # === NEW PARAMETERS FOR SPECIFIC ISSUES ===
    
    # ISSUE 1: Trend following bonuses (NEW)
    'trend_following_bonus': 0.02,        # Bonus for holding while PnL grows
    'trend_following_threshold': 0.01,    # 1% unrealized PnL improvement
    'exit_profitable_trend_penalty': 0.05, # Penalty for exiting during good trends
    'profitable_trend_threshold': 0.02,   # 2% positive trend required
    
    # ISSUE 2: Quick loss cutting rewards (NEW)
    'quick_loss_cut_bonus': 0.03,         # Bonus for cutting losses quickly
    'loss_cut_threshold': -0.01,          # -1% loss threshold
    'max_loss_hold_steps': 5,             # Cut losses within 5 steps for bonus
    
    # ISSUE 3: Exit strategy differentiation (NEW)
    'cancel_close_penalty': 0.1,          # Penalty for CANCEL_CLOSE
    'deliberate_exit_bonus': 0.05,        # Bonus for CLOSE_LONG/CLOSE_SHORT
    'profit_target_achievement_bonus': 0.08, # Bonus for hitting profit targets
    'profit_target_threshold': 0.005,     # 0.5% profit target
    
    # ISSUE 4: Minimum profitability requirements (NEW)
    'minimum_profit_threshold': 0.005,    # 0.5% minimum profit for bonus
    'minimum_profit_bonus': 0.04,         # Bonus for exceeding minimum profit
    'small_position_penalty': 0.02,       # Penalty for positions < $50
    'small_position_threshold': 50.0,     # Dollar threshold for small positions
    'fee_ratio_penalty_threshold': 0.1,   # Penalty if fees > 10% of trade value
    'excessive_fee_ratio_penalty': 0.15,  # Heavy penalty for fee-eroding trades
}

# Conservative version for initial testing (50% of the aggressive changes)
CONSERVATIVE_IMPROVED_CONFIG = {
    # Copy base config
    **IMPROVED_REWARD_CONFIG,
    
    # Moderate the more aggressive changes
    'position_hold_bonus': 1.25,          # 2.0 → 1.25 (moderate increase)
    'optimal_hold_max': 36,               # 48 → 36 (moderate increase)
    'cost_penalty_multiplier': 1000,      # 1500 → 1000 (moderate increase)
    'consecutive_loss_cap': 20.0,         # 25.0 → 20.0 (moderate increase)
    
    # Moderate the new penalties
    'cancel_close_penalty': 0.05,         # 0.1 → 0.05 (gentler)
    'exit_profitable_trend_penalty': 0.025, # 0.05 → 0.025 (gentler)
    'small_position_penalty': 0.01,       # 0.02 → 0.01 (gentler)
}

print("=== IMPROVED REWARD CONFIGURATIONS READY ===")
print("\n1. IMPROVED_REWARD_CONFIG: Aggressive fixes for all 4 issues")
print("2. CONSERVATIVE_IMPROVED_CONFIG: Moderate fixes for gradual improvement")
print("\nTo use in training:")
print("env = FuturesTradingEnv(df=data, reward_config=IMPROVED_REWARD_CONFIG)")
print("env = FuturesTradingEnv(df=data, reward_config=CONSERVATIVE_IMPROVED_CONFIG)")
