# Enhanced Action Space: HOLD and CANCEL Actions

## Overview
The trading environment now supports explicit HOLD, BUY, SELL, and CANCEL actions to encourage better trading behavior, patience, and risk management.

## Problem Solved
Previously, the model was constantly trading (BUY/SELL only) without learning to wait for good opportunities or properly exit positions. This led to:
- Overtrading and excessive transaction costs
- Poor timing on entries and exits
- No concept of "patience" or "waiting for opportunities"

## New Action Space

### Advanced Action Space (Recommended)
When `use_advanced_action_space: true` in config:
```json
{
  "action_type": 0-3,      // 0=HOLD, 1=BUY, 2=SELL, 3=CANCEL
  "leverage": 0.1-5.0,     // Leverage amount when trading
  "risk_percentage": 0.01-1.0  // Risk percentage of equity
}
```

### Legacy Action Space (Enhanced)
When `use_advanced_action_space: false`:
- Continuous leverage from -5.0 to 5.0
- **NEW**: Trading threshold of 0.1
  - Values between -0.1 and 0.1 → HOLD action
  - Values > 0.1 → BUY action  
  - Values < -0.1 → SELL action

## Action Types

### 1. HOLD (Action Type 0)
- **Purpose**: Wait for better opportunities
- **Reward**: Base reward of 0.001 for patience
- **Bonuses**:
  - 2x reward during high volatility (good time to wait)
  - 1.5x reward during unclear trends
  - Penalty for missing clear profitable opportunities

### 2. BUY (Action Type 1)
- **Purpose**: Enter long position
- **Behavior**: Uses positive leverage value
- **Risk**: Controlled by risk_percentage parameter

### 3. SELL (Action Type 2)
- **Purpose**: Enter short position  
- **Behavior**: Uses negative leverage value
- **Risk**: Controlled by risk_percentage parameter

### 4. CANCEL (Action Type 3)
- **Purpose**: Close current position (profit-taking or stop-loss)
- **Rewards**:
  - +0.005 reward for taking profits
  - +0.002 reward for cutting losses (stop-loss)

## Reward Enhancements

### Patience Rewards
- **Hold Streak**: Optimal patience (3-10 holds) → +0.0005 per hold
- **Overpatience**: Too many holds (>20) → -0.0005 penalty per excess hold

### Trading Penalties
- **Overtrading**: Trading streak >5 → increasing penalties
- **Excessive Activity**: Encourages thoughtful decision-making

### Market-Aware Holding
- Higher rewards for holding during volatile periods
- Reduced rewards for holding during clear trends

## Configuration

### Example Configuration
```json
{
  "environment_config": {
    "use_advanced_action_space": true,
    "max_leverage": 5.0,
    "initial_equity": 10000.0
  },
  "advanced_features": {
    "encourage_patience": true,
    "penalize_overtrading": true,
    "reward_good_exits": true,
    "hold_reward_multiplier": 1.5,
    "cancel_reward_multiplier": 2.0
  }
}
```

## Expected Behavior Changes

### Before Enhancement
- Constant BUY/SELL actions every step
- No consideration for market conditions
- Poor risk management
- High transaction costs

### After Enhancement
- Strategic holding during uncertain periods
- Active trading only during clear opportunities
- Better position exits (profit-taking and loss-cutting)
- More sustainable trading patterns

## Usage

### Training with Enhanced Actions
1. Use config: `configs/config_hold_cancel_actions.json`
2. Start training: Models will learn to use all 4 action types
3. Monitor logs: Check for HOLD/CANCEL action usage

### Testing Action Space

The `test_enhanced_actions.py` script referenced here was never written — the
file was committed empty and has since been removed. To check both action spaces
by hand, construct `FuturesTradingEnv` with `use_advanced_action_space=True` and
then `False`, and inspect `env.action_space` and the wrapper produced by
`wrap_environment_for_algorithm()`.

## Benefits
1. **Reduced Overtrading**: Explicit hold action reduces unnecessary trades
2. **Better Risk Management**: Cancel action enables strategic exits
3. **Market Awareness**: Rewards adapt to volatility and trends
4. **Sustainable Performance**: Encourages long-term thinking over frequent trading

The enhanced action space should lead to more realistic and profitable trading behavior that better mimics successful human traders.
