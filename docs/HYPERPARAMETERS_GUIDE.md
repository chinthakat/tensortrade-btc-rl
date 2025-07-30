# Hyperparameter Management Guide

## Overview

The trading bot now includes comprehensive hyperparameter management with interactive configuration, automatic optimization, and environment normalization for stable training.

## Features

### 1. Interactive Hyperparameter Configuration
- **Algorithm-specific parameters**: Each RL algorithm (PPO, A2C, SAC) has tailored hyperparameters
- **Environment normalization**: Optional observation and reward normalization for stable training
- **Smart defaults**: Proven default values for quick setup
- **Configuration persistence**: Save and load hyperparameter configurations

### 2. Environment Normalization
- **VecNormalize wrapper**: Automatically normalizes observations and rewards
- **Improved stability**: Reduces training instability caused by varying scales
- **Configurable clipping**: Prevent extreme values from destabilizing training
- **Persistent statistics**: Save normalization statistics with trained models

### 3. Automated Optimization
- **Optuna integration**: State-of-the-art hyperparameter optimization
- **Multi-algorithm support**: Optimize PPO, A2C, and SAC parameters
- **Pruning**: Early termination of unpromising trials
- **Result persistence**: Save optimization results for analysis

## Quick Start

### Basic Training with Interactive Configuration
```bash
python train_model.py
```
The script will prompt you for:
1. Data file selection
2. Model architecture
3. RL algorithm
4. Training parameters
5. **Hyperparameters** (NEW)
6. **Normalization settings** (NEW)

### Training with Configuration File
```bash
python train_model.py
```
- Select "Load configuration from existing file" when prompted
- Choose from saved configuration files
- Modify parameters as needed

### Hyperparameter Optimization
```bash
python hyperparameter_optimization.py
```
- Automatically finds optimal hyperparameters
- Uses Optuna for efficient search
- Saves results for analysis

## Hyperparameters by Algorithm

### PPO (Proximal Policy Optimization)
- **learning_rate**: Learning rate for the optimizer (default: 3e-4)
- **batch_size**: Number of samples per gradient update (default: 64)
- **n_steps**: Steps per environment per update (default: 2048)
- **n_epochs**: Training epochs per update (default: 10)
- **clip_range**: PPO clipping parameter (default: 0.2)
- **gamma**: Discount factor (default: 0.99)
- **gae_lambda**: GAE lambda parameter (default: 0.95)

### A2C (Advantage Actor-Critic)
- **learning_rate**: Learning rate for the optimizer (default: 7e-4)
- **batch_size**: Number of samples per gradient update (default: 32)
- **n_steps**: Steps per environment per update (default: 5)
- **gamma**: Discount factor (default: 0.99)
- **gae_lambda**: GAE lambda parameter (default: 1.0)
- **ent_coef**: Entropy coefficient (default: 0.0)

### SAC (Soft Actor-Critic)
- **learning_rate**: Learning rate for the optimizer (default: 3e-4)
- **batch_size**: Number of samples per gradient update (default: 256)
- **buffer_size**: Replay buffer size (default: 1000000)
- **train_freq**: Training frequency (default: 1)
- **gradient_steps**: Gradient steps per training (default: 1)
- **tau**: Target network update rate (default: 0.005)
- **gamma**: Discount factor (default: 0.99)

## Environment Normalization

### Benefits
- **Stable training**: Reduces variance in gradients
- **Faster convergence**: Normalized inputs improve learning speed
- **Better generalization**: More robust to different market conditions

### Configuration
- **norm_obs**: Normalize observations (recommended: True)
- **norm_reward**: Normalize rewards (recommended: True for PPO/A2C, False for SAC)
- **clip_obs**: Observation clipping value (default: 10.0)
- **clip_reward**: Reward clipping value (default: 10.0)

## Configuration Files

### Structure
```json
{
  "data_file": "data/BTC_SYNTHETIC_MIXED_15m_2024-01-01_to_2024-12-31.csv",
  "model_architecture": "attention_cnn_lstm",
  "algorithm": "ppo",
  "training_params": {
    "total_timesteps": 1000000,
    "max_leverage": 25.0,
    "initial_equity": 10000.0,
    "window_size": 60,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
    "maintenance_margin_rate": 0.004,
    "liquidation_fee_rate": 0.005,
    "n_envs": 4
  },
  "hyperparameters": {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "n_epochs": 10,
    "clip_range": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "use_normalization": true,
    "norm_obs": true,
    "norm_reward": true,
    "clip_obs": 10.0,
    "clip_reward": 10.0
  }
}
```

### Template File
Use `config_hyperparameters_template.json` as a starting point:
- Contains optimized defaults for each algorithm
- Includes recommended settings profiles
- Detailed parameter explanations

## Best Practices

### Learning Rate
- **Conservative**: 1e-5 to 1e-4 for stable learning
- **Balanced**: 3e-4 (good starting point)
- **Aggressive**: 1e-3 for faster convergence (higher risk)

### Batch Size
- **Small batches** (32-64): Faster updates, more noise
- **Large batches** (128-256): More stable gradients, slower updates

### Normalization
- **Always recommended** for diverse market conditions
- **Observation normalization**: Essential for stable training
- **Reward normalization**: Beneficial for on-policy algorithms (PPO, A2C)

### Environment Count
- **2-4 environments**: Good balance for most systems
- **8+ environments**: Better parallelization on powerful hardware
- **Consider memory usage**: More environments = more RAM

## Troubleshooting

### Training Instability
1. **Enable normalization**: Set `use_normalization: true`
2. **Reduce learning rate**: Try 1e-4 or lower
3. **Increase batch size**: Use 128 or 256
4. **Lower clip range**: Try 0.1 for PPO

### Slow Convergence
1. **Increase learning rate**: Try 1e-3 (monitor for instability)
2. **Adjust exploration**: Increase entropy coefficient for A2C
3. **Check normalization**: Ensure proper scaling

### Memory Issues
1. **Reduce n_envs**: Use fewer parallel environments
2. **Smaller buffer**: Reduce buffer_size for SAC
3. **Smaller batch**: Reduce batch_size

## Model Loading

### With Normalization Statistics
```python
from train_model import load_model_with_normalization

model, vec_normalize = load_model_with_normalization("models/trading_bot_ppo_20250713_120000.zip")
```

The system automatically saves and loads normalization statistics alongside trained models.

## Advanced Usage

### Custom Hyperparameter Ranges
Modify `hyperparameter_optimization.py` to adjust search ranges:
```python
'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
```

### Multi-Objective Optimization
Extend the objective function to consider multiple metrics:
- Mean reward
- Sharpe ratio
- Maximum drawdown
- Trade frequency

### Ensemble Methods
Train multiple models with different hyperparameters and combine predictions for improved robustness.
