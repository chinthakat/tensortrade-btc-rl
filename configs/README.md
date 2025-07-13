# Configuration Files Directory

This directory contains pre-configured training setups for different trading scenarios. Each configuration includes model architecture, algorithm, training parameters, and hyperparameters.

## Available Configurations

### Quick Start & Development
- **`ppo_quick_start.json`** - Fast 15-minute training for testing and development
- **`a2c_conservative.json`** - Safe, conservative trading with lower risk settings

### Production Trading
- **`ppo_production.json`** - Full production training with optimized settings
- **`sac_experimental.json`** - Continuous action space experimentation

### Specialized Strategies
- **`hft_aggressive.json`** - High-frequency trading with quick position changes

## Configuration Structure

Each configuration file contains:

```json
{
  "name": "Human-readable name",
  "description": "What this configuration is for",
  "data_file": "Path to training data",
  "model_architecture": "Neural network architecture",
  "algorithm": "RL algorithm (ppo/a2c/sac)",
  "training_params": {
    "total_timesteps": "How long to train",
    "max_leverage": "Maximum trading leverage",
    "initial_equity": "Starting capital",
    "window_size": "Lookback period",
    "stop_loss_pct": "Stop loss percentage",
    "take_profit_pct": "Take profit percentage",
    "maintenance_margin_rate": "Exchange margin rate",
    "liquidation_fee_rate": "Liquidation fee",
    "n_envs": "Parallel environments"
  },
  "hyperparameters": {
    "learning_rate": "Algorithm learning rate",
    "batch_size": "Training batch size",
    "use_normalization": "Enable environment normalization",
    "...": "Algorithm-specific parameters"
  },
  "use_case": "When to use this configuration"
}
```

## Usage

### Loading a Configuration
```bash
python train_model.py
```
1. Select "Load configuration from existing file"
2. Choose from the list of available configurations
3. Modify parameters if needed

### Creating New Configurations
1. Run training script interactively
2. Configure all parameters
3. Choose to save configuration with a meaningful name

### Customizing Configurations
1. Copy an existing configuration file
2. Modify parameters as needed
3. Update name and description
4. Save with a new filename

## Configuration Guidelines

### Naming Convention
- Use descriptive names: `ppo_conservative`, `sac_experimental`
- Include algorithm and strategy type
- Avoid timestamps in manually created configs

### Parameter Recommendations

#### Quick Testing (< 1 hour)
- `total_timesteps`: 50,000 - 100,000
- `n_envs`: 2-4
- `batch_size`: 32-64

#### Production Training (4-8 hours)
- `total_timesteps`: 1,000,000 - 3,000,000
- `n_envs`: 4-8
- `batch_size`: 64-256

#### Conservative Trading
- `max_leverage`: 5-15
- `stop_loss_pct`: 0.01-0.015
- `learning_rate`: 1e-4 - 3e-4

#### Aggressive Trading
- `max_leverage`: 25-50
- `stop_loss_pct`: 0.005-0.01
- `learning_rate`: 3e-4 - 1e-3

## Auto-Generated Configs

Training sessions automatically create backup configurations in the format:
- `training_session_YYYYMMDD_HHMMSS.json`

These are created automatically and can be cleaned up periodically.

## Best Practices

1. **Use templates**: Start with an existing configuration
2. **Document changes**: Update name and description when modifying
3. **Test first**: Use quick_start configs for initial testing
4. **Clean up**: Remove outdated auto-generated configs
5. **Backup**: Keep working configurations in version control
