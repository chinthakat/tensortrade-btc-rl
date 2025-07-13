# Episode Trade Analysis Tool

This directory contains comprehensive trade analysis tools for multi-episode training logs.

## Directory Structure

```
DATA_ANALYSIS/
├── episode_trade_analyzer.py    # Main analysis script
├── reports/                     # Detailed JSON analysis reports
├── extracts/                    # Top profit/loss trade extracts
├── anomalies/                   # Detected reward-PnL anomalies
├── summaries/                   # CSV summaries by action type
└── README.md                    # This file
```

## Features

### 1. Action Type Analysis
For each episode and action type (OPEN, CLOSE, FLIP, ADJUST, etc.):
- **Count**: Number of actions
- **Percentage**: Percentage of total actions
- **Reward Statistics**: Min, max, average, and 95th percentile rewards
- **PnL Statistics**: Min, max, average, and total PnL

### 2. Top Trades Extraction
- **Top 10 Profit Trades**: Highest PnL trades saved to separate CSV files
- **Top 10 Loss Trades**: Lowest PnL trades saved to separate CSV files

### 3. Anomaly Detection
Detects and extracts trades with reward-PnL mismatches:
- **Positive reward for loss trades**: When a losing trade receives positive reward
- **Negative reward for profit trades**: When a profitable trade receives negative reward
- **Reward magnitude mismatches**: Large PnL with tiny rewards or vice versa

### 4. Comprehensive Reporting
- **Episode-specific reports**: Detailed JSON reports for each episode
- **Combined analysis**: Cross-episode comparison and statistics
- **Summary tables**: CSV files with key metrics per action type

## Usage

### Basic Analysis
```bash
cd DATA_ANALYSIS
python episode_trade_analyzer.py
```

This will:
1. Scan all episodes in the `episodes/` directory
2. Analyze trade logs for each episode
3. Generate comprehensive reports and extracts
4. Display summary statistics in the terminal

### Output Files

For each episode analyzed, the following files are generated:

#### Reports Directory
- `{episode}_analysis_{timestamp}.json`: Detailed analysis report
- `combined_analysis_{timestamp}.json`: Cross-episode comparison

#### Extracts Directory
- `{episode}_top_profits_{timestamp}.csv`: Top 10 most profitable trades
- `{episode}_top_losses_{timestamp}.csv`: Top 10 largest loss trades

#### Anomalies Directory
- `{episode}_anomalies_{timestamp}.csv`: Trades with reward-PnL mismatches

#### Summaries Directory
- `{episode}_summary_{timestamp}.csv`: Action type statistics summary

#### Root Directory
- `combined_trades_{timestamp}.csv`: All trades from all episodes combined

## Analysis Metrics

### Action Type Statistics
- **Count**: Total number of actions of this type
- **Percentage**: Percentage of total actions
- **Min/Max/Avg Reward**: Reward statistics
- **95th Percentile Reward**: High-end reward distribution
- **Min/Max/Avg/Total PnL**: Profit/loss statistics

### Episode Performance
- **Total Trades**: Number of trades in the episode
- **Total PnL**: Net profit/loss for the episode
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of total profits to total losses
- **Average PnL per Trade**: Mean profit/loss per trade

### Anomaly Types
1. **positive_reward_for_loss**: Losing trade (PnL < 0) with positive reward
2. **negative_reward_for_profit**: Profitable trade (PnL > 0) with negative reward
3. **reward_magnitude_mismatch**: Large PnL with disproportionately small reward
4. **excessive_reward_magnitude**: Small PnL with disproportionately large reward

## Example Analysis Output

```
📊 Episode Analysis: episode_01_20250713_151534
┌─────────────────┬────────────┐
│ Metric          │ Value      │
├─────────────────┼────────────┤
│ Total Trades    │ 1247       │
│ Total PnL       │ 342.56     │
│ Win Rate        │ 58.3%      │
│ Profit Factor   │ 1.42       │
└─────────────────┴────────────┘

┌─────────┬───────┬────────────┬─────────────┬─────────────┬──────────┐
│ Action  │ Count │ Percentage │ Avg Reward  │ 95th %ile   │ Avg PnL  │
├─────────┼───────┼────────────┼─────────────┼─────────────┼──────────┤
│ OPEN    │ 423   │ 33.9%      │ 0.0234      │ 0.1456      │ 2.34     │
│ CLOSE   │ 387   │ 31.0%      │ -0.0123     │ 0.0892      │ -1.23    │
│ FLIP    │ 289   │ 23.2%      │ 0.0567      │ 0.2134      │ 3.45     │
│ ADJUST  │ 148   │ 11.9%      │ 0.0089      │ 0.0456      │ 0.89     │
└─────────┴───────┴────────────┴─────────────┴─────────────┴──────────┘
```

## Integration with Training Pipeline

This analysis tool is designed to work seamlessly with the multi-episode training system. After each training session, run the analyzer to:

1. **Identify training issues**: Detect reward function problems
2. **Track learning progress**: Monitor action distribution changes
3. **Optimize strategies**: Analyze which actions are most profitable
4. **Debug anomalies**: Find and fix reward-PnL mismatches

## Requirements

- pandas
- numpy
- rich (for formatted output)
- pathlib (built-in)
- json (built-in)

## Notes

- The analyzer automatically detects episode log files in the standard directory structure
- All timestamps in filenames use format: YYYYMMDD_HHMMSS
- Anomaly detection thresholds can be adjusted in the `detect_anomalies()` method
- The tool handles missing columns gracefully and reports data quality issues
