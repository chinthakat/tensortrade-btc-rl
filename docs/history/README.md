# Development log

These files are a chronological development log, not documentation.

Each one was written while a specific problem was being worked on — a fee that
came out wrong, a position that was left dangling, an index that ran off the end
of the dataframe — and it records the symptom, the diagnosis and the change that
was made at that moment. They were originally dumped in the repository root as
`SHOUTY_CAPS.md` files; they are kept here because they explain *why* several
non-obvious parts of `trading_environment.py` look the way they do.

Read them as history:

- They describe the state of the code on the day they were written. Where they
  disagree with the code, the code is right.
- Percentages, trade counts and "100% fixed" claims in them are from one-off
  analysis runs on logs that are not in this repository.
- Two of them (`COMPREHENSIVE_TRADE_LOGGING_FIXES.md`,
  `TRADE_LOGGING_FIXES_SUMMARY.md`) were originally committed as `.py` files
  whose entire body was a docstring. They have been renamed to `.md` and the
  Python wrapper stripped; the text is unchanged.

For documentation that is meant to be current, see [`../`](../) and the
[main README](../../README.md).

## Contents

### Reward, penalties and logging

| File | Subject |
| --- | --- |
| `REWARD_CONFIGURATION_SUMMARY.md` | Making the reward-function weights configurable via `reward_config` |
| `PNL_AWARE_SYSTEM_SUMMARY.md` | Reward terms that react to unrealised PnL |
| `INVALID_STATE_PENALTY_SUMMARY.md` | Penalties for impossible position states |
| `SILENT_PENALTY_COMPLETE.md` | Suppressing penalty spam from the training console |
| `SEPARATE_LOGGING_COMPLETE.md` | Routing penalty errors to `logs/penalty_errors.log` |
| `COMPREHENSIVE_TRADE_LOGGING_FIXES.md` | Timestamp, duration, trade-id and PnL-attribution fixes in the trade CSV |
| `TRADE_LOGGING_FIXES_SUMMARY.md` | Earlier, shorter pass over the same trade-logging issues |

### Position and order handling

| File | Subject |
| --- | --- |
| `EXCESSIVE_FEE_FIX_SUMMARY.md` | Root-cause analysis of runaway fee accumulation |
| `FEE_CALCULATION_DEBUG.md` | Walkthrough of a single trade with a $1,002 fee |
| `ENTRY_PRICE_FIX_SUMMARY.md` | Entry-price validation against market data |
| `DUST_FILTER_COMPLETE.md` | Rejecting microscopic ("dust") positions |
| `LIQUIDATION_ENHANCEMENT.md` | Binance-style maintenance margin and liquidation |
| `TRADE_VERIFICATION_REPORT.md` | Manual verification of suspicious trades |

### Observation space and indicators

| File | Subject |
| --- | --- |
| `OBSERVATION_SPACE_ANALYSIS.md` | The 27-indicator observation space as it was |
| `SIMPLIFIED_INDICATORS_PROPOSAL.md` | Proposal to cut the feature set down |
| `MINIMAL_INDICATORS_IMPLEMENTATION.md` | The 8-indicator set that is actually in the code today |
| `DATA_LEAKAGE_FIX.md` | Fitting the `StandardScaler` on the training split only |

### Training, models and infrastructure

| File | Subject |
| --- | --- |
| `ACTION_WRAPPER_FIX.md` | Dict-to-Box action-space conversion for PPO |
| `MODEL_SELECTION_IMPROVEMENTS.md` | Filtering the 33+ checkpoints shown when picking a model to continue from |
| `INDEX_BOUNDARY_FIX_COMPLETE.md` | Off-by-one at the end of an episode |
| `CLEANUP_SUMMARY.md` | Removal of a legacy `_close_position` path |

### Whole-system status snapshots

| File | Subject |
| --- | --- |
| `CRITICAL_ISSUES_ANALYSIS.md` | Issue list from one multi-episode run |
| `ANOMALY_REPORT_ANALYSIS.md` | Reading of a `trade_anomaly_analyzer.py` report |
| `EMERGENCY_FIXES_SUCCESS.md` | Status after a batch of urgent fixes |
| `COMPLETE_FIXES_FINAL_SUMMARY.md` | Roll-up of the above |
| `COMPLETE_SYSTEM_OVERHAUL_SUMMARY.md` | Roll-up of the roll-up |
