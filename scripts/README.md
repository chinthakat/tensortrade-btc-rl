# scripts/

One-off diagnostic and analysis scripts written while debugging the trading
environment. They used to sit loose in the repository root.

These are **not** a supported CLI. They are kept because several of them are the
only worked examples of driving `FuturesTradingEnv` directly, and because the
analysis scripts record how the trade CSVs were interrogated when the problems
in [`../docs/history/`](../docs/history/) were being chased down.

Run them from the repository root so the top-level modules import:

```bash
python -m scripts.debug_atr
```

**Expect to edit them before they run.** Many read hard-coded paths under
`data/` or `episodes/` that are not in this repository. Two are known to be
stale against the current code: `simple_dust_test.py` passes an
`enable_funding_costs` argument that `FuturesTradingEnv.__init__` no longer
accepts, and `simple_verification.py` imports a class named
`TradingEnvironment` that does not exist (the class is `FuturesTradingEnv`).

| Script | What it does |
| --- | --- |
| `check_setup.py` | Checks the active conda environment name and that the required packages import |
| `example_usage.py` | Example calls into the CoinAPI downloader |
| `run_test.py` | Shells out to `simple_dust_test.py` (path is relative to the old root location) |
| `analyze_actions.py` | Action-type counts over a training trade log |
| `analyze_dangling_trades.py` | Finds OPEN records that never got a matching CLOSE |
| `analyze_outstanding_issues.py` | Sweeps a trade log for the known logging defects |
| `analyze_price_anomaly.py` | Looks for zero and out-of-range prices in one episode's trade log |
| `analyze_swing_data.py` | Summarises a generated swing-market OHLCV file |
| `analyze_test_trades.py` | Statistics over the trade data produced after the price-alignment fix |
| `analyze_trade_issues.py` | Investigates trades with large net-worth discrepancies |
| `check_prices.py` | Counts zero entry/close prices in one episode's trade log |
| `debug_atr.py` | Prints ATR and the dynamic stop levels derived from it |
| `debug_trades.py` | Builds an environment from `configs/ppo_production.json` and steps it |
| `diagnose_excessive_fees.py` | Traces where the fee totals come from |
| `final_cleanup_verification.py` | Sanity sweep after the code-cleanup pass |
| `penalty_monitor.py` | Reads `logs/penalty_errors.log` and aggregates penalties by type |
| `price_anomaly_pattern_analysis.py` | Tests the theory that trades executed on the previous timestep's price |
| `quick_backtest.py` | Short backtest used to check trade counting |
| `quick_silent_test.py` | Checks that penalties log to file and not to the console |
| `quick_test_trading.py` | Runs a short training session to regenerate trade data |
| `simple_cleanup_test.py` | Smaller version of `final_cleanup_verification.py` |
| `simple_dust_test.py` | Exercises the dust-position filter (stale signature, see above) |
| `simple_verification.py` | Import smoke check (stale class name, see above) |
| `trading_debug_session.py` | Walks the environment over the timeframes where price anomalies appeared |
| `trading_debug_session_clean.py` | Near-identical copy of the above |
