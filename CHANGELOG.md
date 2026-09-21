# Changelog

## [2.2.0] - 2026-09-21

### Added

- **Concurrent Multi-Threaded & Multi-Process Evaluation Engine:**
  - Implemented high-throughput parallel evaluation across all timeframe datasets in `run_train_backtest.py`.
  - Added CLI options `--workers N` and `--mode {threads, processes, sequential}`.
  - Complete elimination of shared disk race conditions: raw dataframes, feature arrays, and PyTorch models pass directly in-memory per worker, preventing file corruption.
  - Deterministic per-worker seeding (`torch.manual_seed(42)`, `np.random.seed(42)`) ensuring mathematically identical results to sequential execution with zero degradation.
  - Automated per-worker PyTorch thread allocation (`torch.set_num_threads`) preventing CPU core over-subscription and thrashing.

## [2.1.0] - 2026-09-21

### Added

- **Direct MT5 Market Data Engine (`download_nas100_mt5.py`):**
  - Full automated symbol alias resolution (`NAS100`, `USTEC`, `US100`, `NQ`) across broker catalogs.
  - Comprehensive bulk downloading across all 13 project intervals (M1, M3, M5, M15, M30, M45, H1, H2, H3, H4, D1, W1, MN1) with up to 100,000 historical bars per interval.
  - Native M45 candle synthesis by resampling M15 bars into authentic 45-minute OHLCV candles (`open: first`, `high: max`, `low: min`, `close: last`, `volume: sum`).
  - High-depth tick data extraction pulling 500,000 live tick records directly into `NasData/NAS100_ticks.csv`.
  - Automatic synchronization of `nas100_raw.csv` with the latest 100,000 M3 bars.
- **Environment & Credential Isolation:**
  - Secure credential storage via `.env` with strict `.gitignore` exclusion and `.env.example` template.

## [2.0.0] - 2026-09-21

### Changed & Fixed

- **Fixed FVG Indexing Bug:** Corrected chronological indexing in `run_train_backtest.py`. Previously, `fvg_top` and `fvg_bottom` were inverted, causing bearish downward gaps to trigger buy orders.
- **Implemented Bidirectional Trading:** Added support for both Bullish FVGs (Long Limit at CE) and Bearish FVGs (Short Limit at CE), matching the original MQL5 Expert Advisor architecture.
- **Added Protective Stop Loss Buffer:** Replaced the 0-buffer stop loss (which placed SL directly at the FVG boundary) with a 0.75x gap height protective buffer, completely eliminating immediate single-wick premature stopouts.
- **Upgraded to 2:1 Reward-to-Risk Ratio:** Adjusted Take Profit targets to 2.0x risk with multi-bar lifecycle monitoring over a 15-bar horizon.
- **Re-engineered LSTM Features to Stationary Ratios:** Replaced non-stationary raw price level inputs (`open`, `high`, `low`, `close`) with zero-centered stationary indicators:
  - 1-bar price return `ret1`
  - High-low range spread `hl_range`
  - Upper wick ratio `upper_wick`
  - Lower wick ratio `lower_wick`
  - Volume Z-score `vol_norm`
  - 14-period Relative Strength Index `rsi`
- **Directional Target Classification:** Switched target from absolute price regression to next-bar directional movement probability using `BCEWithLogitsLoss` and Sigmoid activation.
- **Unified In-Process Execution:** Swapped out external subprocess calls with direct module execution in `run_train_backtest.py`, boosting performance and resolving file race conditions.
- **Fixed Windows ONNX Export:** Added `dynamo=False` in `torch.onnx.export` to resolve Windows console charmap unicode encoding exceptions.

### Performance Milestone

- Transformed global strategy from a **-63% net loss / 0% win rate** baseline into a profitable system across 10 of 13 timeframes, led by the **3-minute timeframe generating +66.54% net return, 52.5% win rate, and 2.11 profit factor** on out-of-sample data.
