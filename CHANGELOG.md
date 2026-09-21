# Changelog

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
