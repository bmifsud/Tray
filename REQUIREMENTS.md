# Project Requirements & Specifications

## System Environment
- **Platform:** Windows x86_64
- **Runtime:** Python 3.13.x
- **Shell Execution:** Strictly `cmd /c` prefix for terminating subprocesses without interactive hang

## Python Dependencies
- `torch >= 2.0.0`: Deep learning LSTM modeling and ONNX serialization
- `pandas >= 2.0.0`: OHLCV time series manipulation and feature engineering
- `numpy >= 1.24.0`: Mathematical vectors and matrix transformations
- `scikit-learn >= 1.2.0`: Feature scaling (`StandardScaler`) and metric computation
- `onnx >= 1.14.0`: Cross-platform model representation for MT5 integration
- `MetaTrader5 >= 5.0.0`: Direct broker terminal API for bar and tick extraction
- `gdown >= 5.0.0`: Automated dataset acquisition from Google Drive

## Input Data Specifications
- **Format:** CSV with headers: `time`, `open`, `high`, `low`, `close`, `tick_volume`
- **Timezone:** UTC
- **Sampling Intervals:** 1m, 3m, 5m, 15m, 30m, 45m, 1h, 2h, 3h, 4h, Daily, Weekly, Monthly

## Machine Learning Constraints
- **Chronological Split:** Strict 80% train / 20% validation split without future lookahead bias
- **Normalization:** `StandardScaler` parameters fit strictly on the training partition and serialized to `scaler_params.json`
- **Target Variable:** Directional movement binary classification `(close[t+1] > close[t])`
- **Inference Format:** Sigmoid probability output in range `[0.0, 1.0]`

## Trade Execution Constraints
- **Risk per Trade:** 1.0% of account equity
- **Reward-to-Risk Ratio:** 2.0:1
- **Stop Loss Buffer:** 0.75x FVG height beyond structural edge
- **Order Type:** Pending Limit order at Consequent Encroachment (50% midpoint)
- **Pending Order Expiration:** 6 bars
- **Maximum Trade Holding Horizon:** 15 bars
