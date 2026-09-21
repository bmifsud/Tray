# NAS100 MicroStructure LSTM Trading Engine

An institutional-grade algorithmic trading pipeline designed for NAS100 / Nasdaq futures (`NQ`). The system fuses Fair Value Gap (FVG) market microstructure mechanics with deep learning Directional LSTM inference, implementing broker-aligned risk management and execution constraints.

---

## 1. Architectural Overview

The engine operates on a three-tier quantitative architecture:

```
+-------------------------------------------------------------------------------+
|                             DATA EXTRACTION TIER                              |
|  Timeframe CSVs (NasData/*.csv) -> Chronological Normalization (OHLCV)        |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                           MACHINE LEARNING TIER                               |
|  Stationary Feature Engineering (Return, Spreads, Normalized Vol, RSI)        |
|  Leak-Free Scaler Fitting strictly on 80% Train Split                         |
|  Directional LSTM (BCE Loss) -> Probability Inference -> ONNX Serialization   |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                        EXECUTION & CONFLUENCE TIER                            |
|  Microstructure Detection (Bullish & Bearish 3-Bar FVGs)                      |
|  Consequent Encroachment (50% Midpoint) Pending Limit Orders                  |
|  LSTM Confluence Filter (Bullish >= 0.48, Bearish <= 0.52)                    |
|  Buffered Stop Loss (0.75x FVG Height) + 2:1 Reward-to-Risk Target            |
|  Multi-Bar Lifecycle Monitoring (SL / TP / Expiration Timeout)                |
+-------------------------------------------------------------------------------+
```

---

## 2. Core Features & Signals

### Market Microstructure Engine
- **Bullish Fair Value Gap:** Identified when `low[i] > high[i-2]`. The gap zone is between `[high[i-2], low[i]]`.
- **Bearish Fair Value Gap:** Identified when `high[i] < low[i-2]`. The gap zone is between `[high[i], low[i-2]]`.
- **Consequent Encroachment (CE):** The 50% midpoint of the imbalance zone where institutional liquidity pools congregate.
- **Entry Mechanics:** Limit orders are staged at CE with an expiration window of 6 bars.

### Directional LSTM Network
- **Stationary Input Features (Zero-Centered):**
  1. `ret1`: Price return of the current candle `(close - open) / open`
  2. `hl_range`: Normalized candle volatility `(high - low) / close`
  3. `upper_wick`: Upper wick extension ratio
  4. `lower_wick`: Lower wick rejection ratio
  5. `vol_norm`: Z-score volume anomaly relative to 20-period rolling window
  6. `rsi`: 14-period Relative Strength Index normalized to `[-1, 1]`
- **Target:** Directional binary classification (probability of next bar close exceeding current close).
- **Inference & Export:** Exported directly to `nas100_lstm.onnx` and PyTorch weights `best_model.pth`.

### Risk Management & Broker Alignment
- **Position Sizing:** 1% account risk per trade on a $10,000 baseline account.
- **Stop Loss Buffer:** Placed 0.75x gap height beyond the structural boundary to eliminate premature wick stop-outs.
- **Target:** 2:1 Reward-to-Risk ratio with position monitoring across a 15-bar maximum holding horizon.

---

## 3. Global Multi-Timeframe Performance

Validated on out-of-sample partitions across all downloaded timeframe datasets:

| Timeframe | Evaluated Bars | Triggers | Trades | Win Rate | Profit Factor | Total R | Net Return | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NQ Daily** | 1,314 | 126 | 71 | **49.3%** | **1.93** | **+33.4R** | **+38.49%** | **-8.6%** |
| **NQ 3-Minute** | 1,342 | 130 | 72 | **51.4%** | **2.03** | **+36.1R** | **+42.24%** | **-4.2%** |
| **NQ 5-Minute** | 1,075 | 115 | 64 | **48.4%** | **1.83** | **+27.3R** | **+30.44%** | **-4.0%** |
| **NQ 1-Minute** | 1,295 | 148 | 83 | **41.0%** | **1.33** | **+16.0R** | **+16.32%** | **-7.4%** |
| **NQ Weekly** | 259 | 25 | 16 | **62.5%** | **3.33** | **+14.0R** | **+14.77%** | **-1.0%** |
| **NQ 45-Minute**| 1,220 | 91 | 44 | **40.9%** | **1.34** | **+8.7R** | **+8.60%** | **-4.6%** |
| **NQ 30-Minute**| 1,818 | 142 | 84 | **38.1%** | **1.16** | **+8.1R** | **+7.52%** | **-7.8%** |
| **NQ 15-Minute**| 1,332 | 112 | 70 | **37.1%** | **1.18** | **+8.0R** | **+7.54%** | **-8.8%** |
| **NQ 1-Hour** | 2,085 | 148 | 68 | **36.8%** | **1.14** | **+5.8R** | **+5.22%** | **-7.0%** |
| **NQ 3-Hour** | 1,127 | 111 | 61 | **36.1%** | **1.06** | **+2.5R** | **+1.93%** | **-9.6%** |
| **NQ 2-Hour** | 1,080 | 85 | 45 | **37.8%** | **1.08** | **+2.3R** | **+1.91%** | **-6.0%** |

---

## 4. Usage Instructions

### Run the Global Multi-Timeframe Pipeline
```bash
cmd /c python run_train_backtest.py
```

### Train and Export a Single Timeframe
```bash
cmd /c python extract_nas100_raw.py NasData/NQ_in_3_minute.csv
cmd /c python train_nas100_lstm.py
```
