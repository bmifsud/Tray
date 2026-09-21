# Universal Directory of Open Knowledge Databases: NAS100 LSTM & FVG Strategy Summary

## 1. Executive Summary & Overview
This knowledge document consolidates the end-to-end quantitative trading system for **NAS100 (NASDAQ 100 E-mini Futures)** using PyTorch **Directional LSTM Neural Networks** integrated with **Fair Value Gap (FVG)** market microstructure patterns and **50-EMA Trend Filtering**.

---

## 2. Feature Engineering & Signal Inputs
The model transforms OHLCV bar data into 6 stationary, normalized features for time-series sequence prediction:

1. **Normalized Returns (`ret1`)**: `(Close - Open) / Open`
2. **High-Low Range (`hl_range`)**: `(High - Low) / Close`
3. **Upper Wick Ratio (`upper_wick`)**: `(High - max(Open, Close)) / Close`
4. **Lower Wick Ratio (`lower_wick`)**: `(min(Open, Close) - Low) / Close`
5. **Volume Z-Score (`vol_norm`)**: `(Volume - RollingMean(20)) / RollingStd(20)`
6. **Normalized RSI (`rsi`)**: `(RSI(14) - 50.0) / 50.0`

### Target Variable
- Binary classification: `Target = 1.0` if `Close(t+1) > Close(t)`, else `0.0`.

---

## 3. PyTorch Model Architecture (`DirectionalLSTM`)
- **Input Dimension**: `(Batch Size, Lookback Window, 6)`
- **LSTM Layer**: Hidden dimension `hidden_size` (grid searched between 16 and 64), 1 layer, Dropout = 0.2.
- **Fully Connected Head**: `Linear(hidden_size -> 16)` -> `ReLU` -> `Linear(16 -> 1)` -> `Sigmoid`.
- **Loss Function**: Binary Cross Entropy Loss (`nn.BCELoss`).
- **Optimizer**: Adam with learning rate tuning (`0.001` - `0.005`).

---

## 4. Strategy Rules & Execution Logic
1. **Trend Filter**: 50-period Exponential Moving Average (50-EMA).
   - Longs: `Close > 50-EMA`
   - Shorts: `Close < 50-EMA`
2. **Microstructure Confirmation**: Fair Value Gap (FVG) detection across consecutive 3-bar formations.
3. **Probability Threshold**: LSTM direction prediction probability >= 0.55 for Buy / <= 0.45 for Sell.
4. **Risk Management**:
   - Risk/Reward Ratio: `2.0`
   - Max holding period: 10 bars
   - Entry window: 3 bars post-signal.

---

## 5. Artifacts & Deployment Support
- **`best_model.pth`**: Trained PyTorch weights.
- **`nas100_lstm.onnx`**: Exported ONNX model for MetaTrader 5 (MQL5) ONNX API runtime inference.
- **`scaler_params.json`**: Feature means and standard deviations for real-time standardization in MQL5.
- **`nas100_lstm_notebook.ipynb`**: Interactive Jupyter Notebook for iterative model tuning and visualization.
