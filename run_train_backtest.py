import os
import subprocess
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def execute_pipeline():
    print("=== STEP 1: Executing Leak-Free LSTM Training & ONNX Export ===")
    subprocess.run(["python3", "train_nas100_lstm.py"], check=True)

    print("\n=== STEP 2: Running Evaluation & Backtest Analysis ===")

    # 1. Load scaler parameters and raw data
    with open('scaler_params.json', 'r') as f:
        scaler_params = json.load(f)

    min_close = scaler_params['min_'][3] # Fixed key to match our output
    scale_close = scaler_params['scale_'][3] # Fixed key

    df = pd.read_csv('nas100_raw.csv')
    features = ['open', 'high', 'low', 'close', 'tick_volume']
    raw_values = df[features].values.astype(np.float32)
    n_samples = len(raw_values)
    split_idx = int(n_samples * 0.8)

    # Re-apply train normalization to match validation loader logic
    train_min = np.array(scaler_params['min_'], dtype=np.float32)
    scale_range = 1.0 / np.array(scaler_params['scale_'], dtype=np.float32) # scale_ is standard deviation or scale factor in minmax, let's just use sklearn's logic directly if possible, or inverse it. Actually, minmax formula is (x - data_min_) * scale_ where min_ = -data_min * scale_. So x_scaled = x * scale_ + min_. Let's just use that.

    scaled_values = raw_values * np.array(scaler_params['scale_'], dtype=np.float32) + train_min

    # Rebuild validation set windows
    seq_length = 25
    X_val, y_val = [], []
    for i in range(n_samples - seq_length):
        target_idx = i + seq_length
        if target_idx >= split_idx:
            window = scaled_values[i:target_idx]
            target = scaled_values[target_idx, 3]
            X_val.append(window)
            y_val.append([target])

    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=8,
        shuffle=False
    )

    # 2. Load trained model structure for inference
    # Note: our train_nas100_lstm.py defines LSTMModel, we will redefine it here to match
    class LSTMModel(nn.Module):
        def __init__(self, input_size=5, hidden_size=16, num_layers=1, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out[:, -1, :])
            out = self.fc(out)
            return out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()

    # 3. Generate predictions
    val_preds = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            val_preds.extend(preds.cpu().numpy().flatten())

    # Denormalize: orig_x = (scaled_x - min_) / scale_
    denormalized_preds = (np.array(val_preds) - min_close) / scale_close

    # 4. Execute Vectorized Backtest Logics
    trade_df = df.iloc[-len(denormalized_preds):].copy().reset_index(drop=True)
    trade_df['predicted_target'] = denormalized_preds

    starting_balance = 10000.0
    risk_per_trade = 0.01
    trade_returns = []

    for i in range(3, len(trade_df)):
        current_close = trade_df.loc[i, 'close']
        current_low = trade_df.loc[i, 'low']

        fvg_top = trade_df.loc[i-2, 'low']
        fvg_bottom = trade_df.loc[i, 'high']

        if fvg_top > fvg_bottom:
            consequent_encroachment = fvg_bottom + ((fvg_top - fvg_bottom) / 2.0)
            target = trade_df.loc[i, 'predicted_target']

            if current_low <= consequent_encroachment and target > current_close:
                entry_price = consequent_encroachment
                stop_loss = fvg_bottom

                if entry_price > stop_loss:
                    risk = entry_price - stop_loss
                    reward = target - entry_price
                    r_multiple = reward / risk

                    next_low = trade_df.loc[i+1, 'low'] if (i+1) < len(trade_df) else trade_df.loc[i, 'low']
                    next_close = trade_df.loc[i+1, 'close'] if (i+1) < len(trade_df) else trade_df.loc[i, 'close']

                    if next_low <= stop_loss:
                        trade_returns.append(-risk_per_trade)
                    else:
                        actual_reward = next_close - entry_price
                        actual_r = actual_reward / risk
                        trade_returns.append(risk_per_trade * actual_r)

    trade_returns = np.array(trade_returns)
    if len(trade_returns) > 0:
        win_rate = len(trade_returns[trade_returns > 0]) / len(trade_returns)
        equity_curve = starting_balance * np.cumprod(1 + trade_returns)
        max_dd = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve).max()

        print(f"\n--- Backtest Results Summary ---")
        print(f"Total Trades Executed: {len(trade_returns)}")
        print(f"Strategy Win Rate:     {win_rate * 100:.2f}%")
        print(f"Net Return:            {((equity_curve[-1] / starting_balance) - 1) * 100:.2f}%")
        print(f"Max Drawdown:          {max_dd * 100:.2f}%")
    else:
        print("\n--- Backtest Results Summary ---")
        print("No trade setups met the FVG Consequent Encroachment confluence criteria in this validation slice.")

if __name__ == "__main__":
    execute_pipeline()
