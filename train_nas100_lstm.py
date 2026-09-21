import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss
import warnings
import logging
import copy
import os

warnings.filterwarnings('ignore')
logging.getLogger("torch.onnx").setLevel(logging.ERROR)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def compute_features(df):
    """
    Computes stationary, zero-centered features from OHLCV data.
    Ensures model is scale-independent across all timeframes.
    """
    close = df['close']
    open_p = df['open']
    high = df['high']
    low = df['low']
    vol = df['tick_volume']

    # 1. Price Return of current candle
    df['ret1'] = (close - open_p) / (open_p + 1e-8)
    
    # 2. Normalized High-Low Range
    df['hl_range'] = (high - low) / (close + 1e-8)
    
    # 3. Upper Wick Ratio
    df['upper_wick'] = (high - np.maximum(open_p, close)) / (close + 1e-8)
    
    # 4. Lower Wick Ratio
    df['lower_wick'] = (np.minimum(open_p, close) - low) / (close + 1e-8)
    
    # 5. Normalized Volume
    vol_mean = vol.rolling(20, min_periods=5).mean()
    vol_std = vol.rolling(20, min_periods=5).std()
    df['vol_norm'] = (vol - vol_mean) / (vol_std + 1e-8)
    
    # 6. Normalized RSI (14-period)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=5).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = (rsi - 50.0) / 50.0  # normalized to [-1, 1] range

    # Target: 1 if next bar closes higher than current close, else 0
    df['target'] = (close.shift(-1) > close).astype(float)

    return df

def create_sequences(X, Y, lookback=15):
    xs, ys = [], []
    for i in range(len(X) - lookback):
        xs.append(X[i:(i + lookback)])
        ys.append(Y[i + lookback])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

class DirectionalLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=1, dropout=0.2):
        super(DirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train():
    print("[INFO] Loading raw data from nas100_raw.csv...")
    raw_df = pd.read_csv('nas100_raw.csv')

    df = compute_features(raw_df.copy())
    feature_cols = ['ret1', 'hl_range', 'upper_wick', 'lower_wick', 'vol_norm', 'rsi']
    
    df_clean = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    
    if len(df_clean) < 60:
        print("[WARN] Insufficient data to train LSTM. Skipping training.")
        return False

    X_all = df_clean[feature_cols].values
    Y_all = df_clean['target'].values

    print("[INFO] Enforcing strict chronological train/validation split (80/20)...")
    train_size = int(len(df_clean) * 0.8)
    X_train_raw = X_all[:train_size]
    Y_train_raw = Y_all[:train_size]
    X_test_raw = X_all[train_size:]
    Y_test_raw = Y_all[train_size:]

    print("[INFO] Fitting Standard Scaler parameters strictly on training partition...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    scaler_params = {
        'feature_names': feature_cols,
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'lookback': 15
    }
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print("       -> Serialized feature normalization parameters to 'scaler_params.json'.")

    lookback = 15
    X_train, Y_train = create_sequences(X_train_scaled, Y_train_raw, lookback)
    X_test, Y_test = create_sequences(X_test_scaled, Y_test_raw, lookback)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

    print(f"[INFO] Initializing Directional LSTM (Inputs: {len(feature_cols)}, Hidden: 32, Dropout: 0.2)...")
    model = DirectionalLSTM(input_size=len(feature_cols), hidden_size=32, dropout=0.2)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    epochs = 40
    patience = 12
    best_loss = float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, Y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_t)
            val_loss = criterion(val_preds, Y_test_t)

        val_loss_val = val_loss.item()
        saved_msg = ""
        if val_loss_val < best_loss:
            best_loss = val_loss_val
            counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            saved_msg = " (Saved Best Model)"
        else:
            counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss_val:.4f}{saved_msg}")

        if counter >= patience:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_loss:.4f}")
            break

    # Restore best weights and save to disk
    model.load_state_dict(best_state_dict)
    torch.save(best_state_dict, 'best_model.pth')
    model.eval()

    with torch.no_grad():
        test_probs = model(X_test_t).numpy().flatten()

    binary_preds = (test_probs >= 0.50).astype(float)
    acc = accuracy_score(Y_test, binary_preds)
    majority_baseline = max(np.mean(Y_test), 1.0 - np.mean(Y_test))
    brier_model = brier_score_loss(Y_test, test_probs)
    brier_naive = brier_score_loss(Y_test, np.full_like(Y_test, 0.5))

    # Export to ONNX
    dummy_input = torch.randn(1, lookback, len(feature_cols))
    export_status = "SUCCESS ('nas100_lstm.onnx')"
    try:
        torch.onnx.export(
            model, dummy_input, "nas100_lstm.onnx",
            export_params=True,
            do_constant_folding=True,
            dynamo=False,
            input_names=['input'],
            output_names=['output']
        )
    except Exception as e:
        export_status = f"FAILED ({str(e)})"

    print("\n==================================================")
    print("              MODEL VALIDATION METRICS")
    print("==================================================")
    print(f"* Out-of-Sample Accuracy:        {acc * 100:.2f}%")
    print(f"* Majority Class Baseline:        {majority_baseline * 100:.2f}%")
    print(f"* Model Brier Score:              {brier_model:.5f} (vs Naive 0.5: {brier_naive:.5f})")
    print(f"* ONNX Export Status:             {export_status}\n")
    return True

if __name__ == '__main__':
    train()
