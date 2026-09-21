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
logging.getLogger('torch.onnx').setLevel(logging.ERROR)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def compute_features(df):
    close = df['close']
    open_p = df['open']
    high = df['high']
    low = df['low']
    vol = df['tick_volume']

    df['ret1'] = (close - open_p) / (open_p + 1e-8)
    df['hl_range'] = (high - low) / (close + 1e-8)
    df['upper_wick'] = (high - np.maximum(open_p, close)) / (close + 1e-8)
    df['lower_wick'] = (np.minimum(open_p, close) - low) / (close + 1e-8)
    
    vol_mean = vol.rolling(20, min_periods=5).mean()
    vol_std = vol.rolling(20, min_periods=5).std()
    df['vol_norm'] = (vol - vol_mean) / (vol_std + 1e-8)

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=5).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = (rsi - 50.0) / 50.0

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

def train_tf_dataset(df_raw, tf_name, optimize_grid=True):
    set_seed(42)
    df = compute_features(df_raw.copy())
    feature_cols = ['ret1', 'hl_range', 'upper_wick', 'lower_wick', 'vol_norm', 'rsi']

    df_clean = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)

    if len(df_clean) < 100:
        return None

    X_all = df_clean[feature_cols].values
    Y_all = df_clean['target'].values

    train_size = int(len(df_clean) * 0.8)
    X_train_raw = X_all[:train_size]
    Y_train_raw = Y_all[:train_size]
    X_test_raw = X_all[train_size:]
    Y_test_raw = Y_all[train_size:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    if optimize_grid and len(df_clean) >= 500:
        param_grid = [
            {'lookback': 15, 'hidden_size': 32, 'dropout': 0.2, 'lr': 0.002},
            {'lookback': 25, 'hidden_size': 32, 'dropout': 0.2, 'lr': 0.002},
            {'lookback': 15, 'hidden_size': 64, 'dropout': 0.3, 'lr': 0.001},
            {'lookback': 10, 'hidden_size': 16, 'dropout': 0.1, 'lr': 0.005},
        ]
    else:
        param_grid = [
            {'lookback': 15, 'hidden_size': 32, 'dropout': 0.2, 'lr': 0.002}
        ]

    best_grid_val_loss = float('inf')
    best_params = param_grid[0]
    best_model_state = None

    for idx, p in enumerate(param_grid):
        lookback = p['lookback']
        hidden_size = p['hidden_size']
        dropout = p['dropout']
        lr = p['lr']

        if len(X_train_scaled) <= lookback or len(X_test_scaled) <= lookback:
            continue

        X_train, Y_train = create_sequences(X_train_scaled, Y_train_raw, lookback)
        X_test, Y_test = create_sequences(X_test_scaled, Y_test_raw, lookback)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

        model = DirectionalLSTM(input_size=len(feature_cols), hidden_size=hidden_size, dropout=dropout)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        epochs = 20
        patience = 6
        cand_best_loss = float('inf')
        cand_counter = 0
        cand_state = None

        batch_size = 128
        dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(X_test_t)
                val_loss = criterion(val_preds, Y_test_t).item()

            if val_loss < cand_best_loss:
                cand_best_loss = val_loss
                cand_counter = 0
                cand_state = copy.deepcopy(model.state_dict())
            else:
                cand_counter += 1

            if cand_counter >= patience:
                break

        if cand_best_loss < best_grid_val_loss:
            best_grid_val_loss = cand_best_loss
            best_params = p
            best_model_state = cand_state

    lookback = best_params['lookback']
    hidden_size = best_params['hidden_size']

    os.makedirs('models', exist_ok=True)
    scaler_params = {
        'feature_names': feature_cols,
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'lookback': lookback,
        'hidden_size': hidden_size
    }
    
    clean_tf = os.path.splitext(tf_name)[0]
    with open(f'models/scaler_params_{clean_tf}.json', 'w') as f:
        json.dump(scaler_params, f, indent=2)

    if best_model_state is not None:
        torch.save(best_model_state, f'models/best_model_{clean_tf}.pth')

    final_model = DirectionalLSTM(input_size=len(feature_cols), hidden_size=hidden_size)
    if best_model_state is not None:
        final_model.load_state_dict(best_model_state)
    final_model.eval()

    dummy_input = torch.randn(1, lookback, len(feature_cols))
    try:
        torch.onnx.export(
            final_model, dummy_input, f'models/nas100_lstm_{clean_tf}.onnx',
            export_params=True,
            do_constant_folding=True,
            dynamo=False,
            input_names=['input'],
            output_names=['output']
        )
    except Exception as e:
        pass

    return {
        'scaler_params': scaler_params,
        'model_state': best_model_state,
        'best_params': best_params,
        'val_loss': best_grid_val_loss,
        'df_clean': df_clean,
        'scaler': scaler
    }
