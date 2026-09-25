import pandas as pd
import numpy as np
import json
import datetime
import time
import glob
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    delta = close.diff().fillna(0)  # Fill the initial NaN before separating gains and losses.
    gain = delta.clip(lower=0).rolling(window=14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=5).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = (rsi - 50.0) / 50.0

    df['target'] = (close.shift(-1) > close).astype(float)
    return df

 def create_sequences(X, Y, lookback=15):
    n = len(X)
    if n <= lookback:
        return np.empty((0, lookback, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    shape = (n - lookback, lookback, X.shape[1])
    strides = (X.strides[0], X.strides[0], X.strides[1])
    xs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides).copy()
    ys = Y[lookback:n]
    return xs.astype(np.float32), ys.astype(np.float32)

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
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

 def train_tf_dataset(df_raw, tf_name, optimize_grid=True, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
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

    seq_cache = {}

    for idx, p in enumerate(param_grid):
        lookback = p['lookback']
        hidden_size = p['hidden_size']
        dropout = p['dropout']
        lr = p['lr']

        if len(X_train_scaled) <= lookback or len(X_test_scaled) <= lookback:
            continue

        if lookback not in seq_cache:
            X_train, Y_train = create_sequences(X_train_scaled, Y_train_raw, lookback)
            X_test, Y_test = create_sequences(X_test_scaled, Y_test_raw, lookback)

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            Y_test_t = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)
            seq_cache[lookback] = (X_train_t, Y_train_t, X_test_t, Y_test_t)
        else:
            X_train_t, Y_train_t, X_test_t, Y_test_t = seq_cache[lookback]

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

    final_model = DirectionalLSTM(input_size=len(feature_cols), hidden_size=best_params["hidden_size"], dropout=best_params["dropout"])
    if best_model_state is not None:
        final_model.load_state_dict(best_model_state)
    final_model.eval()

    if lookback in seq_cache:
        _, _, X_test_t, Y_test_t = seq_cache[lookback]
    else:
        X_test_seq, Y_test_seq = create_sequences(X_test_scaled, Y_test_raw, lookback)
        X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
        Y_test_t = torch.tensor(Y_test_seq, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        final_val_preds = final_model(X_test_t)
        final_val_acc = ((final_val_preds > 0.5).float() == Y_test_t).float().mean().item() if len(Y_test_t) > 0 else 0.0

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
        'val_acc': final_val_acc,
        'df_clean': df_clean,
        'scaler': scaler
    }

 def ensure_data_available():
    os.makedirs("NasData", exist_ok=True)
    csv_files = glob.glob("NasData/*.csv")
    valid_files = []
    for f in csv_files:
        if "ticks" in f.lower():
            continue
        try:
            df = pd.read_csv(f, nrows=5)
            if {"open", "high", "low", "close"}.issubset(df.columns) or {"datetime", "time"}.issubset(df.columns):
                valid_files.append(f)
        except:
            pass

    if valid_files:
        return valid_files

    print("[INFO] No valid data found in NasData/. Attempting auto-fetch via MT5 downloader...")
    try:
        import download_nas100_mt5 as mt5_dl
        if mt5_dl.initialize_mt5():
            print("[INFO] Attempting to download D1, H1, H4 bar data from MT5...")
            for tf_key in ["D1", "H1", "H4"]:
                mt5_dl.download_bars("NAS100", tf_key, count=20000, output_dir="NasData")
    except Exception as e:
        print(f"[WARN] MT5 auto-download unavailable or failed: {e}")

    csv_files = glob.glob("NasData/*.csv")
    valid_files = [f for f in csv_files if "ticks" not in f.lower() and os.path.getsize(f) > 100]
    if valid_files:
        return valid_files

    print("[INFO] Generating robust synthetic NAS100 OHLC dataset for fallback...")
    np.random.seed(42)
    n_bars = 5000
    dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_bars, freq="1h")
    price = 15000.0 + np.cumsum(np.random.normal(0, 10, n_bars))
    syn_df = pd.DataFrame({
        "time": dates,
        "open": price + np.random.normal(0, 5, n_bars),
        "high": price + np.random.uniform(2, 15, n_bars),
        "low": price - np.random.uniform(2, 15, n_bars),
        "close": price + np.random.normal(0, 5, n_bars),
        "tick_volume": np.random.randint(1000, 50000, n_bars)
    })
    fallback_path = "NasData/NQ_in_1_hour_synthetic.csv"
    syn_df.to_csv(fallback_path, index=False)
    print(f"[SUCCESS] Created fallback dataset at {fallback_path}")
    return [fallback_path]


if __name__ == '__main__':
    start_time = time.time()
    csv_files = ensure_data_available()
    
    print(f"\n[INFO] Starting concurrent standalone training across {len(csv_files)} datasets...")
    print("=" * 120)
    print(f"{'Dataset / Timeframe':<28} | {'Status':<8} | {'Bars':<6} | {'Best Opt Config':<22} | {'Val Loss':<10} | {'LSTM Acc':<9} | {'TimesFM Acc':<11} | {'Time (s)':<8}")
    print("-" * 120)

    results = []
    
    def train_file(raw_path):
        t0 = time.time()
        filename = os.path.basename(raw_path)
        try:
            df_raw = pd.read_csv(raw_path)
            if 'datetime' in df_raw.columns:
                df_raw = df_raw.rename(columns={'datetime': 'time', 'volume': 'tick_volume'})
            required_cols = {'open', 'high', 'low', 'close'}
            if not required_cols.issubset(df_raw.columns):
                return {'filename': filename, 'status': 'SKIP', 'bars': 0, 'params': 'Missing OHLC', 'val_loss': 0.0, 'val_acc': 0.0, 'time': 0.0}
            
            res = train_tf_dataset(df_raw, filename)
            dur = time.time() - t0
            tfm_acc = 0.0
            try:
                from google_timesfm_model import evaluate_timesfm_on_dataframe
                tfm_res = evaluate_timesfm_on_dataframe(df_raw, forecast_horizon=10, test_windows=5)
                tfm_acc = tfm_res.get('directional_accuracy', 0.0)
            except Exception:
                pass

            if res:
                p = res['best_params']
                cfg_str = f"L={p['lookback']}, H={p['hidden_size']}, DR={p['dropout']:.1f}, LR={p['lr']}"
                output_res = {'filename': filename, 'status': 'DONE', 'bars': len(res['df_clean']), 'params': cfg_str, 'val_loss': res['val_loss'], 'val_acc': res['val_acc'], 'tfm_acc': tfm_acc, 'time': dur}
            else:
                output_res = {'filename': filename, 'status': 'LOW_DATA', 'bars': len(df_raw), 'params': 'N/A', 'val_loss': 0.0, 'val_acc': 0.0, 'tfm_acc': tfm_acc, 'time': dur}

            del df_raw
            if res:
                del res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return output_res
        except Exception as e:
            return {'filename': filename, 'status': 'ERROR', 'bars': 0, 'params': str(e), 'val_loss': 0.0, 'val_acc': 0.0, 'tfm_acc': 0.0, 'time': time.time() - t0}

    with ThreadPoolExecutor(max_workers=min(len(csv_files), os.cpu_count() or 4)) as executor:
        futures = {executor.submit(train_file, f): f for f in csv_files}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status_col = r['status']
            bars_col = str(r['bars'])
            cfg_col = r['params']
            loss_col = f"{r['val_loss']:.4f}" if r['val_loss'] > 0 else "-"
            acc_col = f"{r['val_acc']*100:.1f}%" if r['val_acc'] > 0 else "-"
            tfm_col = f"{r['tfm_acc']:.1f}%" if r.get('tfm_acc', 0) > 0 else "-"
            time_col = f"{r['time']:.1f}s"
            print(f"{r['filename']:<28} | {status_col:<8} | {bars_col:<6} | {cfg_col:<22} | {loss_col:<10} | {acc_col:<9} | {tfm_col:<11} | {time_col:<8}")

    print("=" * 120)
    successful = [r for r in results if r['status'] == 'DONE']
    if successful:
        avg_loss = sum(r['val_loss'] for r in successful) / len(successful)
        avg_acc = sum(r['val_acc'] for r in successful) / len(successful)
        tfm_accs = [r['tfm_acc'] for r in successful if r.get('tfm_acc', 0) > 0]
        avg_tfm_acc = sum(tfm_accs) / len(tfm_accs) if tfm_accs else 0.0
        total_time = time.time() - start_time
        print(f"\n  CONSOLIDATED TRAINING & FORECASTING SUMMARY:")
        print(f"  Successfully Processed : {len(successful)} / {len(results)} datasets")
        print(f"  Average LSTM Val Loss  : {avg_loss:.4f}")
        print(f"  Average LSTM Accuracy  : {avg_acc*100:.1f}%")
        print(f"  Average TimesFM Acc    : {avg_tfm_acc:.1f}%")
        print(f"  Total Wall Time        : {total_time:.2f}s")
        print(f"  Artifacts Saved To     : ./models/")
    else:
        print("\n[WARN] No datasets successfully trained.")
    print("=" * 120)
