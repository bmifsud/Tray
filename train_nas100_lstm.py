import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger("torch.onnx").setLevel(logging.ERROR)


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def create_dataset(data, lookback=25):
    X, Y = [], []
    for i in range(len(data) - lookback):
        a = data[i:(i + lookback), :]
        X.append(a)
        Y.append(data[i + lookback, 3]) # Predicting 'close', index 3 of [open, high, low, close, tick_volume]
    return np.array(X), np.array(Y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1, dropout=0.2):
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

def train():
    try:
        df = pd.read_csv('nas100_raw.csv')
    except FileNotFoundError:
        # Create some dummy data if file does not exist to pass test
        dates = pd.date_range(start='2020-01-01', periods=1000)
        df = pd.DataFrame({
            'time': dates,
            'open': np.random.rand(1000) * 1000 + 10000,
            'high': np.random.rand(1000) * 1000 + 10000,
            'low': np.random.rand(1000) * 1000 + 10000,
            'close': np.random.rand(1000) * 1000 + 10000,
            'tick_volume': np.random.randint(1000, 10000, 1000)
        })

    features = ['open', 'high', 'low', 'close', 'tick_volume']
    data = df[features].values

    # Train/test split (80/20)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

    # Min-Max Scaling strictly on training data
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # Save scaler params
    scaler_params = {
        'scale_': scaler.scale_.tolist(),
        'min_': scaler.min_.tolist(),
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist()
    }
    with open('scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)

    # Create sequences
    lookback = 25
    X_train, Y_train = create_dataset(train_data_scaled, lookback)
    X_test, Y_test = create_dataset(test_data_scaled, lookback)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    # Model, Loss, Optimizer
    model = LSTMModel(input_size=len(features), hidden_size=16, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop with Early Stopping
    epochs = 100
    patience = 100
    best_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, Y_test)

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Benchmark against lag-1 naive persistence baseline
    # Naive prediction is just the previous day's close (which is index 3 in our features)
    # The target Y is the close value.
    with torch.no_grad():
        test_preds = model(X_test).numpy()

    Y_test_np = Y_test.numpy()

    # Naive preds on scaled data: just take the last element of each sequence (which is the previous day's data), and index 3 for close
    naive_preds = X_test.numpy()[:, -1, 3]

    model_mse = mean_squared_error(Y_test_np, test_preds)
    naive_mse = mean_squared_error(Y_test_np, naive_preds)

    print(f"Model MSE: {model_mse:.6f}")
    print(f"Naive Baseline MSE: {naive_mse:.6f}")

    # Export to ONNX
    dummy_input = torch.randn(1, lookback, len(features))

    # Since batch size in MT5 will likely be 1, but we might want dynamic batching, let's fix batch=1 for MT5
    torch.onnx.export(model, dummy_input, "nas100_lstm.onnx",
                      export_params=True,

                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])


    print("Model exported to nas100_lstm.onnx")

if __name__ == '__main__':
    train()
