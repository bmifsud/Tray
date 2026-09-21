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
    print("[INFO] Loading raw data from nas100_raw.csv...")
    df = pd.read_csv('nas100_raw.csv')

    features = ['open', 'high', 'low', 'close', 'tick_volume']
    data = df[features].values

    print("[INFO] Enforcing strict chronological train/validation split (80/20)...")
    # Train/test split (80/20)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

    print("[INFO] Fitting Min-Max scaler parameters strictly on training partition...")
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
    print("       -> Serialized normalization parameters to 'scaler_params.json'.")

    # Create sequences
    lookback = 25
    X_train, Y_train = create_dataset(train_data_scaled, lookback)
    X_test, Y_test = create_dataset(test_data_scaled, lookback)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    print("[INFO] Initializing Single-Layer LSTM (Hidden Neurons: 16, Dropout: 0.2)...\n")
    # Model, Loss, Optimizer
    model = LSTMModel(input_size=len(features), hidden_size=16, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop with Early Stopping
    epochs = 500
    patience = 100 # keep patience or update if needed? The user doesn't specify patience, but I will keep it at 100
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

        saved_msg = ""
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            saved_msg = " (Saved Best Model)"
        else:
            counter += 1

        # Print logic
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}{saved_msg}")

        if counter >= patience:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}. Best Validation Loss: {best_loss:.6f}")
            break

    if counter < patience:
        print(f"\n[INFO] Training completed. Best Validation Loss: {best_loss:.6f}")


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

    model_rmse = np.sqrt(mean_squared_error(Y_test_np, test_preds))
    naive_rmse = np.sqrt(mean_squared_error(Y_test_np, naive_preds))

    alpha_improvement = ((naive_rmse - model_rmse) / naive_rmse) * 100

    # Export to ONNX
    dummy_input = torch.randn(1, lookback, len(features))

    export_status = "SUCCESS ('nas100_lstm.onnx')"
    try:
        torch.onnx.export(model, dummy_input, "nas100_lstm.onnx",
                          export_params=True,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'])
    except Exception as e:
        export_status = f"FAILED ({str(e)})"

    print("\n==================================================")
    print("              MODEL VALIDATION METRICS")
    print("==================================================")
    print(f"* Root Mean Squared Error (LSTM Model):     {model_rmse:.5f}")
    print(f"* Root Mean Squared Error (Naive Baseline): {naive_rmse:.5f}")
    print(f"* Alpha Performance Improvement:            +{alpha_improvement:.2f}% (Beats Naive Persistence)" if alpha_improvement > 0 else f"* Alpha Performance Improvement:            {alpha_improvement:.2f}% (Underperforms Naive Baseline)")
    print(f"* ONNX Export Status:                       {export_status}\n")

if __name__ == '__main__':
    train()
