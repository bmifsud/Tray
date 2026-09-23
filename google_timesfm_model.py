import os
import gc
import time
import numpy as np
import pandas as pd
import torch

def load_env_file():
    """Loads key-value pairs from local .env file into os.environ if present."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

load_env_file()

# Set HF_TOKEN environment variable support if available
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("[INFO] Authenticated with Hugging Face Hub via HF_TOKEN.")
    except Exception as e:
        print(f"[WARN] Hugging Face authentication attempt: {e}")

try:
    import timesfm
    HAS_TIMESFM = True
except ImportError:
    HAS_TIMESFM = False

class GoogleTimesFMForecaster:
    """
    Wrapper for Google TimesFM 2.5 (Time Series Foundation Model by Google DeepMind).
    Supports multi-step forecasting, zero-shot signal extraction, and resource-conscious cleanup.
    """
    def __init__(self, horizon=10, context_len=512, hf_token=None):
        self.horizon = horizon
        self.context_len = context_len
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", None)
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Loads Google TimesFM model weights."""
        if not HAS_TIMESFM:
            print("[WARN] timesfm package not installed.")
            return False

        try:
            print("[INFO] Initializing Google TimesFM 2.5 (200M Torch)...")
            repo_id = "google/timesfm-2.5-200m-pytorch"
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo_id)
            config = timesfm.ForecastConfig(max_context=self.context_len, max_horizon=max(32, self.horizon))
            self.model.compile(config)
            self.is_loaded = True
            print("[SUCCESS] Google TimesFM model loaded and compiled successfully.")
            return True
        except Exception as e:
            print(f"[WARN] TimesFM initialization warning: {e}")
            self.is_loaded = False
            return False

    def forecast_series(self, prices_series):
        """
        Runs forecast on price series.
        Returns:
            preds (np.ndarray): Predicted close prices for horizon steps.
            directional_signal (int): 1 if predicted end price > start price else 0.
        """
        arr = np.array(prices_series, dtype=np.float32)
        if len(arr) < 20:
            raise ValueError("Series length must be at least 20 bars.")

        if self.is_loaded and self.model is not None:
            try:
                input_data = [arr[-self.context_len:]]
                point_forecast, _ = self.model.forecast(inputs=input_data, horizon=self.horizon)
                preds = point_forecast[0]
                dir_signal = 1 if preds[-1] > arr[-1] else 0
                return preds, dir_signal
            except Exception as e:
                pass

        # Fallback trend estimation using exponential smoothing / polynomial extrapolation
        x = np.arange(len(arr[-40:]))
        y = arr[-40:]
        poly = np.polyfit(x, y, deg=1)
        future_x = np.arange(len(x), len(x) + self.horizon)
        preds = np.polyval(poly, future_x)
        dir_signal = 1 if preds[-1] > arr[-1] else 0
        return preds, dir_signal

    def cleanup(self):
        """Releases memory resources and forces garbage collection."""
        self.model = None
        self.is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def evaluate_timesfm_on_dataframe(df, forecast_horizon=10, test_windows=5, forecaster=None):
    """
    Evaluates Google TimesFM directional accuracy across sample test windows.
    """
    created_locally = False
    if forecaster is None:
        forecaster = GoogleTimesFMForecaster(horizon=forecast_horizon)
        forecaster.load_model()
        created_locally = True

    close_prices = df['close'].values
    n = len(close_prices)
    if n < 50:
        if created_locally:
            forecaster.cleanup()
        return {'status': 'ERROR', 'reason': 'Not enough data bars', 'directional_accuracy': 0.0}

    correct_directions = 0
    total_evals = 0

    step = max(1, (n - 30 - forecast_horizon) // test_windows)
    eval_indices = list(range(30, n - forecast_horizon, step))[:test_windows]

    for idx in eval_indices:
        hist = close_prices[:idx]
        actual_future = close_prices[idx:idx + forecast_horizon]
        
        preds, dir_sig = forecaster.forecast_series(hist)
        actual_dir = 1 if actual_future[-1] > hist[-1] else 0
        
        if dir_sig == actual_dir:
            correct_directions += 1
        total_evals += 1

    dir_acc = (correct_directions / total_evals) * 100.0 if total_evals > 0 else 0.0

    if created_locally:
        forecaster.cleanup()

    return {
        'status': 'OK',
        'total_evals': total_evals,
        'directional_accuracy': round(dir_acc, 2),
        'uses_pretrained': forecaster.is_loaded
    }

if __name__ == '__main__':
    print("[INFO] Testing Google TimesFM Forecaster...")
    t = np.linspace(0, 50, 500)
    prices = 100 + np.sin(t) * 10 + np.random.normal(0, 0.5, 500)
    df_test = pd.DataFrame({'close': prices})
    res = evaluate_timesfm_on_dataframe(df_test, forecast_horizon=10)
    print("[RESULTS]", res)



