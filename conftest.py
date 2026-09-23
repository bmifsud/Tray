import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def synthetic_ohlcv():
    """Generates 50 rows of synthetic OHLCV dataframe in-memory."""
    n = 50
    dates = pd.date_range('2025-01-01', periods=n, freq='15min')
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        'time': dates,
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices + 0.1,
        'tick_volume': np.random.randint(100, 1000, n)
    })

@pytest.fixture
def synthetic_sequences():
    """Generates synthetic (X, Y) sequence tensors for LSTM testing."""
    np.random.seed(42)
    X = np.random.randn(30, 10, 5).astype(np.float32)
    Y = np.random.randint(0, 2, size=(30,)).astype(np.float32)
    return X, Y
