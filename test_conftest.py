import unittest
import pytest
import pandas as pd
import numpy as np

def test_synthetic_ohlcv_fixture(synthetic_ohlcv):
    assert isinstance(synthetic_ohlcv, pd.DataFrame)
    assert len(synthetic_ohlcv) == 50
    assert {'time', 'open', 'high', 'low', 'close', 'tick_volume'}.issubset(synthetic_ohlcv.columns)

def test_synthetic_sequences_fixture(synthetic_sequences):
    X, Y = synthetic_sequences
    assert X.shape == (30, 10, 5)
    assert Y.shape == (30,)
    assert X.dtype == np.float32
    assert Y.dtype == np.float32
