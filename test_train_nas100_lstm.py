import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from train_nas100_lstm import create_sequences, compute_features, DirectionalLSTM, train_tf_dataset

class TestCreateSequences(unittest.TestCase):

    def _create_sequences_loop(self, X, Y, lookback):
        # Helper function to create sequences using a traditional loop (for comparison)
        n = len(X)
        if n <= lookback:
            return np.empty((0, lookback, X.shape[1]), dtype=np.float32), \
                   np.empty((0,), dtype=np.float32)
        
        X_sequences = []
        Y_sequences = []
        for i in range(n - lookback):
            X_sequences.append(X[i:i + lookback])
            Y_sequences.append(Y[i + lookback]) # Target is the value *after* the sequence
        
        return np.array(X_sequences, dtype=np.float32), np.array(Y_sequences, dtype=np.float32)

    def test_basic_sequences(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4], [5, 0.5], [6, 0.6]], dtype=np.float32)
        Y = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
        lookback = 3

        expected_X, expected_Y = self._create_sequences_loop(X, Y, lookback)
        actual_X, actual_Y = create_sequences(X, Y, lookback)

        np.testing.assert_array_almost_equal(actual_X, expected_X)
        np.testing.assert_array_almost_equal(actual_Y, expected_Y)

    def test_edge_case_n_le_lookback(self):
        X = np.array([[1, 0.1], [2, 0.2]], dtype=np.float32)
        Y = np.array([0, 1], dtype=np.float32)
        lookback = 3

        expected_X, expected_Y = self._create_sequences_loop(X, Y, lookback)
        actual_X, actual_Y = create_sequences(X, Y, lookback)

        self.assertEqual(actual_X.shape, expected_X.shape)
        self.assertEqual(actual_Y.shape, expected_Y.shape)
        self.assertTrue(actual_X.size == 0)
        self.assertTrue(actual_Y.size == 0)

    def test_data_types(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]], dtype=np.float64) # Test with different input dtype
        Y = np.array([0, 1, 0, 1], dtype=np.float64)
        lookback = 2

    def test_shape(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4], [5, 0.5]], dtype=np.float32)
        Y = np.array([0, 1, 0, 1, 0], dtype=np.float32)
        lookback = 2

        actual_X, actual_Y = create_sequences(X, Y, lookback)

        n_samples = len(X) - lookback
        n_features = X.shape[1]

        self.assertEqual(actual_X.shape, (n_samples, lookback, n_features))
        self.assertEqual(actual_Y.shape, (n_samples,))

    def test_single_element_sequences(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3]], dtype=np.float32)
        Y = np.array([0, 1, 0], dtype=np.float32)
        lookback = 2

        expected_X = np.array([[[1, 0.1], [2, 0.2]]], dtype=np.float32)
        expected_Y = np.array([0], dtype=np.float32)

        actual_X, actual_Y = create_sequences(X, Y, lookback)

        np.testing.assert_array_almost_equal(actual_X, expected_X)
        np.testing.assert_array_almost_equal(actual_Y, expected_Y)


        actual_X, actual_Y = create_sequences(X, Y, lookback)

class TestComputeFeatures(unittest.TestCase):

    def test_basic_features(self):
        # Sample data for testing
        data = {
            'open':    [100, 102, 101, 103, 105, 104],
            'close':   [102, 101, 103, 105, 104, 106],
            'high':    [103, 103, 104, 106, 106, 107],
            'low':     [ 99, 100, 100, 102, 103, 103],
            'tick_volume': [1000, 1200, 1100, 1300, 1050, 1150]
        }
        df = pd.DataFrame(data)

        # Expected values (calculated manually or with a reference implementation)
        # ret1 = (close - open) / open
        expected_ret1 = [
            (102-100)/100, (101-102)/102, (103-101)/101, (105-103)/103, (104-105)/105, (106-104)/104
        ]

        # hl_range = (high - low) / close
        expected_hl_range = [
            (103-99)/102, (103-100)/101, (104-100)/103, (106-102)/105, (106-103)/104, (107-103)/106
        ]

        # upper_wick = (high - max(open, close)) / close
        expected_upper_wick = [
            (103-max(100,102))/102, (103-max(102,101))/101, (104-max(101,103))/103,
            (106-max(103,105))/105, (106-max(105,104))/104, (107-max(104,106))/106
        ]

        # lower_wick = (min(open, close) - low) / close
        expected_lower_wick = [
            (min(100,102)-99)/102, (min(102,101)-100)/101, (min(101,103)-100)/103,
            (min(103,105)-102)/105, (min(105,104)-103)/104, (min(104,106)-103)/106
        ]

        # target = (close.shift(-1) > close).astype(float)
        expected_target = [
            1.0, 0.0, 1.0, 0.0, 1.0, np.nan # Last value is NaN because of shift(-1)
        ]

        # Note: vol_norm and rsi involve rolling means/stds, which are harder to calculate manually for short data.
        # We check their existence and basic properties, and maybe specific values for longer data.

        result_df = compute_features(df.copy())


        # For vol_norm and rsi, check that they are present and are not all NaNs (for sufficient data)
        self.assertIn('vol_norm', result_df.columns)
        self.assertIn('rsi', result_df.columns)
        # For short data, they might be all NaN initially due to min_periods
        # Let's check for NaN where expected for short data
        self.assertTrue(result_df['vol_norm'].iloc[:4].isnull().all()) # min_periods=5
        self.assertTrue(result_df['rsi'].iloc[:4].isnull().all()) # min_periods=5

    def test_compute_features_sufficient_data_for_rolling(self):
        # Create a larger DataFrame to properly test rolling window calculations
        data = {
            'open': np.arange(100, 160),
            'close': np.arange(101, 161) + np.random.randn(60),
            'high': np.arange(102, 162) + np.abs(np.random.randn(60)),
            'low': np.arange(99, 159) - np.abs(np.random.randn(60)),
            'tick_volume': np.random.randint(500, 1500, 60)
        }
        df = pd.DataFrame(data)
        result_df = compute_features(df.copy())

        # Check that vol_norm and rsi are computed and not all NaNs for sufficient data
        self.assertFalse(result_df['vol_norm'].isnull().all())
        self.assertFalse(result_df['rsi'].isnull().all())
        self.assertTrue(result_df['vol_norm'].iloc[4:].notnull().all()) # First 4 should be NaN (min_periods=5)
        self.assertTrue(result_df['rsi'].iloc[4:].notnull().all()) # First 4 should be NaN (min_periods=5)

    def test_compute_features_empty_df(self):
        df = pd.DataFrame(columns=['open', 'close', 'high', 'low', 'tick_volume'])
        result_df = compute_features(df.copy())

        # All new columns should be empty or NaN
        for col in ['ret1', 'hl_range', 'upper_wick', 'lower_wick', 'vol_norm', 'rsi', 'target']:
            self.assertIn(col, result_df.columns)
            self.assertTrue(result_df[col].isnull().all() or len(result_df[col]) == 0)

    def test_compute_features_division_by_zero_safety(self):
        # Test with values that would normally cause division by zero without safeguards
        data = {
            'open':    [100, 0, 101],
            'close':   [100, 0, 100],
            'high':    [100, 0, 101],
            'low':     [100, 0, 100],
            'tick_volume': [1000, 0, 1000]
        }
        df = pd.DataFrame(data)
        result_df = compute_features(df.copy())

        # Check that no NaNs are introduced due to division by zero (only where mathematically undefined)
        self.assertFalse(np.isinf(result_df['ret1']).any())
        self.assertFalse(np.isinf(result_df['hl_range']).any())
        self.assertFalse(np.isinf(result_df['upper_wick']).any())
        self.assertFalse(np.isinf(result_df['lower_wick']).any())
        self.assertFalse(np.isinf(result_df['vol_norm']).any())
        self.assertFalse(np.isinf(result_df['rsi']).any())

        # Specifically check values where denominator was 0 and 1e-8/1e-9 was added
        # For example, if open_p is 0, ret1 should be (close - 0) / 1e-8
        # For the second row, close=0, open_p=0, high=0, low=0, vol=0
        # ret1 should be (0-0)/(0+1e-8) = 0
        np.testing.assert_almost_equal(result_df.loc[1, 'ret1'], 0.0)
        # hl_range should be (0-0)/(0+1e-8) = 0
        np.testing.assert_almost_equal(result_df.loc[1, 'hl_range'], 0.0)



class TestDirectionalLSTM(unittest.TestCase):
    def test_forward_pass(self):
        model = DirectionalLSTM(input_size=5, hidden_size=16, dropout=0.1)
        x = torch.randn(4, 10, 5)
        out = model(x)
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

class TestTrainTFDataset(unittest.TestCase):
    def test_insufficient_data(self):
        df_short = pd.DataFrame({
            'open': np.ones(50),
            'high': np.ones(50),
            'low': np.ones(50),
            'close': np.ones(50),
            'tick_volume': np.ones(50)
        })
        res = train_tf_dataset(df_short, "test_short.csv")
        self.assertIsNone(res)

if __name__ == '__main__':
    unittest.main()