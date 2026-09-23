import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from run_train_backtest import (
    run_fvg_backtest,
    process_single_timeframe,
    _print_header,
    _print_summary,
    main
)

class TestRunTrainBacktest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_run_fvg_backtest_short_df(self):
        df_short = pd.DataFrame({'close': np.ones(10), 'high': np.ones(10), 'low': np.ones(10)})
        res = run_fvg_backtest(df_short, lstm_probs=np.ones(10))
        self.assertEqual(res['triggers'], 0)
        self.assertEqual(res['win_rate'], 0.0)

    def test_run_fvg_backtest_valid_lstm(self):
        n = 50
        dates = pd.date_range('2025-01-01', periods=n, freq='1h')
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices + 1.0,
            'low': prices - 1.0,
            'close': prices + 0.5,
            'tick_volume': np.random.randint(100, 1000, n)
        })
        lstm_probs = np.random.uniform(0.4, 0.8, n)
        res = run_fvg_backtest(df, lstm_probs=lstm_probs, mode='lstm')
        self.assertIn('win_rate', res)
        self.assertIn('profit_factor', res)
        self.assertIn('total_r', res)

    def test_run_fvg_backtest_valid_ensemble(self):
        n = 50
        dates = pd.date_range('2025-01-01', periods=n, freq='1h')
        prices = 100.0 + np.cumsum(np.random.randn(n))
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices + 1.0,
            'low': prices - 1.0,
            'close': prices + 0.5,
            'tick_volume': np.random.randint(100, 1000, n)
        })
        lstm_probs = np.ones(n) * 0.7
        tfm_signals = np.ones(n) * 1.0
        res = run_fvg_backtest(df, lstm_probs=lstm_probs, tfm_signals=tfm_signals, mode='ensemble')
        self.assertIn('win_rate', res)

    def test_process_single_timeframe_invalid_file(self):
        res = process_single_timeframe("non_existent_file.csv")
        self.assertEqual(res['total_bars'], 0)
        self.assertEqual(res['opt_params'], 'Failed')

    def test_process_single_timeframe_missing_cols(self):
        file_path = os.path.join(self.temp_dir.name, "missing.csv")
        pd.DataFrame({'a': [1, 2]}).to_csv(file_path, index=False)
        res = process_single_timeframe(file_path)
        self.assertEqual(res['opt_params'], 'Invalid Cols')

    @patch('run_train_backtest.train_tf_dataset')
    def test_process_single_timeframe_valid(self, mock_train):
        mock_train.return_value = None
        file_path = os.path.join(self.temp_dir.name, "valid_15m.csv")
        n = 30
        df = pd.DataFrame({
            'datetime': pd.date_range('2025-01-01', periods=n, freq='15min'),
            'open': np.ones(n),
            'high': np.ones(n)+1,
            'low': np.ones(n)-1,
            'close': np.ones(n),
            'volume': np.ones(n)*100
        })
        df.to_csv(file_path, index=False)
        res = process_single_timeframe(file_path)
        self.assertEqual(res['timeframe'], 'valid_15m.csv')
        self.assertEqual(res['opt_params'], 'N/A')

    def test_print_header_and_summary(self):
        _print_header()
        sample_results = [{
            'timeframe': 'M15',
            'total_bars': 100,
            'triggers': 5,
            'trades': 3,
            'win_rate': 66.7,
            'profit_factor': 1.5,
            'total_r': 2.5,
            'net_return': 5.0,
            'max_dd': -1.2,
            'opt_params': "{'lookback': 10}"
        }]
        _print_summary(sample_results)

    @patch('sys.argv', ['run_train_backtest.py', '--mode', 'sequential'])
    @patch('glob.glob', return_value=[])
    def test_main_no_files(self, mock_glob):
        main()

if __name__ == '__main__':
    unittest.main()
