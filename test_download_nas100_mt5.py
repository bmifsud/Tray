import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import download_nas100_mt5

class TestDownloadNas100MT5(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    @patch('MetaTrader5.initialize', return_value=True)
    def test_initialize_mt5_success(self, mock_init):
        res = download_nas100_mt5.initialize_mt5()
        self.assertTrue(res)

    @patch('MetaTrader5.initialize', return_value=False)
    @patch('MetaTrader5.last_error', return_value=(1, "Error"))
    def test_initialize_mt5_failure(self, mock_err, mock_init):
        res = download_nas100_mt5.initialize_mt5()
        self.assertFalse(res)

    @patch('MetaTrader5.symbol_select', return_value=True)
    @patch('MetaTrader5.symbol_info')
    def test_resolve_symbol_requested(self, mock_info, mock_select):
        mock_info.return_value = MagicMock(visible=True)
        res = download_nas100_mt5.resolve_symbol("NAS100")
        self.assertEqual(res, "NAS100")

    @patch('MetaTrader5.copy_rates_from_pos')
    def test_download_bars(self, mock_rates):
        mock_rates.return_value = np.array([
            (1700000000, 100.0, 105.0, 99.0, 104.0, 50, 1, 0),
            (1700000060, 104.0, 106.0, 103.0, 105.0, 60, 1, 0)
        ], dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i4'), ('real_volume', 'i8')])

        out = download_nas100_mt5.download_bars("NAS100", "M1", count=2, output_dir=self.temp_dir.name)
        self.assertTrue(os.path.exists(out))
        df = pd.read_csv(out)
        self.assertEqual(len(df), 2)

    @patch('MetaTrader5.copy_ticks_range')
    def test_download_ticks(self, mock_ticks):
        mock_ticks.return_value = np.array([
            (1700000000, 100.0, 100.1, 0.0, 0, 1700000000000, 0, 0.0),
            (1700000001, 100.1, 100.2, 0.0, 0, 1700000001000, 0, 0.0)
        ], dtype=[('time', 'i8'), ('bid', 'f8'), ('ask', 'f8'), ('last', 'f8'), ('volume', 'i8'), ('time_msc', 'i8'), ('flags', 'i4'), ('volume_real', 'f8')])

        out = download_nas100_mt5.download_ticks("NAS100", count=2, output_dir=self.temp_dir.name)
        self.assertTrue(os.path.exists(out))
        df = pd.read_csv(out)
        self.assertEqual(len(df), 2)

    @patch('sys.argv', ['download_nas100_mt5.py', '--symbol', 'NAS100', '--timeframe', 'M1', '--bars', '10'])
    @patch('download_nas100_mt5.initialize_mt5', return_value=False)
    def test_main_init_failure(self, mock_init):
        with self.assertRaises(SystemExit):
            download_nas100_mt5.main()

if __name__ == '__main__':
    unittest.main()
