import unittest
import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from google_timesfm_model import GoogleTimesFMForecaster, evaluate_timesfm_on_dataframe, load_env_file

class TestGoogleTimesFMModel(unittest.TestCase):

    def test_load_env_file_hf_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = os.path.join(temp_dir, '.env')
            with open(env_file, 'w') as f:
                f.write("HF_TOKEN=test_hf_token_12345\n")
            
            with patch('os.path.abspath', return_value=os.path.join(temp_dir, 'google_timesfm_model.py')):
                with patch.dict(os.environ, {}, clear=False):
                    if 'HF_TOKEN' in os.environ:
                        del os.environ['HF_TOKEN']
                    load_env_file()
                    self.assertEqual(os.environ.get('HF_TOKEN'), 'test_hf_token_12345')

    def test_env_hf_token_configured(self):
        load_env_file()
        self.assertIsNotNone(os.environ.get("HF_TOKEN"))

    def test_init(self):
        forecaster = GoogleTimesFMForecaster(horizon=5, context_len=100, hf_token="fake_token")
        self.assertEqual(forecaster.horizon, 5)
        self.assertEqual(forecaster.context_len, 100)
        self.assertFalse(forecaster.is_loaded)

    def test_forecast_series_short_input(self):
        forecaster = GoogleTimesFMForecaster()
        with self.assertRaises(ValueError):
            forecaster.forecast_series([1.0] * 10)

    def test_forecast_series_fallback(self):
        forecaster = GoogleTimesFMForecaster(horizon=5)
        forecaster.is_loaded = False
        series = np.linspace(100, 110, 30)
        preds, dir_sig = forecaster.forecast_series(series)
        self.assertEqual(len(preds), 5)
        self.assertEqual(dir_sig, 1)

    def test_forecast_series_loaded_mock(self):
        forecaster = GoogleTimesFMForecaster(horizon=5)
        forecaster.is_loaded = True
        mock_model = MagicMock()
        mock_model.forecast.return_value = (np.array([[101, 102, 103, 104, 105]]), None)
        forecaster.model = mock_model

        series = np.linspace(100, 102, 30)
        preds, dir_sig = forecaster.forecast_series(series)
        self.assertEqual(len(preds), 5)
        self.assertEqual(dir_sig, 1)

    def test_cleanup(self):
        forecaster = GoogleTimesFMForecaster()
        forecaster.is_loaded = True
        forecaster.model = "mock_model"
        forecaster.cleanup()
        self.assertFalse(forecaster.is_loaded)
        self.assertIsNone(forecaster.model)

    def test_evaluate_timesfm_short_data(self):
        df_short = pd.DataFrame({'close': np.ones(30)})
        res = evaluate_timesfm_on_dataframe(df_short)
        self.assertEqual(res['status'], 'ERROR')

    def test_evaluate_timesfm_valid_data(self):
        df = pd.DataFrame({'close': np.linspace(100, 150, 100)})
        forecaster = GoogleTimesFMForecaster(horizon=5)
        res = evaluate_timesfm_on_dataframe(df, forecast_horizon=5, test_windows=3, forecaster=forecaster)
        self.assertEqual(res['status'], 'OK')
        self.assertIn('directional_accuracy', res)

if __name__ == '__main__':
    unittest.main()
