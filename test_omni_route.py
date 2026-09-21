import unittest
import os
import json
import tempfile
import pandas as pd
from unittest.mock import patch

from omni_route import OmniRouter

class TestOmniRouter(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_score_performance_low_trades_or_pf(self):
        router = OmniRouter(min_trades=5, min_pf=1.0)
        res1 = {'trades': 2, 'profit_factor': 1.5, 'win_rate': 60, 'total_r': 10, 'max_dd': -2}
        self.assertEqual(router.score_performance(res1), -999.0)

        res2 = {'trades': 10, 'profit_factor': 0.8, 'win_rate': 60, 'total_r': 10, 'max_dd': -2}
        self.assertEqual(router.score_performance(res2), -999.0)

    def test_score_performance_valid(self):
        router = OmniRouter(min_trades=5, min_pf=1.0)
        res = {'trades': 10, 'profit_factor': 2.0, 'win_rate': 50.0, 'total_r': 10.0, 'max_dd': -2.0}
        # wr = 0.5, pf = 2.0, total_r = 10.0, max_dd = 2.0 + 1e-4 -> (10 * 2 * 0.5) / 2.0001 = ~5.0
        score = router.score_performance(res)
        self.assertGreater(score, 0.0)

    @patch('omni_route.process_single_timeframe')
    def test_scan_and_route(self, mock_process):
        mock_process.return_value = {
            'timeframe': 'M15.csv',
            'trades': 10,
            'profit_factor': 2.0,
            'win_rate': 60.0,
            'total_r': 12.0,
            'max_dd': -1.5
        }
        router = OmniRouter()
        dummy_file = os.path.join(self.temp_dir.name, "M15.csv")
        with open(dummy_file, 'w') as f:
            f.write("dummy")

        routes = router.scan_and_route(files=[dummy_file])
        self.assertEqual(routes['best_timeframe'], 'M15.csv')
        self.assertEqual(routes['total_timeframes_evaluated'], 1)

    def test_save_routing_table(self):
        router = OmniRouter()
        router.routes = {'best_timeframe': 'M15'}
        out_path = os.path.join(self.temp_dir.name, "omni_route_table.json")
        router.save_routing_table(output_path=out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_route_inactive(self):
        router = OmniRouter()
        res = router.route("M15")
        self.assertEqual(res['status'], 'INACTIVE')

    def test_route_active(self):
        router = OmniRouter()
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "nas100_lstm_test_tf.onnx")
        scaler_path = os.path.join(models_dir, "scaler_params_test_tf.json")

        with open(model_path, 'w') as f:
            f.write("mock")
        with open(scaler_path, 'w') as f:
            f.write("{}")

        self.addCleanup(lambda: (os.remove(model_path) if os.path.exists(model_path) else None))
        self.addCleanup(lambda: (os.remove(scaler_path) if os.path.exists(scaler_path) else None))

        res = router.route("test_tf.csv")
        self.assertEqual(res['status'], 'ACTIVE')

if __name__ == '__main__':
    unittest.main()
