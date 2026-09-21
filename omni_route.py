import os
import json
import numpy as np
import pandas as pd
from run_train_backtest import process_single_timeframe

class OmniRouter:
    """Dynamic multi-timeframe model router and portfolio selector.
    
    Ranks timeframe models by risk-adjusted expectancy and routes live data
    to the optimal time-horizon model based on performance score.
    """
    def __init__(self, data_dir="NasData", min_trades=5, min_pf=1.0):
        self.data_dir = data_dir
        self.min_trades = min_trades
        self.min_pf = min_pf
        self.routes = {}

    def score_performance(self, res):
        if res['trades'] < self.min_trades or res['profit_factor'] < self.min_pf:
            return -999.0
        wr = res['win_rate'] / 100.0
        pf = res['profit_factor']
        total_r = res['total_r']
        max_dd = abs(res['max_dd']) + 1e-4
        score = (total_r * pf * wr) / max_dd
        return round(float(score), 4)

    def scan_and_route(self, files=None):
        if files is None:
            import glob
            files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        results = []
        for filepath in files:
            res = process_single_timeframe(filepath)
            if res:
                res['score'] = self.score_performance(res)
                results.append(res)

        results.sort(key=lambda x: x['score'], reverse=True)

        self.routes = {
            'best_timeframe': results[0]['timeframe'] if results else None,
            'timestamp': pd.Timestamp.now(tz='UTC').isoformat(),
            'total_timeframes_evaluated': len(results),
            'active_routes': [r for r in results if r['score'] > 0],
            'all_rankings': results
        }
        return self.routes

    def save_routing_table(self, output_path="models/omni_route_table.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.routes, f, indent=2)
        print(f"[OmniRoute] Routing table saved to {output_path}")

    def route(self, timeframe):
        """Route trade requests to the model configured for a given timeframe."""
        tf_clean = os.path.splitext(os.path.basename(timeframe))[0]
        model_path = f"models/nas100_lstm_{tf_clean}.onnx"
        scaler_path = f"models/scaler_params_{tf_clean}.json"

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            return {
                'status': 'ACTIVE',
                'timeframe': tf_clean,
                'onnx_model': model_path,
                'scaler_params': scaler_path
            }
        return {'status': 'INACTIVE', 'reason': 'Model artifacts missing'}

if __name__ == '__main__':
    router = OmniRouter()
    print("[OmniRoute] Initializing automated multi-timeframe model routing scan...")
    routes = router.scan_and_route()
    router.save_routing_table()
    
    print("\n" + "=" * 80)
    print("OMNIROUTE MULTI-TIMEFRAME RANKINGS")
    print("=" * 80)
    for idx, r in enumerate(routes['all_rankings'], 1):
        print(f"{idx:2d}. {r['timeframe']:<22} | Score: {r['score']:6.2f} | PF: {r['profit_factor']:5.2f} | WR: {r['win_rate']:5.1f}% | Total R: {r['total_r']:+6.1f}R")
