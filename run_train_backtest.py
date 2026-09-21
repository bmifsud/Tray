import os
import json
import numpy as np
import pandas as pd
import torch
import glob
from extract_nas100_raw import extract_data
from train_nas100_lstm import train, DirectionalLSTM, compute_features

def execute_pipeline_for_file(filepath):
    filename = os.path.basename(filepath)
    print(f"\n==================================================")
    print(f"      PROCESSING TIMEFRAME: {filename}")
    print(f"==================================================")
    
    # 0. Extract raw data
    print("=== STEP 0: Extracting raw data ===")
    extract_data(filepath, download=False)

    # 1. Train Directional LSTM
    print("\n=== STEP 1: Executing Leak-Free Directional LSTM Training & ONNX Export ===")
    success = train()
    if not success:
        print("[INFO] Skipping backtest due to insufficient dataset size.")
        return {
            'timeframe': filename,
            'total_bars': 0,
            'triggers': 0,
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_r': 0.0,
            'net_return': 0.0,
            'max_dd': 0.0
        }

    print("\n=== STEP 2: Running Broker-Aligned Confluence Backtest Analysis ===")

    if not os.path.exists('scaler_params.json') or not os.path.exists('best_model.pth'):
        print("[WARN] Model artifacts missing. Skipping backtest.")
        return {
            'timeframe': filename,
            'total_bars': 0,
            'triggers': 0,
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_r': 0.0,
            'net_return': 0.0,
            'max_dd': 0.0
        }

    with open('scaler_params.json', 'r') as f:
        scaler_params = json.load(f)

    feature_names = scaler_params['feature_names']
    mean_ = np.array(scaler_params['mean_'], dtype=np.float32)
    scale_ = np.array(scaler_params['scale_'], dtype=np.float32)
    lookback = scaler_params.get('lookback', 15)

    raw_df = pd.read_csv('nas100_raw.csv')
    df = compute_features(raw_df.copy())
    df_clean = df.dropna(subset=feature_names + ['target']).reset_index(drop=True)

    n_samples = len(df_clean)
    split_idx = int(n_samples * 0.8)

    if (n_samples - split_idx) <= lookback + 10:
        print("[INFO] Not enough out-of-sample data points for this timeframe.")
        return {
            'timeframe': filename,
            'total_bars': max(0, n_samples - split_idx),
            'triggers': 0,
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_r': 0.0,
            'net_return': 0.0,
            'max_dd': 0.0
        }

    # Prepare features for test set
    X_raw = df_clean[feature_names].values
    X_scaled = (X_raw - mean_) / (scale_ + 1e-8)

    # Build sequences for out-of-sample validation
    X_val_seq = []
    val_indices = []
    for i in range(split_idx, n_samples - lookback):
        window = X_scaled[i:i + lookback]
        X_val_seq.append(window)
        val_indices.append(i + lookback)

    X_val_seq = np.array(X_val_seq, dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DirectionalLSTM(input_size=len(feature_names), hidden_size=32).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
        lstm_probs = model(val_tensor).cpu().numpy().flatten()

    # Align test dataframe
    test_df = df_clean.iloc[val_indices].copy().reset_index(drop=True)
    test_df['lstm_prob'] = lstm_probs

    # ATR(14) for volatility-adjusted thresholds & EMA(50) for macro trend alignment
    high = test_df['high']
    low = test_df['low']
    close = test_df['close']
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    test_df['atr'] = tr.rolling(14, min_periods=3).mean().bfill()
    test_df['ema50'] = test_df['close'].ewm(span=50, adjust=False).mean()

    # Backtest Parameters
    starting_balance = 10000.0
    risk_per_trade = 0.01  # 1% per trade
    max_holding = 15       # Max bars to hold active position
    fill_window = 6        # Max bars to wait for limit order fill at CE
    rr_ratio = 2.0         # 2:1 Reward to Risk
    sl_buffer_ratio = 0.75 # Buffer beyond FVG edge

    trades = []
    triggers_count = 0

    for i in range(2, len(test_df) - max_holding):
        bar_i2 = test_df.iloc[i - 2]
        bar_i = test_df.iloc[i]
        atr = bar_i['atr']
        prob = bar_i['lstm_prob']
        ema = bar_i['ema50']

        # 1. Bullish FVG: low[i] > high[i-2] (with 50-EMA Trend Filter)
        if bar_i['low'] > bar_i2['high']:
            fvg_top = bar_i['low']
            fvg_bottom = bar_i2['high']
            gap_height = fvg_top - fvg_bottom

            if gap_height >= 0.15 * atr and prob >= 0.48 and bar_i['close'] >= ema:
                triggers_count += 1
                ce = fvg_bottom + 0.5 * gap_height
                sl = fvg_bottom - sl_buffer_ratio * gap_height
                risk = ce - sl

                if risk > 0:
                    tp = ce + rr_ratio * risk
                    
                    # Check limit order fill at CE
                    filled = False
                    fill_bar = -1
                    for b in range(i + 1, min(i + fill_window + 1, len(test_df))):
                        if test_df.iloc[b]['low'] <= ce:
                            filled = True
                            fill_bar = b
                            break

                    if filled and fill_bar > 0:
                        outcome = 'timeout'
                        pnl_r = 0.0

                        for h in range(fill_bar, min(fill_bar + max_holding, len(test_df))):
                            curr_bar = test_df.iloc[h]
                            if curr_bar['low'] <= sl:
                                outcome = 'sl'
                                pnl_r = -1.0
                                break
                            if curr_bar['high'] >= tp:
                                outcome = 'tp'
                                pnl_r = rr_ratio
                                break

                        if outcome == 'timeout':
                            final_c = test_df.iloc[min(fill_bar + max_holding - 1, len(test_df) - 1)]['close']
                            pnl_r = (final_c - ce) / risk

                        trades.append({
                            'type': 'BUY',
                            'entry_bar': fill_bar,
                            'entry_price': ce,
                            'sl': sl,
                            'tp': tp,
                            'outcome': outcome,
                            'pnl_r': pnl_r,
                            'return': pnl_r * risk_per_trade
                        })

        # 2. Bearish FVG: high[i] < low[i-2] (with 50-EMA Trend Filter)
        elif bar_i['high'] < bar_i2['low']:
            fvg_top = bar_i2['low']
            fvg_bottom = bar_i['high']
            gap_height = fvg_top - fvg_bottom

            if gap_height >= 0.15 * atr and prob <= 0.52 and bar_i['close'] <= ema:
                triggers_count += 1
                ce = fvg_bottom + 0.5 * gap_height
                sl = fvg_top + sl_buffer_ratio * gap_height
                risk = sl - ce

                if risk > 0:
                    tp = ce - rr_ratio * risk

                    filled = False
                    fill_bar = -1
                    for b in range(i + 1, min(i + fill_window + 1, len(test_df))):
                        if test_df.iloc[b]['high'] >= ce:
                            filled = True
                            fill_bar = b
                            break

                    if filled and fill_bar > 0:
                        outcome = 'timeout'
                        pnl_r = 0.0

                        for h in range(fill_bar, min(fill_bar + max_holding, len(test_df))):
                            curr_bar = test_df.iloc[h]
                            if curr_bar['high'] >= sl:
                                outcome = 'sl'
                                pnl_r = -1.0
                                break
                            if curr_bar['low'] <= tp:
                                outcome = 'tp'
                                pnl_r = rr_ratio
                                break

                        if outcome == 'timeout':
                            final_c = test_df.iloc[min(fill_bar + max_holding - 1, len(test_df) - 1)]['close']
                            pnl_r = (ce - final_c) / risk

                        trades.append({
                            'type': 'SELL',
                            'entry_bar': fill_bar,
                            'entry_price': ce,
                            'sl': sl,
                            'tp': tp,
                            'outcome': outcome,
                            'pnl_r': pnl_r,
                            'return': pnl_r * risk_per_trade
                        })

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)
    win_rate = 0.0
    profit_factor = 0.0
    total_r = 0.0
    net_return = 0.0
    max_dd = 0.0

    if n_trades > 0:
        wins = trades_df[trades_df['pnl_r'] > 0]
        losses = trades_df[trades_df['pnl_r'] <= 0]
        win_rate = len(wins) / n_trades
        total_r = float(trades_df['pnl_r'].sum())
        
        gross_profit = wins['pnl_r'].sum()
        gross_loss = abs(losses['pnl_r'].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 999.0

        returns = trades_df['return'].values
        equity_curve = starting_balance * np.cumprod(1 + returns)
        max_dd = float(np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve).max())
        net_return = float((equity_curve[-1] / starting_balance) - 1.0)

        print("\n==================================================")
        print(f"       CONFLUENCE BACKTEST SUMMARY ({filename})")
        print("==================================================")
        print(f"* Out-of-Sample Bars:       {len(test_df)}")
        print(f"* FVG + LSTM Triggers:      {triggers_count}")
        print(f"* Executed Limit Trades:    {n_trades}")
        print(f"* Winning Trades:           {len(wins)}")
        print(f"* Losing Trades:            {len(losses)}")
        print(f"* Win Rate:                 {win_rate * 100:.2f}%")
        print(f"* Profit Factor:            {profit_factor:.2f}")
        print(f"* Total R Gained:           {total_r:+.2f}R")
        print(f"* Net Return on Account:    {'+' if net_return > 0 else ''}{net_return * 100:.2f}% (Starting: ${starting_balance:,.2f})")
        print(f"* Maximum Drawdown:         -{max_dd * 100:.2f}%")
        print(f"* Trade Breakdown:          {trades_df['outcome'].value_counts().to_dict()}")
    else:
        print("\n==================================================")
        print(f"       CONFLUENCE BACKTEST SUMMARY ({filename})")
        print("==================================================")
        print(f"* Out-of-Sample Bars:       {len(test_df)}")
        print(f"* FVG + LSTM Triggers:      {triggers_count}")
        print(f"* Executed Limit Trades:    0")

    return {
        'timeframe': filename,
        'total_bars': len(test_df),
        'triggers': triggers_count,
        'trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_r': total_r,
        'net_return': net_return,
        'max_dd': max_dd
    }

def main():
    files = glob.glob('NasData/*.csv')
    files = sorted(files)
    
    results = []
    for f in files:
        res = execute_pipeline_for_file(f)
        results.append(res)
        
    print("\n\n=======================================================================================================")
    print("                              GLOBAL AGGREGATED TIMEFRAME RESULTS")
    print("=======================================================================================================")
    print(f"{'Timeframe':<22} | {'Bars':<6} | {'Triggers':<8} | {'Trades':<6} | {'Win Rate':<8} | {'P.Factor':<8} | {'Total R':<8} | {'Return':<9} | {'Max DD':<8}")
    print("-" * 103)
    for r in results:
        wr_str = f"{r['win_rate']*100:.1f}%"
        pf_str = f"{r['profit_factor']:.2f}"
        r_str = f"{r['total_r']:+.1f}R"
        ret_str = f"{r['net_return']*100:+.2f}%"
        dd_str = f"-{r['max_dd']*100:.1f}%"
        print(f"{r['timeframe']:<22} | {r['total_bars']:<6} | {r['triggers']:<8} | {r['trades']:<6} | {wr_str:<8} | {pf_str:<8} | {r_str:<8} | {ret_str:<9} | {dd_str:<8}")

if __name__ == "__main__":
    main()
