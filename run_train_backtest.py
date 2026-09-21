import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from train_nas100_lstm import train_tf_dataset, DirectionalLSTM, compute_features

def process_single_timeframe(filepath):
    filename = os.path.basename(filepath)
    clean_tf = os.path.splitext(filename)[0]
    
    try:
        raw_df = pd.read_csv(filepath)
    except Exception as e:
        return {'timeframe': filename, 'total_bars': 0, 'triggers': 0, 'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_r': 0.0, 'net_return': 0.0, 'max_dd': 0.0, 'opt_params': 'Failed'}

    raw_df = raw_df.rename(columns={'datetime': 'time', 'volume': 'tick_volume'})
    if 'time' in raw_df.columns:
        raw_df['time'] = pd.to_datetime(raw_df['time'], utc=True)
        raw_df = raw_df.sort_values('time').reset_index(drop=True)

    required_cols = {'open', 'high', 'low', 'close', 'tick_volume'}
    if not required_cols.issubset(raw_df.columns):
        return {'timeframe': filename, 'total_bars': 0, 'triggers': 0, 'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_r': 0.0, 'net_return': 0.0, 'max_dd': 0.0, 'opt_params': 'Invalid Cols'}

    train_res = train_tf_dataset(raw_df, filename, optimize_grid=True)
    if train_res is None:
        return {'timeframe': filename, 'total_bars': 0, 'triggers': 0, 'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_r': 0.0, 'net_return': 0.0, 'max_dd': 0.0, 'opt_params': 'N/A'}

    df_clean = train_res['df_clean']
    scaler_params = train_res['scaler_params']
    best_model_state = train_res['model_state']
    best_params = train_res['best_params']

    feature_names = scaler_params['feature_names']
    mean_ = np.array(scaler_params['mean_'], dtype=np.float32)
    scale_ = np.array(scaler_params['scale_'], dtype=np.float32)
    lookback = scaler_params['lookback']
    hidden_size = scaler_params['hidden_size']

    n_samples = len(df_clean)
    split_idx = int(n_samples * 0.8)

    if (n_samples - split_idx) <= lookback + 10:
        return {'timeframe': filename, 'total_bars': max(0, n_samples - split_idx), 'triggers': 0, 'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_r': 0.0, 'net_return': 0.0, 'max_dd': 0.0, 'opt_params': f"L={lookback},H={hidden_size}"}

    X_raw = df_clean[feature_names].values
    X_scaled = (X_raw - mean_) / (scale_ + 1e-8)

    X_val_seq = []
    val_indices = []
    for i in range(split_idx, n_samples - lookback):
        window = X_scaled[i:i + lookback]
        X_val_seq.append(window)
        val_indices.append(i + lookback)

    X_val_seq = np.array(X_val_seq, dtype=np.float32)

    device = torch.device('cpu')
    model = DirectionalLSTM(input_size=len(feature_names), hidden_size=hidden_size).to(device)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
        lstm_probs = model(val_tensor).numpy().flatten()

    test_df = df_clean.iloc[val_indices].copy().reset_index(drop=True)
    test_df['lstm_prob'] = lstm_probs

    high = test_df['high']
    low = test_df['low']
    close = test_df['close']
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    test_df['atr'] = tr.rolling(14, min_periods=5).mean().bfill()
    test_df['ema'] = close.ewm(span=50, adjust=False).mean()

    starting_balance = 10000.0
    balance = starting_balance
    risk_per_trade = 0.01
    rr_ratio = 2.0
    sl_buffer_ratio = 0.1
    max_holding = 20
    fill_window = 3

    triggers_count = 0
    trades = []

    for i in range(2, len(test_df)):
        bar_i = test_df.iloc[i]
        bar_i2 = test_df.iloc[i - 2]
        prob = bar_i['lstm_prob']
        atr = bar_i['atr']
        ema = bar_i['ema']

        # Bullish FVG
        if bar_i['low'] > bar_i2['high']:
            fvg_bottom = bar_i2['high']
            fvg_top = bar_i['low']
            gap_height = fvg_top - fvg_bottom

            if gap_height >= 0.15 * atr and prob >= 0.48 and bar_i['close'] >= ema:
                triggers_count += 1
                ce = fvg_bottom + 0.5 * gap_height
                sl = fvg_bottom - sl_buffer_ratio * gap_height
                risk = ce - sl

                if risk > 0:
                    tp = ce + rr_ratio * risk
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

                        trades.append({'outcome': outcome, 'pnl_r': pnl_r, 'return': pnl_r * risk_per_trade})

        # Bearish FVG
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

                        trades.append({'outcome': outcome, 'pnl_r': pnl_r, 'return': pnl_r * risk_per_trade})

    n_trades = len(trades)
    win_rate = 0.0
    profit_factor = 0.0
    total_r = 0.0
    net_return = 0.0
    max_dd = 0.0

    if n_trades > 0:
        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df['pnl_r'] > 0]
        losses = trades_df[trades_df['pnl_r'] < 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

        gross_profit = wins['pnl_r'].sum()
        gross_loss = abs(losses['pnl_r'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        total_r = trades_df['pnl_r'].sum()

        equity_curve = [starting_balance]
        for ret in trades_df['return']:
            equity_curve.append(equity_curve[-1] * (1.0 + ret))

        equity_curve = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak
        max_dd = np.max(drawdowns)
        net_return = (equity_curve[-1] - starting_balance) / starting_balance

    opt_str = f"L={lookback},H={hidden_size}"
    print(f"[DONE] {filename:<22} | Opt: {opt_str:<10} | Trades: {n_trades:<5} | WR: {win_rate*100:5.1f}% | Total R: {total_r:+6.1f}R | Return: {net_return*100:+6.1f}%")

    return {
        'timeframe': filename,
        'total_bars': len(test_df),
        'triggers': triggers_count,
        'trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_r': total_r,
        'net_return': net_return,
        'max_dd': max_dd,
        'opt_params': opt_str
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded concurrent evaluation")
    parser.add_argument("--mode", type=str, default="threads", choices=["threads", "sequential"], help="Execution mode")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads")
    args = parser.parse_args()

    files = [f for f in glob.glob('NasData/*.csv') if 'ticks' not in f.lower()]
    files = sorted(files)

    print(f"[INFO] Launching Multi-Threaded Grid Optimization and Backtest across {len(files)} timeframes (Mode: {args.mode}, Workers: {args.workers})...")
    print("=" * 118)

    results = []
    if args.mode == "threads" and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {executor.submit(process_single_timeframe, f): f for f in files}
            for future in as_completed(future_to_file):
                res = future.result()
                results.append(res)
    else:
        for f in files:
            res = process_single_timeframe(f)
            results.append(res)

    results = sorted(results, key=lambda x: x['timeframe'])

    print("\n\n=======================================================================================================================")
    print("                                  GLOBAL MULTI-TIMEFRAME OPTIMIZATION AND BACKTEST RESULTS")
    print("=======================================================================================================================")
    print(f"{'Timeframe':<22} | {'Opt Config':<10} | {'Bars':<6} | {'Triggers':<8} | {'Trades':<6} | {'Win Rate':<8} | {'P.Factor':<8} | {'Total R':<8} | {'Return':<9} | {'Max DD':<8}")
    print("-" * 118)
    for r in results:
        wr_str = f"{r['win_rate']*100:.1f}%"
        pf_str = f"{r['profit_factor']:.2f}"
        r_str = f"{r['total_r']:+.1f}R"
        ret_str = f"{r['net_return']*100:+.2f}%"
        dd_str = f"-{r['max_dd']*100:.1f}%"
        opt_str = r.get('opt_params', 'N/A')
        print(f"{r['timeframe']:<22} | {opt_str:<10} | {r['total_bars']:<6} | {r['triggers']:<8} | {r['trades']:<6} | {wr_str:<8} | {pf_str:<8} | {r_str:<8} | {ret_str:<9} | {dd_str:<8}")

if __name__ == '__main__':
    main()
