import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
import torch
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from train_nas100_lstm import train_tf_dataset, DirectionalLSTM, compute_features
from google_timesfm_model import GoogleTimesFMForecaster
def run_fvg_backtest(test_df, lstm_probs, tfm_signals=None, mode='ensemble'):
    """
    Backtests ICT Fair Value Gap (FVG) execution strategy with given ML signals.
    Modes:
      - 'lstm': uses only LSTM probability threshold
      - 'timesfm': uses only TimesFM directional signal
      - 'ensemble': requires agreement between LSTM and TimesFM
    """
    _close = test_df['close'].values
    _high  = test_df['high'].values
    _low   = test_df['low'].values
    N = len(test_df)
    
    if N < 20:
        return {'triggers': 0, 'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0, 'total_r': 0.0, 'net_return': 0.0, 'max_dd': 0.0}

    tr = np.maximum(_high - _low, np.maximum(np.abs(_high - np.roll(_close, 1)), np.abs(_low - np.roll(_close, 1))))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    ema = test_df['close'].ewm(span=20, adjust=False).mean().values

    arr_low  = test_df['low'].values
    arr_high = test_df['high'].values

    # Pre-calculate probabilities & signals arrays to avoid loops
    if lstm_probs is None:
        lstm_probs = np.full(N, 0.5)
    elif len(lstm_probs) < N:
        lstm_probs = np.concatenate([lstm_probs, np.full(N - len(lstm_probs), 0.5)])
    if tfm_signals is None:
        tfm_signals = np.full(N, 0.5)
    elif len(tfm_signals) < N:
        tfm_signals = np.concatenate([tfm_signals, np.full(N - len(tfm_signals), 0.5)])

    if mode == 'lstm':
        allow_bull = lstm_probs >= 0.48
        allow_bear = lstm_probs <= 0.52
    elif mode == 'timesfm':
        allow_bull = tfm_signals == 1
        allow_bear = tfm_signals == 0
    else: # ensemble
        allow_bull = (lstm_probs >= 0.48) & (tfm_signals == 1)
        allow_bear = (lstm_probs <= 0.52) & (tfm_signals == 0)

    atr_thresh = 0.15 * atr

    # Vectorized FVG condition evaluation (shifting arrays by appropriate offsets)
    lo_arr   = arr_low[:N-3]
    hi2_arr  = arr_high[:N-3]
    hi_arr   = arr_high[2:N-1]
    lo2_arr  = arr_low[2:N-1]
    cl_arr   = _close[2:N-1]
    ema_arr  = ema[2:N-1]
    bull_arr = allow_bull[2:N-1]
    bear_arr = allow_bear[2:N-1]
    ath_arr  = atr_thresh[2:N-1]

    # Bullish FVG pre-calculation
    bull_gap_height = lo_arr - hi2_arr
    bull_outer_cond = (lo_arr > hi2_arr)
    bullish_cond = bull_outer_cond & (bull_gap_height >= ath_arr) & bull_arr & (cl_arr >= ema_arr)

    # Bearish FVG pre-calculation
    bear_gap_height = lo2_arr - hi_arr
    bearish_cond = (hi_arr < lo2_arr) & (bear_gap_height >= ath_arr) & bear_arr & (cl_arr <= ema_arr)

    # Eliminate overlapping signals: in original code `elif hi < lo2` ensures bearish evaluates ONLY if `lo > hi2` was false.
    bearish_cond = bearish_cond & ~bull_outer_cond

    bull_indices = np.where(bullish_cond)[0] + 2
    bear_indices = np.where(bearish_cond)[0] + 2

    # Combine indices while keeping track of types for processing order
    all_indices = np.concatenate((bull_indices, bear_indices))
    is_bull = np.concatenate((np.ones(len(bull_indices), dtype=bool), np.zeros(len(bear_indices), dtype=bool)))

    if len(all_indices) > 0:
        sort_idx = np.argsort(all_indices)
        all_indices = all_indices[sort_idx]
        is_bull = is_bull[sort_idx]

    pnl_r_list = []
    triggers_count = len(all_indices)
    max_holding = 20
    fill_window = 10
    rr_ratio = 2.0
    sl_buffer_ratio = 0.5
    starting_balance = 10000.0
    risk_per_trade = 0.01

    # Python loop executes only on actual triggered bars
    for idx in range(len(all_indices)):
        i = all_indices[idx]
        if is_bull[idx]:
            hi2_val = arr_high[i-2]
            lo_val = arr_low[i-2]
            gap_height = lo_val - hi2_val
            ce = hi2_val + 0.5 * gap_height
            sl = hi2_val - sl_buffer_ratio * gap_height
            risk = ce - sl
            if risk > 0:
                tp = ce + rr_ratio * risk
                fill_bar = -1
                end_fill = min(i + fill_window + 1, N)
                for b in range(i + 1, end_fill):
                    if arr_low[b] <= ce:
                        fill_bar = b
                        break
                if fill_bar > 0:
                    pnl_r = 0.0
                    end_hold = min(fill_bar + max_holding, N)
                    for h in range(fill_bar, end_hold):
                        if arr_low[h] <= sl:
                            pnl_r = -1.0
                            break
                        if arr_high[h] >= tp:
                            pnl_r = rr_ratio
                            break
                    pnl_r_list.append(pnl_r)
        else:
            lo2_val = arr_low[i]
            hi_val = arr_high[i]
            gap_height = lo2_val - hi_val
            ce = hi_val + 0.5 * gap_height
            sl = lo2_val + sl_buffer_ratio * gap_height
            risk = sl - ce
            if risk > 0:
                tp = ce - rr_ratio * risk
                fill_bar = -1
                end_fill = min(i + fill_window + 1, N)
                for b in range(i + 1, end_fill):
                    if arr_high[b] >= ce:
                        fill_bar = b
                        break
                if fill_bar > 0:
                    pnl_r = 0.0
                    end_hold = min(fill_bar + max_holding, N)
                    for h in range(fill_bar, end_hold):
                        if arr_high[h] >= sl:
                            pnl_r = -1.0
                            break
                        if arr_low[h] <= tp:
                            pnl_r = rr_ratio
                            break
                    pnl_r_list.append(pnl_r)

    n_trades = len(pnl_r_list)
    win_rate = profit_factor = total_r = net_return = max_dd = 0.0

    if n_trades > 0:
        pnl_arr      = np.array(pnl_r_list, dtype=np.float64)
        wins_mask    = pnl_arr > 0
        gross_profit = pnl_arr[wins_mask].sum()
        gross_loss   = -pnl_arr[~wins_mask & (pnl_arr < 0)].sum()
        win_rate     = wins_mask.sum() / n_trades
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        total_r      = pnl_arr.sum()

        returns      = pnl_arr * risk_per_trade
        equity_curve = starting_balance * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
        peak         = np.maximum.accumulate(equity_curve)
        max_dd       = np.max((peak - equity_curve) / peak)
        net_return   = (equity_curve[-1] - starting_balance) / starting_balance

    return {
        'triggers': triggers_count,
        'trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_r': total_r,
        'net_return': net_return,
        'max_dd': max_dd
    }



def process_single_timeframe(filepath):
    torch.set_num_threads(1)
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

    train_res = train_tf_dataset(raw_df, filename, optimize_grid=True, seed=42)

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

    val_len = n_samples - split_idx - lookback
    val_indices = list(range(split_idx + lookback, n_samples))
    shape = (val_len, lookback, len(feature_names))
    strides = (X_scaled.strides[0], X_scaled.strides[0], X_scaled.strides[1])
    X_val_seq = np.lib.stride_tricks.as_strided(X_scaled[split_idx:], shape=shape, strides=strides).copy().astype(np.float32)

    model = DirectionalLSTM(input_size=len(feature_names), hidden_size=hidden_size)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        lstm_probs = model(torch.tensor(X_val_seq, dtype=torch.float32)).numpy().flatten()

    test_df = df_clean.iloc[val_indices].copy().reset_index(drop=True)
    test_df['lstm_prob'] = lstm_probs

    _close = test_df['close']
    _high  = test_df['high']
    _low   = test_df['low']
    tr = np.maximum(_high - _low, np.maximum(np.abs(_high - _close.shift(1)), np.abs(_low - _close.shift(1))))
    test_df['atr'] = tr.rolling(14, min_periods=5).mean().bfill()
    test_df['ema'] = _close.ewm(span=50, adjust=False).mean()

    # --- extract to numpy once; kills all per-row .iloc overhead ---
    arr_high  = test_df['high'].to_numpy()
    arr_low   = test_df['low'].to_numpy()
    arr_close = test_df['close'].to_numpy()
    arr_atr   = test_df['atr'].to_numpy()
    arr_ema   = test_df['ema'].to_numpy()
    arr_prob  = test_df['lstm_prob'].to_numpy()
    N         = len(arr_high)

    risk_per_trade  = 0.01
    rr_ratio        = 2.0
    sl_buffer_ratio = 0.1
    max_holding     = 20
    fill_window     = 3
    starting_balance = 10000.0

    triggers_count = 0
    pnl_r_list: list[float] = []

    for i in range(2, N):
        prob  = arr_prob[i]
        atr   = arr_atr[i]
        ema   = arr_ema[i]
        hi    = arr_high[i]
        lo    = arr_low[i]
        cl    = arr_close[i]
        hi2   = arr_high[i - 2]
        lo2   = arr_low[i - 2]

        # Bullish FVG
        if lo > hi2:
            gap_height = lo - hi2
            if gap_height >= 0.15 * atr and prob >= 0.48 and cl >= ema:
                triggers_count += 1
                fvg_bottom = hi2
                ce  = fvg_bottom + 0.5 * gap_height
                sl  = fvg_bottom - sl_buffer_ratio * gap_height
                risk = ce - sl
                if risk > 0:
                    tp = ce + rr_ratio * risk
                    fill_bar = -1
                    end_fill = min(i + fill_window + 1, N)
                    for b in range(i + 1, end_fill):
                        if arr_low[b] <= ce:
                            fill_bar = b
                            break
                    if fill_bar > 0:
                        pnl_r = 0.0
                        end_hold = min(fill_bar + max_holding, N)
                        for h in range(fill_bar, end_hold):
                            if arr_low[h] <= sl:
                                pnl_r = -1.0
                                break
                            if arr_high[h] >= tp:
                                pnl_r = rr_ratio
                                break
                        pnl_r_list.append(pnl_r)

        # Bearish FVG
        elif hi < lo2:
            gap_height = lo2 - hi
            if gap_height >= 0.15 * atr and prob <= 0.52 and cl <= ema:
                triggers_count += 1
                fvg_top = lo2
                ce   = hi + 0.5 * gap_height
                sl   = fvg_top + sl_buffer_ratio * gap_height
                risk = sl - ce
                if risk > 0:
                    tp = ce - rr_ratio * risk
                    fill_bar = -1
                    end_fill = min(i + fill_window + 1, N)
                    for b in range(i + 1, end_fill):
                        if arr_high[b] >= ce:
                            fill_bar = b
                            break
                    if fill_bar > 0:
                        pnl_r = 0.0
                        end_hold = min(fill_bar + max_holding, N)
                        for h in range(fill_bar, end_hold):
                            if arr_high[h] >= sl:
                                pnl_r = -1.0
                                break
                            if arr_low[h] <= tp:
                                pnl_r = rr_ratio
                                break
                        pnl_r_list.append(pnl_r)

    n_trades = len(pnl_r_list)
    win_rate = profit_factor = total_r = net_return = max_dd = 0.0

    if n_trades > 0:
        pnl_arr    = np.array(pnl_r_list, dtype=np.float64)
        wins_mask  = pnl_arr > 0
        gross_profit = pnl_arr[wins_mask].sum()
        gross_loss   = -pnl_arr[~wins_mask & (pnl_arr < 0)].sum()
        win_rate     = wins_mask.sum() / n_trades
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        total_r      = pnl_arr.sum()

        returns      = pnl_arr * risk_per_trade
        equity_curve = starting_balance * np.cumprod(np.concatenate([[1.0], 1.0 + returns]))
        peak         = np.maximum.accumulate(equity_curve)
        max_dd       = np.max((peak - equity_curve) / peak)
        net_return   = (equity_curve[-1] - starting_balance) / starting_balance

    opt_str = f"L={lookback},H={hidden_size}"

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

_W = 118  # table width constant

def _print_row(r):
    wr_str  = f"{r['win_rate']*100:.1f}%"
    pf_str  = f"{r['profit_factor']:.2f}"
    r_str   = f"{r['total_r']:+.1f}R"
    ret_str = f"{r['net_return']*100:+.2f}%"
    dd_str  = f"-{r['max_dd']*100:.1f}%"
    opt_str = r.get('opt_params', 'N/A')
    print(f"{r['timeframe']:<22} | {opt_str:<10} | {r['total_bars']:<6} | {r['triggers']:<8} | "
          f"{r['trades']:<6} | {wr_str:<8} | {pf_str:<8} | {r_str:<8} | {ret_str:<9} | {dd_str:<8}")

def _print_header():
    print("=" * _W)
    print(f"{'Timeframe':<22} | {'Opt Config':<10} | {'Bars':<6} | {'Triggers':<8} | "
          f"{'Trades':<6} | {'Win Rate':<8} | {'P.Factor':<8} | {'Total R':<8} | {'Return':<9} | {'Max DD':<8}")
    print("-" * _W)

def _print_summary(results):
    active = [r for r in results if r['trades'] > 0]
    if not active:
        print("\n[WARN] No tradeable timeframes found.")
        return

    # portfolio-level aggregation (equal-weight across TFs)
    total_trades   = sum(r['trades']    for r in active)
    total_triggers = sum(r['triggers']  for r in active)
    all_r          = sum(r['total_r']   for r in active)
    avg_wr         = sum(r['win_rate']  for r in active) / len(active)
    avg_pf         = sum(r['profit_factor'] for r in active) / len(active)
    avg_dd         = sum(r['max_dd']    for r in active) / len(active)
    avg_ret        = sum(r['net_return'] for r in active) / len(active)

    # score = (total_R × PF × WR) / (MaxDD + ε)  — same as OmniRouter
    def _score(r):
        return (r['total_r'] * r['profit_factor'] * r['win_rate']) / (r['max_dd'] + 1e-4) \
               if r['trades'] > 0 else -999.0

    ranked = sorted(active, key=_score, reverse=True)
    best   = ranked[0]

    print("\n" + "=" * _W)
    print("  CONSOLIDATED PORTFOLIO SUMMARY")
    print("=" * _W)
    print(f"  Timeframes evaluated : {len(results):>4}   |  Active (trades>0) : {len(active):>4}")
    print(f"  Total triggers       : {total_triggers:>6} |  Total trades       : {total_trades:>6}")
    print(f"  Avg Win Rate         : {avg_wr*100:>5.1f}% |  Avg Profit Factor  : {avg_pf:>6.2f}")
    print(f"  Portfolio Total R    : {all_r:>+6.1f}R |  Avg Net Return     : {avg_ret*100:>+6.2f}%")
    print(f"  Avg Max Drawdown     : {avg_dd*100:>5.1f}%")
    print("-" * _W)
    print(f"  🏆 BEST TIMEFRAME : {best['timeframe']}  "
          f"| Score: {_score(best):.3f}  | WR: {best['win_rate']*100:.1f}%  "
          f"| PF: {best['profit_factor']:.2f}  | Total R: {best['total_r']:+.1f}R")
    print("\n  TOP 5 RANKED TIMEFRAMES (by risk-adj. expectancy):")
    for idx, r in enumerate(ranked[:5], 1):
        print(f"    {idx}. {r['timeframe']:<25} score={_score(r):>7.3f}  "
              f"wr={r['win_rate']*100:.1f}%  pf={r['profit_factor']:.2f}  "
              f"R={r['total_r']:+.1f}  dd={r['max_dd']*100:.1f}%")
    print("=" * _W)

def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe LSTM backtest pipeline")
    parser.add_argument("--mode",    type=str, default="threads",
                        choices=["threads", "sequential"], help="Execution mode")
    parser.add_argument("--workers", type=int, default=0,
                        help="Thread workers (0 = auto: min(files, cpu_count-1))")
    args = parser.parse_args()

    files = sorted(f for f in glob.glob('NasData/*.csv') if 'ticks' not in f.lower())
    n     = len(files)

    # auto-size: leave 1 core for OS, cap at file count
    auto_workers = max(1, min(n, (os.cpu_count() or 4) - 1))
    workers      = args.workers if args.workers > 0 else auto_workers

    print(f"\n[INFO] {n} timeframes | mode={args.mode} | workers={workers}")
    print(f"[INFO] CPU cores available: {os.cpu_count()}")
    _print_header()

    results    = []
    done_count = [0]  # mutable for closure in thread context

    def _collect(res):
        done_count[0] += 1
        results.append(res)
        _print_row(res)        # live line as each TF finishes

    if args.mode == "threads" and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_single_timeframe, f): f for f in files}
            for fut in as_completed(futs):
                try:
                    _collect(fut.result())
                except Exception as exc:
                    fname = os.path.basename(futs[fut])
                    print(f"[ERROR] {fname}: {exc}")
    else:
        for f in files:
            _collect(process_single_timeframe(f))

    _print_summary(results)

if __name__ == '__main__':
    main()
