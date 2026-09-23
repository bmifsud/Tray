import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
import argparse
import os
import sys

# Standard MT5 Timeframes Mapping
TIMEFRAME_MAP = {
    'M1': (mt5.TIMEFRAME_M1, '1_minute'),
    'M3': (mt5.TIMEFRAME_M3, '3_minute'),
    'M5': (mt5.TIMEFRAME_M5, '5_minute'),
    'M15': (mt5.TIMEFRAME_M15, '15_minute'),
    'M30': (mt5.TIMEFRAME_M30, '30_minute'),
    'M45': (mt5.TIMEFRAME_M15, '45_minute'), # fallback if broker does not support M45
    'H1': (mt5.TIMEFRAME_H1, '1_hour'),
    'H2': (mt5.TIMEFRAME_H2, '2_hour'),
    'H3': (mt5.TIMEFRAME_H3, '3_hour'),
    'H4': (mt5.TIMEFRAME_H4, '4_hour'),
    'D1': (mt5.TIMEFRAME_D1, 'daily'),
    'W1': (mt5.TIMEFRAME_W1, 'weekly'),
    'MN1': (mt5.TIMEFRAME_MN1, 'monthly'),
}

# Common broker symbol aliases for Nasdaq 100
DEFAULT_SYMBOL_CANDIDATES = [
    "NAS100", "NAS100.cash", "USTEC", "US100", "NQ", "US100.cash",
    "NAS100m", "NAS100USD", "NDX", "US Tech 100", "TECH100"
]

def initialize_mt5(path=None, login=None, password=None, server=None):
    """
    Initializes connection to MetaTrader 5 terminal.
    """
    # Automatically load local .env file if present
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_file):
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

    init_kwargs = {}
    
    # Check custom path or known default
    default_path = r"C:\Program Files\Blueberry Markets MetaTrader 5\terminal64.exe"
    target_path = path or os.environ.get("MT5_PATH") or (default_path if os.path.exists(default_path) else None)
    if target_path:
        init_kwargs['path'] = target_path

    # Check login credentials
    acc_login = login or os.environ.get("MT5_LOGIN")
    acc_pass = password or os.environ.get("MT5_PASSWORD")
    acc_server = server or os.environ.get("MT5_SERVER")

    if acc_login and acc_pass and acc_server:
        init_kwargs['login'] = int(acc_login)
        init_kwargs['password'] = acc_pass
        init_kwargs['server'] = acc_server

    print(f"[INFO] Initializing MetaTrader 5 connection...")
    if target_path:
        print(f"       -> Target Terminal Path: {target_path}")

    initialized = mt5.initialize(**init_kwargs)
    if not initialized:
        err = mt5.last_error()
        print(f"[ERROR] mt5.initialize() failed. Error code: {err}")
        print("\n[TROUBLESHOOTING GUIDE]")
        print("1. Ensure MetaTrader 5 terminal is open on your desktop.")
        print("2. Ensure you are logged into your trading account (Demo or Live) inside the terminal.")
        print("3. In MT5: click Tools -> Options -> Expert Advisors -> check 'Allow Algorithmic Trading'.")
        print("4. You can also pass credentials via CLI: --login <number> --password <pwd> --server <server>")
        return False

    terminal_info = mt5.terminal_info()
    account_info = mt5.account_info()
    
    if terminal_info:
        print(f"[INFO] Connected to: {terminal_info.name} (Build: {terminal_info.build})")
    if account_info:
        print(f"[INFO] Active Account: {account_info.login} ({account_info.server}) | Balance: ${account_info.balance:,.2f}")
    else:
        print("[WARN] Connected to terminal, but no active account login detected. Some data requests may fail.")

    return True

def resolve_symbol(requested_symbol=None):
    """
    Identifies and selects the exact broker symbol for NAS100.
    """
    if requested_symbol:
        selected = mt5.symbol_select(requested_symbol, True)
        if selected:
            print(f"[INFO] Selected symbol: {requested_symbol}")
            return requested_symbol
        else:
            print(f"[WARN] Requested symbol '{requested_symbol}' not found. Searching broker symbols...")

    # Search candidates
    for cand in DEFAULT_SYMBOL_CANDIDATES:
        info = mt5.symbol_info(cand)
        if info is not None:
            mt5.symbol_select(cand, True)
            print(f"[INFO] Automatically resolved NAS100 to broker symbol: '{cand}'")
            return cand

    # Broader pattern search
    patterns = ["*NAS*", "*100*", "*USTEC*", "*TECH*"]
    found_symbols = []
    for pat in patterns:
        symbols = mt5.symbols_get(pat)
        if symbols:
            for s in symbols:
                found_symbols.append(s.name)

    found_symbols = list(set(found_symbols))
    if found_symbols:
        best_match = found_symbols[0]
        mt5.symbol_select(best_match, True)
        print(f"[INFO] Matched broker symbol from catalog: '{best_match}' (available: {found_symbols[:5]})")
        return best_match

    print("[ERROR] Could not find any NAS100 / Nasdaq symbol in broker catalog.")
    return None

def download_bars(symbol, timeframe_key, count=100000, output_dir="NasData"):
    """
    Downloads historical OHLCV bar data from MT5 and formats as CSV.
    For M45 (which MT5 does not natively support), resamples M15 bars into authentic 45-minute candles.
    """
    os.makedirs(output_dir, exist_ok=True)

    if timeframe_key not in TIMEFRAME_MAP:
        print(f"[ERROR] Invalid timeframe: {timeframe_key}. Choices: {list(TIMEFRAME_MAP.keys())}")
        return None

    tf_enum, tf_label = TIMEFRAME_MAP[timeframe_key]
    print(f"\n[INFO] Downloading up to {count:,} bars for {symbol} on {timeframe_key} ({tf_label})...")

    if timeframe_key == 'M45':
        m15_count = min(count * 3, 300000)
        print(f"       -> M45 not native to MT5; fetching {m15_count:,} M15 bars to resample into 45-minute candles...")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, m15_count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            print(f"[ERROR] Failed to fetch M15 rates for M45 resampling. Error: {err}")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.set_index('time')
        # Resample to 45T
        resampled = df.resample('45min', origin='start_day').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum',
            'spread': 'mean',
            'real_volume': 'sum'
        }).dropna(subset=['open', 'close']).reset_index()
        df = resampled[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
        df = df.sort_values('time').reset_index(drop=True)
        if len(df) > count:
            df = df.iloc[-count:].reset_index(drop=True)
    else:
        rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            print(f"[ERROR] Failed to fetch rates for {symbol} ({timeframe_key}). Error: {err}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
        df = df.sort_values('time').reset_index(drop=True)

    filename = f"NQ_in_{tf_label}.csv"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        try:
            old_df = pd.read_csv(output_path)
            if 'datetime' in old_df.columns:
                old_df = old_df.rename(columns={'datetime': 'time', 'volume': 'tick_volume'})
            old_df['time'] = pd.to_datetime(old_df['time'], utc=True)
            df = pd.concat([old_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=['time'], keep='last')
            df = df.sort_values('time').reset_index(drop=True)
            print(f"       -> Incremental update: merged new bars with existing CSV (total: {len(df):,} bars)")
        except Exception as e:
            print(f"[WARN] Failed to merge with existing CSV ({e}), overwriting...")

    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved {len(df):,} bars to: {output_path}")
    print(f"          Date Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return output_path

def download_ticks(symbol, count=500000, output_dir="NasData"):
    """
    Downloads high-resolution tick data from MT5.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Downloading up to {count:,} ticks for {symbol}...")

    now = datetime.datetime.now(datetime.timezone.utc)
    # Estimate days needed based on typical NAS100 tick volume (~150,000-250,000 ticks per active trading day)
    days_back = max(3, int(count / 150000 * 2) + 3)
    start_time = now - datetime.timedelta(days=days_back)

    print(f"       -> Querying broker tick range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC to now...")
    ticks = mt5.copy_ticks_range(symbol, start_time, now, mt5.COPY_TICKS_ALL)

    # If range returned fewer or failed, fallback to copy_ticks_from
    if ticks is None or len(ticks) == 0:
        print(f"[WARN] copy_ticks_range returned no ticks, falling back to copy_ticks_from...")
        ticks = mt5.copy_ticks_from(symbol, start_time, count, mt5.COPY_TICKS_ALL)

    if ticks is None or len(ticks) == 0:
        err = mt5.last_error()
        print(f"[ERROR] Failed to fetch ticks for {symbol}. Error: {err}")
        return None

    df = pd.DataFrame(ticks)
    # Convert millisecond timestamp to UTC datetime
    df['time'] = pd.to_datetime(df['time_msc'], unit='ms', utc=True)

    # Sort and take the latest `count` ticks
    df = df.sort_values('time_msc').reset_index(drop=True)
    if len(df) > count:
        df = df.iloc[-count:].reset_index(drop=True)

    filename = f"{symbol.replace('.', '_')}_ticks.csv"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        try:
            old_df = pd.read_csv(output_path)
            if 'time_msc' in old_df.columns:
                old_df['time'] = pd.to_datetime(old_df['time_msc'], unit='ms', utc=True)
            elif 'time' in old_df.columns:
                old_df['time'] = pd.to_datetime(old_df['time'], utc=True)
            df = pd.concat([old_df, df], ignore_index=True)
            subset_col = 'time_msc' if 'time_msc' in df.columns else 'time'
            df = df.drop_duplicates(subset=[subset_col], keep='last')
            df = df.sort_values(subset_col).reset_index(drop=True)
            print(f"       -> Incremental update: merged new ticks with existing CSV (total: {len(df):,} ticks)")
        except Exception as e:
            print(f"[WARN] Failed to merge existing tick CSV ({e}), overwriting...")

    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved {len(df):,} ticks to: {output_path}")
    print(f"          Date Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Download NAS100 Bar and Tick Data directly from MetaTrader 5.")
    parser.add_argument("--symbol", type=str, default=None, help="Broker symbol name (default: auto-detect e.g. NAS100, USTEC, US100)")
    parser.add_argument("--type", type=str, choices=['bars', 'ticks', 'both'], default='both', help="Data type to download: bars, ticks, or both (default: both)")
    parser.add_argument("--timeframe", type=str, default="all", help=f"Timeframe for bar data ({list(TIMEFRAME_MAP.keys())} or 'all')")
    parser.add_argument("--bars", type=int, default=100000, help="Number of bars to download (default: 100000)")
    parser.add_argument("--ticks", type=int, default=500000, help="Number of ticks to download (default: 500000)")
    parser.add_argument("--output_dir", type=str, default="NasData", help="Output directory to save CSVs (default: NasData)")
    parser.add_argument("--path", type=str, default=None, help="Path to terminal64.exe")
    parser.add_argument("--login", type=str, default=None, help="MT5 Account Login Number")
    parser.add_argument("--password", type=str, default=None, help="MT5 Account Password")
    parser.add_argument("--server", type=str, default=None, help="MT5 Broker Server Name")
    parser.add_argument("--update_raw", action="store_true", help="Also overwrite nas100_raw.csv with downloaded bar data")

    args = parser.parse_args()

    # 1. Initialize MT5
    if not initialize_mt5(path=args.path, login=args.login, password=args.password, server=args.server):
        sys.exit(1)

    try:
        # 2. Resolve Symbol
        symbol = resolve_symbol(args.symbol)
        if not symbol:
            sys.exit(1)

        # 3. Download Bar Data
        saved_bar_file = None
        if args.type in ['bars', 'both']:
            if args.timeframe.upper() == 'ALL':
                for tf in TIMEFRAME_MAP.keys():
                    saved_bar_file = download_bars(symbol, tf, count=args.bars, output_dir=args.output_dir)
            else:
                tf_clean = args.timeframe.upper()
                saved_bar_file = download_bars(symbol, tf_clean, count=args.bars, output_dir=args.output_dir)

            # If requested, update nas100_raw.csv for immediate model training
            if args.update_raw:
                m3_file = os.path.join(args.output_dir, "NQ_in_3_minute.csv")
                target_source = m3_file if os.path.exists(m3_file) else saved_bar_file
                if target_source and os.path.exists(target_source):
                    raw_df = pd.read_csv(target_source)
                    raw_df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].to_csv('nas100_raw.csv', index=False)
                    print(f"\n[INFO] Successfully updated 'nas100_raw.csv' with latest MT5 data (from {os.path.basename(target_source)}, {len(raw_df):,} bars) for training pipeline.")

        # 4. Download Tick Data
        if args.type in ['ticks', 'both']:
            download_ticks(symbol, count=args.ticks, output_dir=args.output_dir)

        print("\n==================================================")
        print("          MT5 DATA EXTRACTION COMPLETE")
        print("==================================================")
        print(f"* Symbol:      {symbol}")
        print(f"* Destination: {os.path.abspath(args.output_dir)}")
        print("==================================================\n")

    finally:
        mt5.shutdown()
        print("[INFO] MetaTrader 5 connection closed cleanly.")

if __name__ == "__main__":
    main()
