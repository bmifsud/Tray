import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

def extract_data():
    if not mt5.initialize():
        print("initialize() failed")
        mt5.shutdown()
        return

    symbol = "NAS100"

    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")

    timezone = pytz.timezone("UTC")
    utc_now = datetime.now(timezone)

    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_D1, utc_now, 10000)

    if rates is None or len(rates) == 0:
        print("Failed to get rates")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)

    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Filter out unclosed active daily bars
    df = df[df['time'].dt.date < utc_now.date()]

    # Select required columns: OHLCV + tick_volume
    # Actually in MT5 OHLCV is tick_volume or real_volume, we'll keep both if necessary but prompt asks for tick volume
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    df.to_csv('nas100_raw.csv', index=False)

    mt5.shutdown()

if __name__ == '__main__':
    extract_data()
