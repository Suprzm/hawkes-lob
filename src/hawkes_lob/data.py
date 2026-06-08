import ccxt
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


def fetch_trades(symbol: str = 'BTC/USDT',
                 n_trades: int = 1000,
                 save: bool = True) :
    """
    Fetches recent public trades from Binance.

    Args:
        symbol   : trading pair
        n_trades : number of recent trades to fetch (max 1000 per request)
        save     : whether to save raw data to data/raw/

    Returns:
        DataFrame with columns: timestamp, price, amount, side
    """
    exchange = ccxt.binance({'enableRateLimit': True})

    print(f"Fetching {n_trades} trades for {symbol}...")
    raw = exchange.fetch_trades(symbol, limit=n_trades)

    df = pd.DataFrame([{
        'timestamp': t['timestamp'] / 1000.0,  # convert ms to seconds
        'price':     t['price'],
        'amount':    t['amount'],
        'side':      t['side'],                 # 'buy' or 'sell'
    } for t in raw])

    df = df.sort_values('timestamp').reset_index(drop=True)

    if save:
        path = Path('data/raw')
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(path / f'{symbol.replace("/", "_")}_trades.csv', index=False)
        print(f"Saved to data/raw/{symbol.replace('/', '_')}_trades.csv")

    return df


def extract_event_times(df: pd.DataFrame,
                        side: str = 'both',
                        time_unit: float = 1.0) :
    """
    Args:
        time_unit : rescale time — use 0.001 to work in milliseconds,
                    10.0 to work in units of 10 seconds.
                    Default 1.0 = seconds.
    """
    if side != 'both':
        df = df[df['side'] == side]

    timestamps = df['timestamp'].values
    T = (timestamps[-1] - timestamps[0]) / time_unit
    events = (timestamps - timestamps[0]) / time_unit

    return events, T


def describe_trades(df: pd.DataFrame) :
    """Prints a summary of the fetched trade data."""
    T_seconds = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    n_buys = (df['side'] == 'buy').sum()
    n_sells = (df['side'] == 'sell').sum()

    print(f"Total trades     : {len(df)}")
    print(f"Time window      : {T_seconds:.1f} seconds ({T_seconds/60:.1f} minutes)")
    print(f"Buy trades       : {n_buys} ({100*n_buys/len(df):.1f}%)")
    print(f"Sell trades      : {n_sells} ({100*n_sells/len(df):.1f}%)")
    print(f"Avg trade rate   : {len(df)/T_seconds:.2f} trades/second")
    print(f"Price range      : {df['price'].min():.2f} — {df['price'].max():.2f}")