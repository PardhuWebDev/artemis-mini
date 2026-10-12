"""
ARTEMIS Mini - Data Loader
Loads OHLCV data from CSV and computes technical indicators used as
features in the trading environment's observation space.

Expected CSV columns (case-insensitive): Date, Open, High, Low, Close, Volume
Works with exports from yfinance, NSE bhavcopy, or IndiaQuant MCP.
"""

import pandas as pd
import numpy as np


def load_ohlcv(csv_path: str) -> pd.DataFrame:
    """Load a raw OHLCV CSV and normalize column names."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {"adj close": "close", "adjclose": "close"}
    df = df.rename(columns=rename_map)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, moving averages, MACD, and volatility features."""
    df = df.copy()

    # Simple & exponential moving averages
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI (14-period)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Rolling volatility (20-day std of returns)
    df["returns"] = df["close"].pct_change()
    df["volatility_20"] = df["returns"].rolling(20).std()

    # Price relative to moving averages (normalized features)
    df["price_to_sma10"] = df["close"] / df["sma_10"] - 1
    df["price_to_sma50"] = df["close"] / df["sma_50"] - 1

    df = df.dropna().reset_index(drop=True)
    return df


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Convenience wrapper: load raw CSV -> full feature dataframe."""
    df = load_ohlcv(csv_path)
    df = add_technical_indicators(df)
    return df


def make_synthetic_data(n_days: int = 750, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing the pipeline before
    real data is available. Uses a geometric Brownian motion walk
    with occasional regime shifts to mimic trending/ranging markets.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    price = 2500.0
    prices = []
    drift = 0.0003
    vol = 0.015

    for i in range(n_days):
        # occasional regime shift
        if i % 90 == 0:
            drift = rng.normal(0.0002, 0.0006)
            vol = abs(rng.normal(0.014, 0.005)) + 0.005

        shock = rng.normal(drift, vol)
        price *= (1 + shock)
        prices.append(price)

    close = np.array(prices)
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.004, n_days)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.004, n_days)))
    volume = rng.integers(500_000, 5_000_000, n_days)

    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


if __name__ == "__main__":
    # Quick smoke test using synthetic data
    raw = make_synthetic_data()
    raw.to_csv("/home/claude/artemis-mini/data/synthetic_sample.csv", index=False)
    prepared = add_technical_indicators(raw)
    print(prepared.tail())
    print(f"\nRows after indicator warm-up: {len(prepared)} / {len(raw)}")
