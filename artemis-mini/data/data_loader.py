"""
ARTEMIS Mini - Data Loader
Loads the real NSE dataset (artemis_dataset.csv) — 4 stocks, 2015-2026,
pre-engineered technical indicators and an LLM-derived regime signal,
with a pre-made chronological train/val/test split.

Dataset columns:
Date, ticker, Open, High, Low, Close, Volume, ret_1d, sma_20, sma_50,
ema_12, ema_26, macd, macd_signal, macd_hist, rsi_14, volatility_20,
price_to_sma50, vol_ratio, regime, regime_code, split
"""

import pandas as pd

VALID_TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the full multi-stock dataset and parse dates."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_ticker_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Filter to one ticker, sort chronologically, and rename columns to
    the lowercase names the TradingEnv expects (close, returns, etc.).
    Keeps the 'split' column so train/val/test stays exactly as
    pre-assigned in the dataset (chronological, no lookahead bias).
    """
    if ticker not in VALID_TICKERS:
        raise ValueError(f"Unknown ticker '{ticker}'. Choose from {VALID_TICKERS}")

    sub = df[df["ticker"] == ticker].sort_values("Date").reset_index(drop=True)

    sub = sub.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "ret_1d": "returns",
    })

    return sub


def split_by_phase(df: pd.DataFrame):
    """Split a single-ticker dataframe into train/val/test using the
    dataset's own 'split' column (chronological, pre-assigned)."""
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, val, test


def load_and_prepare(csv_path: str, ticker: str):
    """
    Convenience wrapper: load the real dataset, filter to one ticker,
    and return (train_df, val_df, test_df) ready for TradingEnv.
    """
    df = load_dataset(csv_path)
    ticker_df = get_ticker_data(df, ticker)
    return split_by_phase(ticker_df)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/artemis_dataset.csv"
    for ticker in VALID_TICKERS:
        train, val, test = load_and_prepare(csv_path, ticker)
        print(f"{ticker}: train={len(train)} val={len(val)} test={len(test)} "
              f"| test range {test['Date'].min().date()} -> {test['Date'].max().date()}")
