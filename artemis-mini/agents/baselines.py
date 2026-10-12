"""
ARTEMIS Mini - Baseline Strategies
Simple baselines that the PPO agent must beat: buy-and-hold and
a moving-average crossover strategy. Both return a net_worth series
aligned with the trading environment's step indexing for fair comparison.
"""

import numpy as np


def buy_and_hold(df, initial_balance: float = 100_000.0, transaction_cost_pct: float = 0.0015):
    """Buy on day 0 with all capital, hold until the end."""
    prices = df["close"].values
    entry_price = prices[0]
    shares = int(initial_balance // (entry_price * (1 + transaction_cost_pct)))
    cash = initial_balance - shares * entry_price * (1 + transaction_cost_pct)

    net_worth = cash + shares * prices
    return net_worth


def moving_average_crossover(
    df,
    initial_balance: float = 100_000.0,
    transaction_cost_pct: float = 0.0015,
    fast_col: str = "sma_20",
    slow_col: str = "sma_50",
):
    """
    Go long when fast MA crosses above slow MA, exit to cash when it
    crosses below. Requires sma_10/sma_50 columns (already produced by
    data_loader.add_technical_indicators).
    """
    prices = df["close"].values
    fast = df[fast_col].values
    slow = df[slow_col].values

    cash = initial_balance
    shares = 0
    net_worth = []

    for i in range(len(df)):
        price = prices[i]

        if fast[i] > slow[i] and shares == 0 and cash > price:
            shares = int(cash // (price * (1 + transaction_cost_pct)))
            cash -= shares * price * (1 + transaction_cost_pct)
        elif fast[i] < slow[i] and shares > 0:
            cash += shares * price * (1 - transaction_cost_pct)
            shares = 0

        net_worth.append(cash + shares * price)

    return np.array(net_worth)
