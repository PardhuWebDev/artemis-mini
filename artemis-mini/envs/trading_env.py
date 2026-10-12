"""
ARTEMIS Mini - Trading Environment
A Gymnasium-compatible environment for training a PPO agent on
single-stock discrete trading decisions (Buy / Hold / Sell).

Observation space: technical indicators + portfolio state
Action space: Discrete(3) -> 0 = Hold, 1 = Buy, 2 = Sell
Reward: change in portfolio value, with a small penalty for overtrading
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Matches the real artemis_dataset.csv schema (data_loader.py renames
    # Open/High/Low/Close/Volume -> lowercase and ret_1d -> returns, but
    # the indicator columns below are already lowercase in the dataset).
    FEATURE_COLUMNS = [
        "price_to_sma50", "macd", "macd_signal", "macd_hist",
        "rsi_14", "volatility_20", "returns", "vol_ratio",
    ]

    def __init__(
        self,
        df,
        initial_balance: float = 100_000.0,
        transaction_cost_pct: float = 0.0015,  # ~15 bps, brokerage + slippage
        overtrade_penalty: float = 0.0005,
        window_size: int = 1,
        llm_signal_col: str | None = "regime_code",  # ARTEMIS's LLM-derived regime signal
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.overtrade_penalty = overtrade_penalty
        self.window_size = window_size
        self.llm_signal_col = llm_signal_col

        feature_dim = len(self.FEATURE_COLUMNS)
        if llm_signal_col and llm_signal_col in self.df.columns:
            feature_dim += 1

        # portfolio state features: cash_ratio, position_ratio, unrealized_pnl_pct
        obs_dim = feature_dim + 3

        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._max_steps = len(self.df) - 1
        self.reset()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        features = [row[col] for col in self.FEATURE_COLUMNS]
        if self.llm_signal_col and self.llm_signal_col in self.df.columns:
            features.append(row[self.llm_signal_col])

        portfolio_value = self.cash + self.shares_held * row["close"]
        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 0
        position_ratio = (self.shares_held * row["close"]) / portfolio_value if portfolio_value > 0 else 0
        unrealized_pnl_pct = (
            (row["close"] - self.avg_buy_price) / self.avg_buy_price
            if self.shares_held > 0 and self.avg_buy_price > 0 else 0.0
        )

        obs = np.array(features + [cash_ratio, position_ratio, unrealized_pnl_pct], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_balance
        self.shares_held = 0
        self.avg_buy_price = 0.0
        self.net_worth_history = [self.initial_balance]
        self.trade_log = []
        return self._get_obs(), {}

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row["close"]
        prev_net_worth = self.cash + self.shares_held * price

        traded = False
        if action == 1:  # Buy - deploy 100% of available cash
            if self.cash > price:  # can afford at least 1 share
                shares_to_buy = int(self.cash // (price * (1 + self.transaction_cost_pct)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.transaction_cost_pct)
                    total_shares = self.shares_held + shares_to_buy
                    self.avg_buy_price = (
                        (self.avg_buy_price * self.shares_held + price * shares_to_buy) / total_shares
                    )
                    self.cash -= cost
                    self.shares_held = total_shares
                    traded = True
                    self.trade_log.append((self.current_step, "BUY", shares_to_buy, price))

        elif action == 2:  # Sell - liquidate full position
            if self.shares_held > 0:
                proceeds = self.shares_held * price * (1 - self.transaction_cost_pct)
                self.cash += proceeds
                self.trade_log.append((self.current_step, "SELL", self.shares_held, price))
                self.shares_held = 0
                self.avg_buy_price = 0.0
                traded = True

        self.current_step += 1
        terminated = self.current_step >= self._max_steps
        truncated = False

        next_row = self.df.iloc[min(self.current_step, self._max_steps)]
        new_net_worth = self.cash + self.shares_held * next_row["close"]
        self.net_worth_history.append(new_net_worth)

        # Reward: pct change in net worth, minus a small friction penalty on every trade
        reward = (new_net_worth - prev_net_worth) / prev_net_worth
        if traded:
            reward -= self.overtrade_penalty

        info = {
            "net_worth": new_net_worth,
            "cash": self.cash,
            "shares_held": self.shares_held,
            "step": self.current_step,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        step = min(self.current_step, self._max_steps)
        row = self.df.iloc[step]
        net_worth = self.cash + self.shares_held * row["close"]
        print(f"Step {step} | Price {row['close']:.2f} | Cash {self.cash:.2f} | "
              f"Shares {self.shares_held} | Net worth {net_worth:.2f}")
