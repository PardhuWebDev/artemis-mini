"""
ARTEMIS Mini - PPO Training Script
Trains a PPO agent on the TradingEnv using Stable-Baselines3, on the
real NSE dataset (artemis_dataset.csv), one stock at a time.

Usage:
    python agents/train_ppo.py --ticker RELIANCE --timesteps 150000
    python agents/train_ppo.py --ticker TCS --timesteps 200000 --model-out results/ppo_tcs.zip
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from data.data_loader import load_and_prepare, VALID_TICKERS
from envs.trading_env import TradingEnv


def make_env(df, **env_kwargs):
    def _init():
        env = TradingEnv(df, **env_kwargs)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True, choices=VALID_TICKERS)
    parser.add_argument("--csv", type=str, default="data/artemis_dataset.csv")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--model-out", type=str, default=None)
    parser.add_argument("--no-llm-signal", action="store_true",
                         help="Ablation: train without the regime_code feature, for comparison.")
    args = parser.parse_args()

    model_out = args.model_out or f"results/ppo_{args.ticker.lower()}.zip"

    train_df, val_df, test_df = load_and_prepare(args.csv, args.ticker)
    print(f"[{args.ticker}] train={len(train_df)} rows "
          f"({train_df['Date'].min().date()} -> {train_df['Date'].max().date()})")
    print(f"[{args.ticker}] val={len(val_df)} rows "
          f"({val_df['Date'].min().date()} -> {val_df['Date'].max().date()})")
    print(f"[{args.ticker}] test={len(test_df)} rows "
          f"({test_df['Date'].min().date()} -> {test_df['Date'].max().date()})")

    llm_signal_col = None if args.no_llm_signal else "regime_code"
    env = DummyVecEnv([make_env(train_df, llm_signal_col=llm_signal_col)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # encourage exploration so it doesn't collapse to always-Hold
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)
    print(f"\nModel saved to {model_out}")

    # Save val/test splits alongside the model so backtest.py always
    # uses the exact same chronological holdout for this ticker/run.
    val_csv_path = model_out.replace(".zip", "_val_data.csv")
    test_csv_path = model_out.replace(".zip", "_test_data.csv")
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    print(f"Validation data saved to {val_csv_path}")
    print(f"Held-out test data saved to {test_csv_path}")


if __name__ == "__main__":
    main()
