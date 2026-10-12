"""
ARTEMIS Mini - PPO Training Script
Trains a PPO agent on the TradingEnv using Stable-Baselines3.

Usage:
    python agents/train_ppo.py --csv data/synthetic_sample.csv --timesteps 100000
    python agents/train_ppo.py --csv data/RELIANCE.csv --timesteps 200000 --model-out results/ppo_reliance.zip
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from data.data_loader import load_and_prepare, make_synthetic_data, add_technical_indicators
from envs.trading_env import TradingEnv


def make_env(df, **env_kwargs):
    def _init():
        env = TradingEnv(df, **env_kwargs)
        return Monitor(env)
    return _init


def train_test_split(df, test_frac: float = 0.2):
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None,
                         help="Path to OHLCV CSV. If omitted, uses synthetic data.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-out", type=str, default="results/ppo_agent.zip")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    if args.csv:
        df = load_and_prepare(args.csv)
        print(f"Loaded {len(df)} rows from {args.csv}")
    else:
        print("No CSV provided - training on synthetic data (for pipeline testing only).")
        raw = make_synthetic_data()
        df = add_technical_indicators(raw)

    train_df, test_df = train_test_split(df, args.test_frac)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    env = DummyVecEnv([make_env(train_df)])

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

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    model.save(args.model_out)
    print(f"\nModel saved to {args.model_out}")

    # Save the test split alongside the model so backtest.py uses the exact same holdout
    test_csv_path = args.model_out.replace(".zip", "_test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Held-out test data saved to {test_csv_path}")


if __name__ == "__main__":
    main()
