"""
ARTEMIS Mini - Backtest & Evaluation
Runs the trained PPO agent on held-out test data and compares it
against buy-and-hold and moving-average crossover baselines using
Sharpe ratio, cumulative return, and max drawdown.

Usage:
    python results/backtest.py --model results/ppo_agent.zip --test-data results/ppo_agent_test_data.csv
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from envs.trading_env import TradingEnv
from agents.baselines import buy_and_hold, moving_average_crossover


def compute_metrics(net_worth: np.ndarray, trading_days_per_year: int = 252) -> dict:
    net_worth = np.asarray(net_worth, dtype=float)
    returns = np.diff(net_worth) / net_worth[:-1]

    cumulative_return = (net_worth[-1] / net_worth[0]) - 1
    annualized_return = (1 + cumulative_return) ** (trading_days_per_year / len(net_worth)) - 1

    if returns.std() > 1e-9:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trading_days_per_year)
    else:
        sharpe = 0.0

    running_max = np.maximum.accumulate(net_worth)
    drawdown = (net_worth - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "cumulative_return_pct": cumulative_return * 100,
        "annualized_return_pct": annualized_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "final_net_worth": net_worth[-1],
    }


def run_ppo_agent(model_path: str, test_df: pd.DataFrame, initial_balance: float = 100_000.0):
    model = PPO.load(model_path)
    env = TradingEnv(test_df, initial_balance=initial_balance)
    obs, _ = env.reset()

    net_worth = [initial_balance]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        net_worth.append(info["net_worth"])
        done = terminated or truncated

    return np.array(net_worth), env.trade_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--initial-balance", type=float, default=100_000.0)
    parser.add_argument("--out", type=str, default=None,
                         help="Output CSV for the comparison table (default: results/backtest_comparison.csv)")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_data)

    print("Running PPO agent on held-out test data...")
    ppo_net_worth, trade_log = run_ppo_agent(args.model, test_df, args.initial_balance)

    print("Running baseline strategies...")
    bh_net_worth = buy_and_hold(test_df, args.initial_balance)
    mac_net_worth = moving_average_crossover(test_df, args.initial_balance)

    results = {
        "ARTEMIS (PPO)": compute_metrics(ppo_net_worth),
        "Buy & Hold": compute_metrics(bh_net_worth),
        "MA Crossover": compute_metrics(mac_net_worth),
    }

    print("\n" + "=" * 70)
    print(f"{'Strategy':<20}{'Return %':>12}{'Ann. Return %':>16}{'Sharpe':>10}{'Max DD %':>12}")
    print("=" * 70)
    for name, m in results.items():
        print(f"{name:<20}{m['cumulative_return_pct']:>12.2f}"
              f"{m['annualized_return_pct']:>16.2f}{m['sharpe_ratio']:>10.2f}"
              f"{m['max_drawdown_pct']:>12.2f}")
    print("=" * 70)
    print(f"\nPPO agent made {len(trade_log)} trades over {len(test_df)} trading days.")

    # Save results for the report
    out_path = args.out or "results/backtest_comparison.csv"
    out_df = pd.DataFrame(results).T
    out_df.to_csv(out_path)
    print(f"\nSaved comparison table to {out_path}")

    return results


if __name__ == "__main__":
    main()
