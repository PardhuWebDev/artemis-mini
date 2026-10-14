"""
ARTEMIS - Live Viva Demo Script (Fully Standalone & Path-Safe)
"""

import sys
import time
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from data.data_loader import load_and_prepare, VALID_TICKERS
from agents.baselines import buy_and_hold, moving_average_crossover
from results.backtest import compute_metrics, run_ppo_agent

BANNER = r"""
============================================================
   ARTEMIS - Adaptive Reinforcement Trading with
             Multi-signal Intelligence System
============================================================
"""

def line():
    print("-" * 62)

def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "HDFCBANK"

    print(BANNER)
    if ticker not in VALID_TICKERS:
        print(f"'{ticker}' is not valid. Choose from: {', '.join(VALID_TICKERS)}")
        sys.exit(1)

    # Locate model in results_real or results
    results_dir = ROOT_DIR / "results_real"
    if not results_dir.exists():
        results_dir = ROOT_DIR / "results"

    model_path = results_dir / f"ppo_{ticker.lower()}.zip"
    if not model_path.exists():
        model_path = ROOT_DIR / "results" / f"ppo_{ticker.lower()}.zip"

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    dataset_path = ROOT_DIR / "data" / "artemis_dataset.csv"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Stock selected : {ticker}")
    print(f"Model file     : {model_path.name}")
    print("Loading test dataset and PPO agent...")
    t0 = time.time()

    # Get chronological test split directly from the main dataset
    _, _, test_df = load_and_prepare(str(dataset_path), ticker)
    print(f"Held-out test period: {len(test_df)} trading days (2024-2026)")

    line()
    print("Running ARTEMIS (PPO agent) on test period...")
    ppo_nw, trade_log = run_ppo_agent(str(model_path), test_df)

    print("Running Buy & Hold baseline...")
    bh_nw = buy_and_hold(test_df)

    print("Running Moving-Average Crossover baseline...")
    mac_nw = moving_average_crossover(test_df)

    elapsed = time.time() - t0
    line()

    results = {
        "ARTEMIS (PPO)": compute_metrics(ppo_nw),
        "Buy & Hold": compute_metrics(bh_nw),
        "MA Crossover": compute_metrics(mac_nw),
    }

    print(f"RESULTS - {ticker} (Held-out test period)")
    line()
    print(f"{'Strategy':<18}{'Return %':>12}{'Ann. Return %':>16}{'Sharpe':>10}{'Max DD %':>12}")
    line()
    for name, m in results.items():
        print(f"{name:<18}{m['cumulative_return_pct']:>12.2f}"
              f"{m['annualized_return_pct']:>16.2f}{m['sharpe_ratio']:>10.2f}"
              f"{m['max_drawdown_pct']:>12.2f}")
    line()
    print(f"ARTEMIS made {len(trade_log)} trades over {len(test_df)} trading days.")
    print(f"Completed in {elapsed:.1f} seconds (model loaded, no retraining).")
    print(BANNER)

if __name__ == "__main__":
    main()