# ARTEMIS Mini Project

A scoped-down version of ARTEMIS (Adaptive Reinforcement Trading with
Multi-signal Intelligence System) — a PPO-based RL agent for NSE stock
trading, extendable with an LLM signal layer.

## Status

The full pipeline (data → environment → PPO training → backtest vs
baselines) is built and tested end-to-end on synthetic data. **Next
step: swap in real NSE data.**

## Project structure

```
artemis-mini/
├── data/
│   └── data_loader.py       # CSV loader + technical indicators (RSI, MACD, SMA, volatility)
├── envs/
│   └── trading_env.py       # Gymnasium environment (Discrete: Hold/Buy/Sell)
├── agents/
│   ├── train_ppo.py         # PPO training (Stable-Baselines3)
│   └── baselines.py         # Buy & hold, MA crossover strategies
├── results/
│   └── backtest.py          # Evaluation: Sharpe, cumulative return, max drawdown
└── requirements.txt
```

## Getting real data (do this on your own machine — this sandbox
## can't reach Yahoo Finance / NSE due to network restrictions)

```python
import yfinance as yf
df = yf.download("RELIANCE.NS", start="2020-01-01", end="2026-01-01")
df.reset_index(inplace=True)
df.to_csv("data/RELIANCE.csv", index=False)
```

Repeat for your other 3-4 chosen stocks (TCS.NS, INFY.NS, HDFCBANK.NS,
etc). Alternatively, pull the same data through your IndiaQuant MCP's
NSE data layer if you'd rather stay in one ecosystem for the report's
narrative.

Then upload the CSV(s) here and I'll run training on real data.

## Running the pipeline

```bash
pip install -r requirements.txt

# Train (use ~100k-200k timesteps for a real run, not the 30k smoke test)
python agents/train_ppo.py --csv data/RELIANCE.csv --timesteps 150000 --model-out results/ppo_reliance.zip

# Backtest against baselines
python results/backtest.py --model results/ppo_reliance.zip --test-data results/ppo_reliance_test_data.csv
```

## What's next (Week 3 of the plan)

The LLM augmentation layer isn't wired in yet. The environment already
has a hook for it: pass `llm_signal_col="llm_regime_signal"` to
`TradingEnv` and add that column to your dataframe (e.g., a
bullish/bearish/neutral score from Claude reading recent price action
or news) before training. This becomes part of the observation space
automatically.

## Notes on the reward function

Reward = % change in net worth per step, minus a small penalty
(`overtrade_penalty`) every time the agent trades, to discourage
noise-trading. Transaction cost is modeled at 15 bps (`transaction_cost_pct`)
to reflect realistic brokerage + slippage on NSE.
