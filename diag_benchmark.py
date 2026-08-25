"""The benchmark every strategy must beat: doing nothing.

An Indian equity universe drifted upward through 2023-2026. Any long-only
strategy inherits that drift, so "positive average return" proves nothing. The
honest question is whether trading beats holding the same basket — after costs.
"""
import numpy as np, pandas as pd
import model_15m as M
from upstox_data import load_15m

panel = load_15m(); c = panel["Close"]
# Restrict to the out-of-sample half the walk-forward actually traded.
oos = c.iloc[int(len(c) * 0.5):]
r = oos.pct_change()
eq = (1 + r.mean(axis=1).fillna(0)).cumprod()      # equal-weight, no trading
sessions = len(oos.index.normalize().unique())
years = sessions / 250
tot = (eq.iloc[-1] - 1) * 100
print(f"OOS window: {oos.index[0]:%Y-%m-%d} → {oos.index[-1]:%Y-%m-%d} "
      f"({sessions} sessions, {years:.2f}y)")
print(f"\nBUY & HOLD equal-weight universe (zero trading after entry):")
print(f"  total {tot:+.1f}%   CAGR {((eq.iloc[-1])**(1/years)-1)*100:+.1f}%")
dd = (eq / eq.cummax() - 1).min() * 100
print(f"  max drawdown {dd:.1f}%")

print(f"\nWhat the 15m model must clear to be worth running:")
print(f"  cost per round trip        {M.COST}%")
print(f"  measured alpha, 8 sessions +0.121%   (excess over the universe)")
print(f"  measured alpha, 15 sessions +0.197%")
print(f"  → every trade pays {M.COST}% to harvest at most 0.20% of stock-selection edge.")
n_trades_to_beat = tot / M.COST
print(f"\n  A strategy churning through the same {years:.1f} years pays {M.COST}% per trade;")
print(f"  {n_trades_to_beat:.0f} round trips of pure cost would consume the entire "
      f"{tot:+.0f}% that holding delivered for free.")
