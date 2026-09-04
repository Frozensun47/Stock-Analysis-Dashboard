"""Hypothesis: with 20 trades the momentum result is a coin flip, not a strategy.

Concentration is what makes it fragile — 5 positions held 120 days is ~5 bets per
year. A diversified low-turnover tilt should keep most of the factor premium with
far less variance, and should degrade gracefully rather than flipping sign.
"""
import numpy as np, pandas as pd
import benchmark as B, strategies as S

data = B.load("5y")

def mom_n(lookback=120):
    def f(data, i, n=5):
        k = S.cache(data)
        up = k.c.iloc[i] > k.sma200.iloc[i]
        return S._top(k.c.pct_change(lookback).iloc[i], n, up)
    return f

print("=== diversification sweep, low turnover (hold=120, trail=None, TRAIN) ===")
print(f"{'n_pos':>6} {'lookback':>9} {'trades':>7} {'cost%':>6} {'CAGR%':>7} "
      f"{'alpha%':>8} {'Sharpe':>7} {'maxDD%':>7}")
best = []
for n_pos in [5, 10, 15, 20, 30]:
    for lb in [60, 120, 250]:
        m, _ = B.run(mom_n(lb), data, B.TRAIN, n_pos=n_pos, max_hold=120, trail=None)
        best.append((n_pos, lb, m))
        print(f"{n_pos:>6} {lb:>9} {m['trades']:>7} {m['cost_drag']:>6.1f} "
              f"{m['cagr']:>+7.1f} {m['alpha_total_pct']:>+8.1f} {m['sharpe']:>7.2f} "
              f"{m['max_dd']:>7.1f}")

print("\n=== stability: how many (n_pos, lookback) cells beat the benchmark? ===")
win = [(n, l) for n, l, m in best if m["alpha_total_pct"] > 0]
print(f"{len(win)}/{len(best)} cells beat the benchmark")
for n_pos in [5, 10, 15, 20, 30]:
    cells = [m["alpha_total_pct"] for n, l, m in best if n == n_pos]
    print(f"  n_pos={n_pos:<3} alpha across lookbacks: "
          f"{' '.join(f'{a:+.0f}%' for a in cells)}   mean {np.mean(cells):+.1f}%")
for n_pos, lb, m in best:
    B.log(f"momentum{lb}d n_pos={n_pos} hold=120 trail=None", m,
          hypothesis="diversification keeps the premium with less variance",
          verdict="beats" if m["alpha_total_pct"] > 0 else "loses")
