"""Is the momentum result real, or one lucky cell?

A finding that only exists at one parameter setting is noise. A real factor
premium should survive changes to concentration, rebalance cadence, lookback and
holding period, and should not come from a single year.
"""
import numpy as np, pandas as pd
import benchmark as B, strategies as S

data = B.load("5y")
K = S.cache(data)

def mom(lookback):
    def f(data, i, n=5):
        k = S.cache(data)
        r = k.c.pct_change(lookback).iloc[i]
        up = k.c.iloc[i] > k.sma200.iloc[i]
        return S._top(r, n, up)
    return f

print("=== A. does it survive different concentration and cadence? (TRAIN) ===")
print(f"{'n_pos':>6} {'rebal':>6} {'hold':>5} {'trades':>7} {'CAGR%':>7} {'alpha%':>8} {'Sharpe':>7} {'maxDD%':>7}")
for n_pos in [3, 5, 10]:
    for reb in [5, 21]:
        m, _ = B.run(S.REGISTRY["momentum 6m >SMA200"], data, B.TRAIN,
                     n_pos=n_pos, rebalance=reb, max_hold=120, trail=None)
        print(f"{n_pos:>6} {reb:>6} {120:>5} {m['trades']:>7} {m['cagr']:>+7.1f} "
              f"{m['alpha_total_pct']:>+8.1f} {m['sharpe']:>7.2f} {m['max_dd']:>7.1f}")

print("\n=== B. does it survive a different momentum lookback? (TRAIN) ===")
print(f"{'lookback':>9} {'hold':>5} {'trades':>7} {'CAGR%':>7} {'alpha%':>8} {'Sharpe':>7}")
for lb in [60, 120, 180, 250]:
    for hold in [60, 120, 250]:
        m, _ = B.run(mom(lb), data, B.TRAIN, max_hold=hold, trail=None)
        print(f"{lb:>9} {hold:>5} {m['trades']:>7} {m['cagr']:>+7.1f} "
              f"{m['alpha_total_pct']:>+8.1f} {m['sharpe']:>7.2f}")

print("\n=== C. is the alpha spread across years or one lucky year? (TRAIN) ===")
print(f"{'year':>6} {'trades':>7} {'strat%':>8} {'bench%':>8} {'alpha%':>8}")
for y in [2022, 2023, 2024]:
    try:
        m, _ = B.run(S.REGISTRY["momentum 6m >SMA200"], data, (f"{y}-01-01", f"{y}-12-31"),
                     max_hold=120, trail=None)
        print(f"{y:>6} {m['trades']:>7} {m['total_pct']:>+8.1f} "
              f"{m['bench_total_pct']:>+8.1f} {m['alpha_total_pct']:>+8.1f}")
    except Exception as e:
        print(f"{y:>6}  {e}")
