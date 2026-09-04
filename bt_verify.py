"""Verify bt_lib.score_panel reproduces engine.scan's Buy %."""
import numpy as np, pandas as pd
from bt_lib import load_data, score_panel, groww_cost, cost_pct
from engine import scan

data = load_data()
sp = score_panel(data)
print("panel", sp.shape, sp.index[0].date(), sp.index[-1].date())

rng = np.random.default_rng(0)
idxs = rng.choice(range(300, len(data) - 1), 6, replace=False)
worst = 0.0
for i in idxs:
    ref = scan(data, asof=int(i)).set_index("Symbol")["Buy %"]
    mine = sp.iloc[i].dropna()
    mine.index = [c.replace(".NS", "") for c in mine.index]
    common = ref.index.intersection(mine.index)
    diff = (ref[common] - mine[common]).abs()
    print(f"{data.index[i].date()} n_ref={len(ref)} n_mine={len(mine)} common={len(common)} maxdiff={diff.max():.3f} n>0.11={int((diff>0.11).sum())}")
    worst = max(worst, diff.max())
print("worst abs diff:", round(worst, 3))

print("\n-- cost table (round trip, one scrip) --")
for tk in [1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000]:
    print(f"ticket Rs {tk:>7,}  cost Rs {groww_cost(tk, tk):7.2f}  = {cost_pct(tk):5.3f}%")
