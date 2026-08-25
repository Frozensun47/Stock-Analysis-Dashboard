"""Does the improved model beat buy-and-hold on the SAME out-of-sample window?

Reports every strategy as an equity curve on identical capital, not as a
per-trade average — a per-trade number hides how often you trade, and cost is
paid per trade. The benchmark is holding the equal-weight universe.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import model_15m as M
from upstox_data import load_15m

panel = load_15m(); c = panel["Close"]; N = len(c)

def run(H, top_k, min_pred, rebalance):
    """Walk-forward, then hold top_k names for H bars, rebalancing every `rebalance` bars."""
    M.HORIZON = H
    F, fwd = M.build_features(panel)
    tab = M.flatten(F, fwd, c)
    res = []
    for s0 in range(int(N * 0.5), N - 1, 1000):
        tr = tab[(tab.i < s0 - H) & tab.fwd.notna()]
        te = tab[(tab.i >= s0) & (tab.i < min(s0 + 1000, N - 1))].copy()
        if len(tr) < 20000 or te.empty:
            continue
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.06,
                                          random_state=0).fit(tr[M.FEATS], tr.fwd)
        te["pred"] = m.predict(te[M.FEATS]); res.append(te)
    te = pd.concat(res)
    ent = []
    for i, g in te[te.i % rebalance == 0].groupby("i"):
        for _, r in g.nlargest(top_k, "pred").query("pred > @min_pred").iterrows():
            ent.append((int(r.i), c.columns[int(r.j)]))
    s = M.simulate(ent, panel, trail=None, horizon=H)
    return s, len(ent)

oos = c.iloc[int(N * 0.5):]
years = len(oos.index.normalize().unique()) / 250
bh = (1 + oos.pct_change().mean(axis=1).fillna(0)).cumprod().iloc[-1]
print(f"OOS {oos.index[0]:%Y-%m-%d} → {oos.index[-1]:%Y-%m-%d} ({years:.2f}y)")
print(f"BUY & HOLD equal-weight: {(bh-1)*100:+.1f}% total, {(bh**(1/years)-1)*100:+.1f}% CAGR\n")
print(f"{'horizon':>8} {'rebal':>6} {'k':>3} {'trades':>7} {'net/trade':>10} "
      f"{'win%':>6} {'total%':>8} {'CAGR%':>7}")
for H, reb, k, mp in [(100, 25, 5, 1.0), (200, 50, 5, 1.0), (200, 50, 10, 0.5),
                      (375, 100, 5, 1.0), (375, 100, 10, 0.5)]:
    s, n = run(H, k, mp, reb)
    if s.empty:
        print(f"{H:>8} {reb:>6} {k:>3}      no trades"); continue
    # k concurrent positions, capital split k ways, compounded over the window
    total = (1 + s / 100 / k).prod()
    print(f"{H:>8} {reb:>6} {k:>3} {n:>7} {s.mean():>+10.4f} {(s>0).mean()*100:>6.1f} "
          f"{(total-1)*100:>+8.1f} {(total**(1/years)-1)*100:>+7.1f}")
