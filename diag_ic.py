"""Does ANY tradeable signal exist? Rank-IC by horizon on the clean Upstox panel.

This is the question that decides whether more model work is worth doing. Trade
P&L is far too noisy to answer it — the rank information coefficient (Spearman
correlation of prediction vs realised forward return, computed per bar, then
averaged) is the standard test. It is measured strictly out-of-sample.

Benchmark: a rank-IC of +0.03 with a t-stat above 3 is a weak but real
cross-sectional equity signal. Below +0.01, nothing survives trading costs.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import model_15m as M
from upstox_data import load_15m

panel = load_15m()
c = panel["Close"]; N = len(c)
F, _ = M.build_features(panel)
print(f"panel {c.shape[0]:,} bars × {c.shape[1]} stocks\n")
print(f"{'horizon':>9} {'~sessions':>10} {'rank-IC':>9} {'t':>7} {'D9-D0 gross':>12} {'D9 gross':>9}")
for H in [25, 50, 100, 200, 375]:
    fwd = (c.shift(-H) / c - 1) * 100
    tab = M.flatten(F, fwd, c)
    res = []
    for s0 in range(int(N * 0.5), N - 1, 1000):
        tr = tab[(tab.i < s0 - H) & tab.fwd.notna()]
        te = tab[(tab.i >= s0) & (tab.i < min(s0 + 1000, N - 1)) & tab.fwd.notna()].copy()
        if len(tr) < 20000 or te.empty:
            continue
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.06,
                                          random_state=0).fit(tr[M.FEATS], tr.fwd)
        te["pred"] = m.predict(te[M.FEATS]); res.append(te)
    if not res:
        continue
    te = pd.concat(res)
    ic = te.groupby("i").apply(lambda g: g.pred.corr(g.fwd, method="spearman"),
                               include_groups=False).dropna()
    q = pd.qcut(te.pred, 10, labels=False, duplicates="drop")
    g = te.groupby(q).fwd.mean()
    t = ic.mean() / ic.std() * np.sqrt(len(ic))
    print(f"{H:>9} {H/25:>10.1f} {ic.mean():>+9.4f} {t:>7.2f} "
          f"{g.iloc[-1]-g.iloc[0]:>+12.4f} {g.iloc[-1]:>+9.4f}")
print(f"\nround-trip cost for reference: {M.COST}%")
