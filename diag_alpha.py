"""Separate alpha from beta: is the top decile beating the MARKET, or just riding it?

Over 8 sessions the whole universe drifts up ~0.5%, so a top decile earning
+0.99% gross looks tradeable until you subtract what every stock earned that day.
Demeaning each bar's forward returns cross-sectionally removes the market move
and leaves only the part a stock-picking model could actually claim.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import model_15m as M
from upstox_data import load_15m

panel = load_15m(); c = panel["Close"]; N = len(c)
F, _ = M.build_features(panel)
print(f"{'H':>5} {'sess':>5} {'univ mean':>10} {'D9 raw':>8} {'D9 alpha':>9} "
      f"{'D0 alpha':>9} {'spread':>8} {'alpha t':>8}")
for H in [25, 100, 200, 375]:
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
    te = pd.concat(res)
    # alpha = the stock's forward return minus what the whole universe did that bar
    te["alpha"] = te.fwd - te.groupby("i").fwd.transform("mean")
    q = pd.qcut(te.pred, 10, labels=False, duplicates="drop")
    graw, ga = te.groupby(q).fwd.mean(), te.groupby(q).alpha.mean()
    top = te[q == q.max()].alpha
    t = top.mean() / top.std() * np.sqrt(len(top))
    print(f"{H:>5} {H/25:>5.1f} {te.groupby('i').fwd.mean().mean():>+10.4f} "
          f"{graw.iloc[-1]:>+8.4f} {ga.iloc[-1]:>+9.4f} {ga.iloc[0]:>+9.4f} "
          f"{ga.iloc[-1]-ga.iloc[0]:>+8.4f} {t:>8.2f}")
print(f"\nCost to beat: {M.COST}% round trip. Alpha is what survives after the market move.")
