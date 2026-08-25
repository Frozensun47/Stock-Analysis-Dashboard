"""Long-short (market-neutral) experiment on the 15m walk-forward model.

At each rebalance bar (every REBAL bars), long the top-N predicted stocks and
short the bottom-N. Hold HORIZON bars. Cost per leg = COST (round trip).
Shorting NSE cash equity beyond intraday requires stock futures / SLB; we
assume futures-style shorting is available for the universe (approximation).
"""
import numpy as np, pandas as pd
import model_15m as M
from model_15m import FEATS, HORIZON, COST
from sklearn.ensemble import HistGradientBoostingRegressor

M.RETRAIN_EVERY = 1000
REBAL = 25
START_FRAC = 0.5


def collect_preds(tab, N):
    """Walk-forward, but return the full out-of-sample prediction table."""
    start = int(N * START_FRAC)
    out = []
    for s0 in range(start, N - 1, M.RETRAIN_EVERY):
        train = tab[(tab.i < s0 - HORIZON) & tab.fwd.notna()]
        if len(train) < 5000:
            continue
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                          learning_rate=0.06, random_state=0)
        m.fit(train[FEATS], train.fwd)
        test = tab[(tab.i >= s0) & (tab.i < min(s0 + M.RETRAIN_EVERY, N - 1))].copy()
        if test.empty:
            continue
        test["pred"] = m.predict(test[FEATS])
        out.append(test[["i", "j", "pred"]])
        print(f"  trained @bar {s0}/{N} ({len(train):,} rows)", flush=True)
    return pd.concat(out, ignore_index=True)


def trade_returns(preds, panel, n, N):
    """Return DataFrames of per-trade net returns for long and short legs,
    keyed by rebalance bar."""
    o, c = panel["Open"].values, panel["Close"].values
    grp = dict(tuple(preds.groupby("i")))
    longs, shorts = [], []
    start = int(N * START_FRAC)
    for i in range(start, N - 1, REBAL):
        g = grp.get(i)
        if g is None or len(g) < 2 * n:
            continue
        g = g.sort_values("pred")
        top, bot = g.tail(n), g.head(n)
        ex = min(i + HORIZON, N - 1)
        for _, r in top.iterrows():
            j = int(r.j); e = o[i + 1, j]; xp = c[ex, j]
            if np.isfinite(e) and e > 0 and np.isfinite(xp):
                longs.append((i, (xp / e - 1) * 100 - COST))
        for _, r in bot.iterrows():
            j = int(r.j); e = o[i + 1, j]; xp = c[ex, j]
            if np.isfinite(e) and e > 0 and np.isfinite(xp):
                shorts.append((i, -(xp / e - 1) * 100 - COST))
    return (pd.DataFrame(longs, columns=["i", "ret"]),
            pd.DataFrame(shorts, columns=["i", "ret"]))


def stats(name, per_trade, per_reb):
    """per_trade: Series of trade returns; per_reb: Series indexed by rebalance
    bar of the portfolio return per rebalance (equal-weight, notional 1/leg)."""
    eq = per_reb.cumsum()
    dd = (eq - eq.cummax()).min()
    # annualize Sharpe by rebalance frequency: 25 bars = 1 session, 250 sess/yr
    per_year = 250 * 25 / REBAL
    sh = per_reb.mean() / (per_reb.std() or 1) * np.sqrt(per_year)
    print(f"{name}: trades={len(per_trade):<5} avg/trade={per_trade.mean():+.4f}%  "
          f"win={(per_trade>0).mean()*100:4.1f}%  rebals={len(per_reb)}  "
          f"avg/rebal={per_reb.mean():+.4f}%  annSharpe={sh:+.2f}  "
          f"cum={eq.iloc[-1]:+7.1f}%  maxDD={dd:+.1f}%")


if __name__ == "__main__":
    from upstox_data import load_15m
    panel = load_15m()
    close = panel["Close"]
    N = len(close)
    print(f"panel: {N:,} bars x {close.shape[1]} stocks; REBAL={REBAL}, "
          f"HORIZON={HORIZON}, RETRAIN_EVERY={M.RETRAIN_EVERY}")
    F, fwd = M.build_features(panel)
    tab = M.flatten(F, fwd, close)
    preds = collect_preds(tab, N)
    print(f"OOS predictions: {len(preds):,} rows\n")

    for n in [3, 5, 10, 20]:
        L, S = trade_returns(preds, panel, n, N)
        lr = L.groupby("i").ret.mean()
        sr = S.groupby("i").ret.mean()
        both = pd.concat([lr.rename("L"), sr.rename("S")], axis=1).dropna()
        print(f"--- N={n} ---")
        stats("  long-only ", L.ret, lr)
        stats("  short-only", S.ret, sr)
        stats("  long-short", pd.concat([L.ret, S.ret]), (both.L + both.S) / 2)
        print()
