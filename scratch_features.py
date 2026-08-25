"""Feature-set experiment for model_15m: cross-sectional + long-lookback features.

Measures out-of-sample rank-IC (per-bar Spearman of pred vs fwd return),
decile spread, and net per-trade return for several feature sets.
Does not modify model_15m.py.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
import model_15m as M
from upstox_data import load_15m

RETRAIN_EVERY = 1000
HORIZON = M.HORIZON
WARMUP = 520


def build_all(panel):
    F, fwd = M.build_features(panel)
    c, h, l, o, v = (panel[k] for k in ["Close", "High", "Low", "Open", "Volume"])
    day = pd.Series(c.index.normalize(), index=c.index)

    # --- long-lookback / new raw features ---
    ret1 = c.pct_change(1)
    vol78 = ret1.rolling(78).std().replace(0, np.nan)
    for L in (26, 78, 200, 500):
        r = c.pct_change(L)
        F[f"relstr{L}"] = (r.sub(r.median(axis=1), axis=0)) * 100      # rel strength vs universe median
    F["ret200"] = c.pct_change(200) * 100
    F["ret500"] = c.pct_change(500) * 100
    F["sma500"] = (c / c.rolling(500).mean() - 1) * 100
    F["vnr78"] = c.pct_change(78) / (vol78 * np.sqrt(78))              # vol-normalised return
    F["vnr200"] = c.pct_change(200) / (ret1.rolling(200).std().replace(0, np.nan) * np.sqrt(200))
    F["dhi500"] = (c / h.rolling(500).max() - 1) * 100                 # dist from ~20-session high
    F["dlo500"] = (c / l.rolling(500).min() - 1) * 100
    # overnight gap: today's open vs prev session close, broadcast within day
    day_open = o.groupby(day.values).transform("first")
    prev_sess_close = c.shift(1).groupby(day.values).transform("first")  # last close of prev session
    F["gap"] = (day_open / prev_sess_close - 1) * 100

    raw_names = list(F.keys())
    # --- cross-sectional pct-ranks of everything (skip constants across stocks) ---
    skip = {"bar_of_day", "mkt_ret8"}
    cs_names = []
    for k in raw_names:
        if k in skip:
            continue
        F["cs_" + k] = F[k].rank(axis=1, pct=True)
        cs_names.append("cs_" + k)
    return F, fwd, raw_names, cs_names


def flatten(F, fwd, close, feats, warmup=WARMUP):
    X = np.stack([F[k].values for k in feats], axis=-1)
    y = fwd.values
    ok = np.isfinite(X).all(axis=-1) & np.isfinite(close.values)
    ok[:warmup] = False
    ii, jj = np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[1]), indexing="ij")
    tab = pd.DataFrame(X[ok], columns=feats)
    tab["i"], tab["j"], tab["fwd"] = ii[ok], jj[ok], y[ok]
    return tab


def walk_forward_preds(tab, feats, N, start_frac=0.5):
    start = int(N * start_frac)
    outs, model = [], None
    for s0 in range(start, N - 1, RETRAIN_EVERY):
        train = tab[(tab.i < s0 - HORIZON) & tab.fwd.notna()]
        if len(train) < 5000:
            continue
        model = HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                              learning_rate=0.06, random_state=0)
        model.fit(train[feats], train.fwd)
        test = tab[(tab.i >= s0) & (tab.i < min(s0 + RETRAIN_EVERY, N - 1))].copy()
        if test.empty:
            continue
        test["pred"] = model.predict(test[feats])
        outs.append(test[["i", "j", "fwd", "pred"]])
        print(f"    @bar {s0}/{N} rows={len(train):,}", flush=True)
    return pd.concat(outs), model


def evaluate(name, preds, close, panel, top_k=5, min_pred=1.0):
    ics, spreads = [], []
    for i, g in preds.groupby("i"):
        gg = g.dropna(subset=["fwd"])
        if len(gg) < 30:
            continue
        ics.append(spearmanr(gg.pred, gg.fwd).statistic)
        q = pd.qcut(gg.pred.rank(method="first"), 10, labels=False)
        spreads.append(gg.fwd[q == 9].mean() - gg.fwd[q == 0].mean())
    ics, spreads = np.array(ics), np.array(spreads)
    tstat = ics.mean() / ics.std() * np.sqrt(len(ics))
    # trades
    ent = []
    for i, g in preds.groupby("i"):
        g = g.nlargest(top_k, "pred")
        for _, r in g[g.pred > min_pred].iterrows():
            ent.append((int(r.i), close.columns[int(r.j)]))
    s = M.simulate(ent, panel)
    print(f"\n== {name} ==")
    print(f"  rank-IC mean={ics.mean():+.4f} t={tstat:+.2f} (n_bars={len(ics)})  "
          f"decile spread={spreads.mean():+.3f}%")
    M.report(f"  net top-{top_k} pred>{min_pred}%", s)
    return ics.mean(), tstat, spreads.mean(), (s.mean() if len(s) else np.nan)


if __name__ == "__main__":
    panel = load_15m()
    close = panel["Close"]
    N = len(close)
    print(f"panel {close.shape}", flush=True)
    F, fwd, raw_names, cs_names = build_all(panel)

    base = M.FEATS
    new_raw = [k for k in raw_names if k not in base]
    sets = {
        "A baseline (15 raw)": base,
        "B baseline + CS-ranks of baseline": base + ["cs_" + k for k in base if "cs_" + k in cs_names],
        "C CS-ranks only (all feats)": cs_names + ["bar_of_day", "mkt_ret8"],
        "D full: raw+new+all CS-ranks": raw_names + cs_names,
        "E new raw + CS-ranks (no base raw)": new_raw + cs_names + ["bar_of_day", "mkt_ret8"],
    }
    results = {}
    for name, feats in sets.items():
        feats = list(dict.fromkeys(feats))
        print(f"\n### {name} · {len(feats)} features", flush=True)
        tab = flatten(F, fwd, close, feats)
        preds, model = walk_forward_preds(tab, feats, N)
        results[name] = evaluate(name, preds, close, panel)
        if name.startswith("D"):
            imp = pd.Series(  # crude importance: permutation too slow; use split gain proxy
                getattr(model, "feature_importances_", np.zeros(len(feats))), index=feats)
    print("\n===== SUMMARY =====")
    for name, (ic, t, sp, net) in results.items():
        print(f"{name:40s} IC={ic:+.4f} t={t:+.2f} spread={sp:+.3f}% net/trade={net:+.4f}%")
