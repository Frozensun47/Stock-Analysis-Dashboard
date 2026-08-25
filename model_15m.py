"""15-minute intraday ML model, walk-forward validated, net of costs.

Signal: for each 15m bar, rank stocks by predicted forward return over the next
HORIZON bars using intraday technical features. Trained walk-forward (expanding
window, retrained monthly, always predicting strictly-future bars) so reported
numbers are out-of-sample.

Execution realism:
- Entry at the OPEN of the bar AFTER the signal bar (never the observed close).
- Exit: trailing stop on 15m closes, or time exit at HORIZON bars, and every
  position is force-closed at the last bar of the session (no overnight gap risk).
- Costs: COST_BPS round trip (intraday brokerage + STT + slippage).
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import dump, load

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
MODEL_PATH = os.path.join(CACHE, "model_15m.joblib")

COST = 0.15        # % round trip, intraday
HORIZON = 8        # bars (~2 hours)
TRAIL = 0.8        # % trailing stop on 15m closes
RETRAIN_EVERY = 500  # bars (~1 month of 25 bars/day)

FEATS = ["ret1", "ret4", "ret8", "ret26", "rsi14", "vwapdev", "sma26", "sma78",
         "atr", "vsurge", "hl", "bar_of_day", "day_ret", "range_pos", "mkt_ret8"]


def build_features(panel):
    """panel: dict of wide DataFrames. Returns (F dict, fwd label, close/open frames)."""
    c, h, l, o, v = (panel[k] for k in ["Close", "High", "Low", "Open", "Volume"])
    day = pd.Series(c.index.normalize(), index=c.index)
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    tp = (h + l + c) / 3
    cum_pv = (tp * v).groupby(day.values).cumsum()
    cum_v = v.groupby(day.values).cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v
    day_open = c.groupby(day.values).transform("first")
    hi_d = h.groupby(day.values).cummax()
    lo_d = l.groupby(day.values).cummin()
    bar_idx = pd.Series(range(len(c)), index=c.index).groupby(day.values).cumcount()
    mkt = c.pct_change(8).mean(axis=1) * 100
    F = {
        "ret1": c.pct_change(1) * 100,
        "ret4": c.pct_change(4) * 100,
        "ret8": c.pct_change(8) * 100,
        "ret26": c.pct_change(26) * 100,
        "rsi14": 100 - 100 / (1 + up / dn.replace(0, np.nan)),
        "vwapdev": (c / vwap - 1) * 100,
        "sma26": (c / c.rolling(26).mean() - 1) * 100,
        "sma78": (c / c.rolling(78).mean() - 1) * 100,
        "atr": ((h - l) / c).rolling(14).mean() * 100,
        "vsurge": v / v.rolling(26).mean().replace(0, np.nan),
        "hl": ((h - l) / c) * 100,
        "bar_of_day": pd.DataFrame({t: bar_idx for t in c.columns}, index=c.index),
        "day_ret": (c / day_open - 1) * 100,
        "range_pos": (c - lo_d) / (hi_d - lo_d).replace(0, np.nan),
        "mkt_ret8": pd.DataFrame({t: mkt for t in c.columns}, index=c.index),
    }
    fwd = (c.shift(-HORIZON) / c - 1) * 100
    # no label across a session boundary
    same_day = pd.DataFrame({t: day.shift(-HORIZON).values == day.values for t in c.columns}, index=c.index)
    fwd = fwd.where(same_day)
    return F, fwd


def flatten(F, fwd, close, warmup=80):
    """Long table: one row per (bar_index, stock)."""
    X = np.stack([F[k].values for k in FEATS], axis=-1)   # (T, S, K)
    y = fwd.values
    cl = close.values
    T, S, K = X.shape
    ii, jj = np.meshgrid(np.arange(T), np.arange(S), indexing="ij")
    ok = np.isfinite(X).all(axis=-1) & np.isfinite(cl)
    ok[:warmup] = False
    tab = pd.DataFrame(X[ok], columns=FEATS)
    tab["i"], tab["j"], tab["fwd"] = ii[ok], jj[ok], y[ok]
    return tab


def simulate(entries, panel, trail=TRAIL, horizon=HORIZON):
    """entries: (signal_bar_i, ticker). Enter next bar's open, trail-stop on
    closes, exit by horizon or at the session's last bar, whichever comes first."""
    c, o = panel["Close"], panel["Open"]
    day = c.index.normalize()
    last_bar_of_day = pd.Series(day).groupby(day).transform(lambda s: s.index[-1]).values
    cv, ov, N = c.values, o.values, len(c)
    cols = {t: k for k, t in enumerate(c.columns)}
    rets = []
    for i, t in entries:
        j = cols[t]
        if i + 1 >= N:
            continue
        e = ov[i + 1, j]
        if not np.isfinite(e) or e <= 0:
            continue
        peak, ep = e, None
        stop_at = min(i + horizon, int(last_bar_of_day[i]), N - 1)
        for k in range(i + 1, stop_at + 1):
            px = cv[k, j]
            if not np.isfinite(px):
                continue
            if px <= peak * (1 - trail / 100):
                ep = px; break
            peak = max(peak, px)
        if ep is None:
            ep = cv[stop_at, j]
            if not np.isfinite(ep):
                continue
        rets.append((ep / e - 1) * 100 - COST)
    return pd.Series(rets, dtype=float)


def report(name, s):
    if s.empty:
        print(f"{name}: no trades"); return s
    print(f"{name}: n={len(s):<5} win={(s>0).mean()*100:5.1f}%  avg={s.mean():+.4f}%  "
          f"sum={s.sum():+8.1f}  worst={s.min():+.2f}  sharpe={s.mean()/(s.std() or 1)*np.sqrt(len(s)):.2f}")
    return s


def walk_forward(tab, close, top_k=1, min_pred=0.0, start_frac=0.5):
    """Expanding-window retrain; returns entries plus the final fitted model."""
    N = len(close)
    start = int(N * start_frac)
    ent, model = [], None
    for s0 in range(start, N - 1, RETRAIN_EVERY):
        train = tab[(tab.i < s0 - HORIZON) & tab.fwd.notna()]
        if len(train) < 5000:
            continue
        model = HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                              learning_rate=0.06, random_state=0)
        model.fit(train[FEATS], train.fwd)
        test = tab[(tab.i >= s0) & (tab.i < min(s0 + RETRAIN_EVERY, N - 1))].copy()
        if test.empty:
            continue
        test["pred"] = model.predict(test[FEATS])
        for i, g in test.groupby("i"):
            g = g.nlargest(top_k, "pred")
            for _, r in g[g.pred > min_pred].iterrows():
                ent.append((int(r.i), close.columns[int(r.j)]))
        print(f"  trained @bar {s0}/{N} · {len(train):,} rows · {len(ent)} signals so far", flush=True)
    return ent, model


def train_and_save(panel):
    """Fit on ALL labelled history and persist for live inference."""
    F, fwd = build_features(panel)
    tab = flatten(F, fwd, panel["Close"])
    train = tab[tab.fwd.notna()]
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.06, random_state=0)
    m.fit(train[FEATS], train.fwd)
    dump({"model": m, "feats": FEATS, "horizon": HORIZON, "trail": TRAIL,
          "trained_through": str(panel["Close"].index[-1])}, MODEL_PATH)
    print(f"saved {MODEL_PATH} ({len(train):,} training rows)")
    return m


def live_signals(panel, top_k=5):
    """Rank the universe on the most recent 15m bar with the saved model."""
    b = load(MODEL_PATH)
    F, fwd = build_features(panel)
    close = panel["Close"]
    row = {k: F[k].iloc[-1] for k in b["feats"]}
    X = pd.DataFrame(row).dropna()
    if X.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"Symbol": [t.replace(".NS", "") for t in X.index],
                        "pred_%": b["model"].predict(X[b["feats"]]),
                        "price": close.iloc[-1].reindex(X.index).values})
    return out.sort_values("pred_%", ascending=False).head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    from upstox_data import load_15m
    panel = load_15m()
    close = panel["Close"]
    print(f"panel: {close.shape[0]:,} bars × {close.shape[1]} stocks "
          f"({close.index[0]} → {close.index[-1]})")
    F, fwd = build_features(panel)
    tab = flatten(F, fwd, close)
    print(f"feature table: {len(tab):,} rows")

    print(f"\n=== Walk-forward, 15m execution, {COST}% round-trip cost, {TRAIL}% trail, {HORIZON}-bar horizon ===")
    # baseline: intraday mean-reversion rule (RSI dip within uptrend)
    rsi_, sma = F["rsi14"], F["sma78"]
    N = len(close); start = int(N * 0.5)
    rule = []
    for i in range(start, N - 1):
        mask = (sma.iloc[i] > 0) & (rsi_.iloc[i] < 35)
        cand = rsi_.iloc[i][mask]
        if len(cand):
            rule.append((i, cand.idxmin()))
    report("Rule: RSI<35 above SMA78 ", simulate(rule, panel))

    for k, mp in [(1, 0.0), (1, 0.1), (3, 0.1)]:
        ent, _ = walk_forward(tab, close, top_k=k, min_pred=mp)
        report(f"ML top-{k}/bar pred>{mp}%   ", simulate(ent, panel))

    train_and_save(panel)
    print("\nLive top picks:\n", live_signals(panel).to_string(index=False))
