"""ML stock ranker vs improved rule-based strategy — walk-forward, net of Groww costs.

Realism upgrades vs earlier tests:
- Entry at NEXT-DAY OPEN after the signal (you can't buy at the close you just observed).
- Nifty regime filter: only enter when Nifty > its 50d SMA.
- Costs: 0.348% round trip (Groww delivery, Rs50k).

ML: HistGradientBoosting regressor predicting 10-day forward return from
technical features, trained walk-forward (expanding window, retrained every
~3 months, always predicting strictly future data).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

COST = 0.348
data = pd.read_pickle("cache/prices_5y.pkl")
close, high, low, opn, vol = data["Close"], data["High"], data["Low"], data["Open"], data["Volume"]
nifty = pd.read_pickle("cache/nifty_5y.pkl")["Close"].iloc[:, 0].reindex(close.index).ffill()
N = len(close)

# ---------- vectorized features (panel: date x stock) ----------
d = close.diff()
gain = d.clip(lower=0).rolling(14).mean()
loss = (-d.clip(upper=0)).rolling(14).mean()
F = {
    "rsi":     100 - 100 / (1 + gain / loss.replace(0, np.nan)),
    "ret1":    close.pct_change(1) * 100,
    "ret5":    close.pct_change(5) * 100,
    "ret10":   close.pct_change(10) * 100,
    "ret20":   close.pct_change(20) * 100,
    "sma20":   (close / close.rolling(20).mean() - 1) * 100,
    "sma50":   (close / close.rolling(50).mean() - 1) * 100,
    "sma200":  (close / close.rolling(200).mean() - 1) * 100,
    "vol20":   (close.pct_change().rolling(20).std() * 100),
    "vsurge":  vol / vol.rolling(20).mean(),
    "hl":      ((high - low) / close).rolling(5).mean() * 100,
    "dist_hi": (close / close.rolling(252).max() - 1) * 100,
    "nifty_r5":  pd.DataFrame({t: nifty.pct_change(5) * 100 for t in close.columns}),
    "nifty_sma": pd.DataFrame({t: (nifty / nifty.rolling(50).mean() - 1) * 100 for t in close.columns}),
}
FWD = close.shift(-10) / close - 1  # 10d forward return (label)
FEATS = list(F.keys())

def flatten():
    """Long-format table: one row per (date_idx, stock)."""
    rows = []
    stack = {k: v.values for k, v in F.items()}
    y = FWD.values
    cl = close.values
    cols = list(close.columns)
    recs = []
    for i in range(210, N):
        for j, t in enumerate(cols):
            x = [stack[k][i, j] for k in FEATS]
            if not all(np.isfinite(x)) or not np.isfinite(cl[i, j]):
                continue
            recs.append((i, j, *x, y[i, j] if np.isfinite(y[i, j]) else np.nan))
    df = pd.DataFrame(recs, columns=["i", "j", *FEATS, "fwd"])
    return df

print("Building feature table…")
TAB = flatten()
print(f"{len(TAB):,} rows")

# ---------- shared trade simulator ----------
def simulate(entries):
    """entries: list of (signal_day_i, ticker). Entry at next-day open;
    exit: 4% trail on closes or 15-day time exit. Returns net % series."""
    rets = []
    for i, t in entries:
        if i + 1 >= N:
            continue
        e = opn[t].iloc[i + 1]
        if not np.isfinite(e):
            continue
        peak, ep = e, None
        for j in range(i + 1, min(i + 16, N)):
            c = close[t].iloc[j]
            if not np.isfinite(c):
                continue
            if c <= peak * 0.96:
                ep = c; break
            peak = max(peak, c)
        if ep is None:
            ep = close[t].iloc[min(i + 15, N - 1)]
        rets.append((ep / e - 1) * 100 - COST)
    return pd.Series(rets, dtype=float)

def report(name, s):
    if s.empty:
        print(f"{name}: no trades"); return
    print(f"{name}: n={len(s):<4} win={(s>0).mean()*100:5.1f}%  avg={s.mean():+.3f}%  "
          f"sum={s.sum():+8.1f}  worst={s.min():+.1f}")

regime_ok = (nifty > nifty.rolling(50).mean()).values

# ---------- rule-based (improved) ----------
sma50p = close.rolling(50).mean()
rsi_p = F["rsi"]
def rule_entries(use_regime):
    ent = []
    for i in range(210, N - 1):
        if use_regime and not regime_ok[i]:
            continue
        mask = (close.iloc[i] > sma50p.iloc[i]) & (rsi_p.iloc[i] < 40)
        cand = rsi_p.iloc[i][mask]
        if len(cand):
            ent.append((i, cand.idxmin()))
    return ent

# ---------- ML walk-forward ----------
def ml_entries(top_k=1, min_pred=0.0, use_regime=True):
    ent = []
    split0 = TAB[TAB.i < 700]  # ~first 2.8y initial train
    for start in range(700, N - 1, 63):  # retrain quarterly
        train = TAB[(TAB.i < start - 10) & TAB.fwd.notna()]  # 10d gap = no label leakage
        m = HistGradientBoostingRegressor(max_iter=200, max_depth=6, random_state=0)
        m.fit(train[FEATS], train.fwd)
        test = TAB[(TAB.i >= start) & (TAB.i < min(start + 63, N - 1))].copy()
        test["pred"] = m.predict(test[FEATS])
        for i, g in test.groupby("i"):
            if use_regime and not regime_ok[int(i)]:
                continue
            g = g.sort_values("pred", ascending=False).head(top_k)
            for _, r in g[g.pred > min_pred].iterrows():
                ent.append((int(r.i), close.columns[int(r.j)]))
    return ent

print("\n=== All results: entry at next-day open, 4% trail/15d exit, net of 0.348% costs ===")
rb = simulate(rule_entries(False));  report("Rules, no regime filter          ", rb)
rr = simulate(rule_entries(True));   report("Rules + Nifty>SMA50 regime filter", rr)
print("(training ML…)")
m1 = simulate(ml_entries(1, 0.0, True));  report("ML top-1/day, pred>0, regime     ", m1)
m2 = simulate(ml_entries(1, 0.02, True)); report("ML top-1/day, pred>2%, regime    ", m2)
m3 = simulate(ml_entries(1, 0.02, False));report("ML top-1/day, pred>2%, no regime ", m3)

# rule-based on the SAME period as ML (i>=700) for fair comparison
rr_late = simulate([(i, t) for i, t in rule_entries(True) if i >= 700])
report("Rules+regime, ML period only     ", rr_late)
