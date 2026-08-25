"""Verify the validated daily strategy with 15-minute execution (yfinance, ~60 days).

Signals stay DAILY (above SMA50, lowest RSI<40, 1/day). Execution goes intraday:
entry at next day's first 15m open, trailing stop evaluated on 15m closes.
Compares daily-bar vs 15m execution over the same window, and tests trail widths.
"""
import numpy as np, pandas as pd

COST = 0.348
daily = pd.read_pickle("cache/prices_5y.pkl")
dc = daily["Close"]
intra = pd.read_pickle("cache/prices_15m.pkl")
ic, io = intra["Close"], intra["Open"]
idates = ic.index.tz_localize(None).normalize()

# daily signals (same rule as strategy.py)
d = dc.diff()
rsi = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan))
sma50 = dc.rolling(50).mean()

first_intra_day = idates.min()
signals = []  # (signal_date, ticker)
for i in range(len(dc) - 1):
    if dc.index[i] < first_intra_day - pd.Timedelta(days=3):
        continue
    mask = (dc.iloc[i] > sma50.iloc[i]) & (rsi.iloc[i] < 40)
    cand = rsi.iloc[i][mask]
    if len(cand):
        signals.append((dc.index[i], cand.idxmin()))

def run_15m(trail_pct, max_days=15):
    rets, details = [], []
    for sd, t in signals:
        fut = ic[t][idates > sd].dropna()
        fop = io[t][idates > sd].dropna()
        if fut.empty or fop.empty:
            continue
        entry = fop.iloc[0]
        days = pd.Series(fut.index.tz_localize(None).normalize())
        uniq = days.unique()
        cutoff = uniq[min(max_days, len(uniq) - 1)]
        peak, ep = entry, None
        for ts, c in fut.items():
            if pd.Timestamp(ts).tz_localize(None).normalize() > cutoff:
                break
            if c <= peak * (1 - trail_pct / 100):
                ep = c
                break
            peak = max(peak, c)
        if ep is None:
            ep = fut[days.values <= cutoff].iloc[-1] if (days.values <= cutoff).any() else fut.iloc[-1]
        r = (ep / entry - 1) * 100 - COST
        rets.append(r)
        details.append((str(sd.date()), t.replace(".NS", ""), round(float(entry), 2), round(float(ep), 2), round(r, 2)))
    return pd.Series(rets), details

def run_daily(trail_pct=4.0, max_days=15):
    do, dcl = daily["Open"], dc
    rets = []
    for sd, t in signals:
        i = dcl.index.get_loc(sd)
        if i + 1 >= len(dcl):
            continue
        entry = do[t].iloc[i + 1]
        if not np.isfinite(entry):
            continue
        peak, ep = entry, None
        for j in range(i + 1, min(i + 1 + max_days, len(dcl))):
            c = dcl[t].iloc[j]
            if not np.isfinite(c):
                continue
            if c <= peak * (1 - trail_pct / 100):
                ep = c
                break
            peak = max(peak, c)
        if ep is None:
            ep = dcl[t].iloc[min(i + max_days, len(dcl) - 1)]
        rets.append((ep / entry - 1) * 100 - COST)
    return pd.Series(rets)

def rep(name, s):
    print(f"{name}: n={len(s):<3} win={(s>0).mean()*100:5.1f}%  avg={s.mean():+.3f}%  sum={s.sum():+7.1f}  worst={s.min():+.1f}")

print(f"{len(signals)} signals from {signals[0][0].date()} to {signals[-1][0].date()}\n")
rep("DAILY  execution, 4% trail   ", run_daily(4.0))
for tr in (4.0, 3.0, 2.0, 1.5):
    s, det = run_15m(tr)
    rep(f"15-MIN execution, {tr}% trail  ", s)

s, det = run_15m(4.0)
print("\nLast 10 trades (15m, 4% trail):")
print(pd.DataFrame(det[-10:], columns=["signal", "sym", "entry", "exit", "net %"]).to_string(index=False))
