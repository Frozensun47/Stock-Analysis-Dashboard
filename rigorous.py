"""Rigorous walk-forward test of the dip-buy strategy over 5 years, net of Groww costs.
Vectorized indicators; picks precomputed once per (rsi_max) setting."""
import numpy as np, pandas as pd

data = pd.read_pickle("cache/prices_5y.pkl")
close, high, opn = data["Close"], data["High"], data["Open"]
N = len(close)

# ---- costs (Groww delivery, Rs50k position) ----
def cost_pct(val=50_000):
    brok = min(20, val * 0.0005) * 2 * 1.18
    return (brok + val*0.001*2 + val*0.00015 + 13.5*1.18 + val*0.0000297*2*1.18 + val*0.000001*2) / val * 100
C = cost_pct()

# ---- vectorized indicators over the whole panel ----
d = close.diff()
up = d.clip(lower=0).rolling(14).mean()
dn = (-d.clip(upper=0)).rolling(14).mean()
RSI = 100 - 100 / (1 + up / dn.replace(0, np.nan))
SMA50 = close.rolling(50).mean()
SMA200 = close.rolling(200).mean()

def make_picks(rsi_max, need_sma200=False):
    """For each day, the eligible stock with lowest RSI (uptrend + oversold)."""
    mask = (close > SMA50) & (RSI < rsi_max)
    if need_sma200:
        mask &= close > SMA200
    r = RSI.where(mask)
    any_ok = r.notna().any(axis=1)
    out = pd.Series(np.nan, index=r.index, dtype=object)
    out[any_ok] = r[any_ok].idxmin(axis=1)
    return out.reset_index(drop=True)

def bt(picks, tp, trail, max_hold, i0, i1):
    """Simulate on day range [i0, i1). tp=None disables take-profit. Returns Series of net %."""
    rets = []
    for i in range(i0, min(i1, N - 1)):
        t = picks.iloc[i]
        if not isinstance(t, str):
            continue
        entry = close[t].iloc[i]
        if not np.isfinite(entry):
            continue
        peak, ep = entry, None
        for j in range(i + 1, min(i + 1 + max_hold, N)):
            h, c = high[t].iloc[j], close[t].iloc[j]
            if not np.isfinite(c):
                continue
            if tp and h >= entry * (1 + tp / 100):
                ep = entry * (1 + tp / 100); break
            if c <= peak * (1 - trail / 100):
                ep = c; break
            peak = max(peak, c)
        if ep is None:
            ep = close[t].iloc[min(i + max_hold, N - 1)]
        rets.append((ep / entry - 1) * 100 - C)
    return pd.Series(rets)

def stats(s):
    if s.empty: return "no trades"
    ann = s.sum() / 5  # rough: sum of per-trade % per year (1 position at a time approx)
    return f"n={len(s):<4} win={(s>0).mean()*100:5.1f}% avg={s.mean():+.3f}% sum={s.sum():+7.1f} maxDD_trade={s.min():+.1f}"

print(f"Round-trip cost: {C:.3f}%  |  {N} days, walk-forward split at day {N//2} ({close.index[N//2].date()})\n")

GRID = [(tp, tr, mh, rm) for tp in (3.0, 4.0, 5.0, 6.0, None)
        for tr in (2.0, 3.0, 4.0)
        for mh in (15, 30)
        for rm in (35, 40)]

split = N // 2
results = []
picks_cache = {}
for tp, tr, mh, rm in GRID:
    if rm not in picks_cache:
        picks_cache[rm] = make_picks(rm)
    p = picks_cache[rm]
    tr_ret = bt(p, tp, tr, mh, 210, split)
    results.append((tp, tr, mh, rm, tr_ret.sum(), tr_ret))
results.sort(key=lambda x: -x[4])

print("TOP 8 on TRAIN (2021-2024):")
print("tp    trail hold rsi  | train                                        | TEST (unseen 2024-2026)")
for tp, tr, mh, rm, _, tr_ret in results[:8]:
    te_ret = bt(picks_cache[rm], tp, tr, mh, split, N - 1)
    print(f"{str(tp):<5} {tr:<5} {mh:<4} {rm:<4} | {stats(tr_ret)} | {stats(te_ret)}")

# robustness of the best config: year-by-year
tp, tr, mh, rm = results[0][:4]
print(f"\nBest config (tp={tp}, trail={tr}, hold={mh}, rsi<{rm}) year by year:")
p = picks_cache[rm]
years = pd.Series(close.index.year, index=range(N))
for y in sorted(set(close.index.year))[1:]:
    idx = [i for i in range(210, N-1) if close.index[i].year == y]
    if idx:
        s = bt(p, tp, tr, mh, idx[0], idx[-1]+1)
        print(f"  {y}: {stats(s)}")
