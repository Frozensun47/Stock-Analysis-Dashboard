"""Strategy library. Every function has the same signature so benchmark.run can
score them identically: f(data, i) -> list of tickers to hold as of bar i.

Only data up to and including bar i may be read. The harness enters at i+1's open.
"""
import numpy as np, pandas as pd
from functools import lru_cache

def _rsi(df, n=14):
    d = df.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

class Cache:
    """Indicators are computed once per dataset, then sliced — recomputing a
    rolling mean at every rebalance is what made the old backtests slow."""
    def __init__(self, data):
        c = data["Close"]
        self.c = c
        self.rsi = _rsi(c)
        self.sma20 = c.rolling(20).mean()
        self.sma50 = c.rolling(50).mean()
        self.sma200 = c.rolling(200).mean()
        self.ret5 = c.pct_change(5) * 100
        self.ret20 = c.pct_change(20) * 100
        self.ret60 = c.pct_change(60) * 100
        self.ret120 = c.pct_change(120) * 100
        self.vol20 = c.pct_change().rolling(20).std() * 100
        self.vol60 = c.pct_change().rolling(60).std() * 100
        self.hi252 = c.rolling(252).max()

_CACHE = {}
def cache(data):
    k = id(data["Close"])
    if k not in _CACHE:
        _CACHE[k] = Cache(data)
    return _CACHE[k]

def _top(series, n, mask=None):
    s = series.dropna()
    if mask is not None:
        s = s[mask.reindex(s.index).fillna(False)]
    return list(s.nlargest(n).index)


# ---------------- baselines ----------------
def random5(data, i, n=5, seed=0):
    """Control. If a strategy cannot beat this, it has no selection skill."""
    rng = np.random.default_rng(seed + i)
    avail = data["Close"].iloc[i].dropna().index
    return list(rng.choice(avail, size=min(n, len(avail)), replace=False))

def meanrev(data, i, n=5):
    """The repo's original idea: dip inside an uptrend, most oversold first."""
    k = cache(data)
    up = k.c.iloc[i] > k.sma50.iloc[i]
    r = k.rsi.iloc[i][up & (k.rsi.iloc[i] < 40)]
    return list(r.nsmallest(n).index)

def momentum(data, i, n=5):
    """12-1 style momentum: strongest 6-month gainers still in an uptrend."""
    k = cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    return _top(k.ret120.iloc[i], n, up)

def lowvol(data, i, n=5):
    """Low-volatility anomaly: the least volatile names in an uptrend."""
    k = cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    v = k.vol60.iloc[i][up.reindex(k.vol60.columns).fillna(False)]
    return list(v.dropna().nsmallest(n).index)

def breakout(data, i, n=5):
    """Names closest to a 52-week high — trend continuation."""
    k = cache(data)
    prox = (k.c.iloc[i] / k.hi252.iloc[i])
    up = k.c.iloc[i] > k.sma50.iloc[i]
    return _top(prox, n, up)

def mom_lowvol(data, i, n=5):
    """Momentum ranked, then risk-adjusted: return per unit of volatility."""
    k = cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    score = k.ret120.iloc[i] / k.vol60.iloc[i].replace(0, np.nan)
    return _top(score, n, up)

def momentum_60(data, i, n=5):
    """The leading hypothesis from TRAIN.

    3-month momentum, restricted to names in a long-term uptrend, held ~6 months
    with no stop. Every design choice here is a measured one, not a preference:

    - LOW TURNOVER is the whole game. The same signal at trail=4%/hold=20 pays
      17.5% of capital in costs over 3.3 years and loses to buy & hold; at
      trail=None/hold=120 it pays 1.7% and beats it. Cost is the dominant term.
    - NO TRAILING STOP. Swept at 2/3/4/6/8/10% on daily bars and on 15-minute
      bars: every width reduces net return. Stops buy tail protection, not return.
    - CONCENTRATED. Alpha falls monotonically as positions are added
      (+55% at 5, +17% at 10, -1% at 15, -19% at 20, -38% at 30) — the premium
      lives in the few strongest names and dilutes away.
    """
    k = cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    return _top(k.c.pct_change(60).iloc[i], n, up)


REGISTRY = {
    "momentum 3m >SMA200 (leading)": momentum_60,
    "random5 (control)": random5,
    "meanrev RSI<40 >SMA50": meanrev,
    "momentum 6m >SMA200": momentum,
    "lowvol >SMA200": lowvol,
    "breakout near 52w high": breakout,
    "momentum/vol >SMA200": mom_lowvol,
}


def momentum_broad(data, i, n=30):
    """Broad, low-turnover momentum tilt -- 12-1 lookback, ~30 names.

    Parameters come from published factor evidence (Fama-French 2017 EM momentum;
    Nifty200 Momentum 30, live since 2020), NOT from mining this repo's TRAIN
    split. 12-1 momentum skips the most recent month to avoid short-term
    reversal. Breadth is the point: random-5 books average -24.9% alpha here, so
    concentration is a structural penalty, not an edge.
    """
    k = cache(data)
    mom = (k.c.shift(21) / k.c.shift(252) - 1).iloc[i]   # 12-1, skip last month
    up = k.c.iloc[i] > k.sma200.iloc[i]
    return _top(mom, n, up)
