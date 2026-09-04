"""Exploration strategies: low-turnover SELECTION signals.

Same contract as strategies.py:  f(data, i) -> list[str] of tickers to hold as
of bar i, reading only data up to and including i.

Designed to be run with trail=None and max_hold in {60,120,250}, because
exp_exits.py established that turnover, not signal, is the binding cost.
"""
import numpy as np, pandas as pd
import strategies as S

# ---------------- extra indicator cache ----------------
class XCache:
    def __init__(self, data):
        c = data["Close"]
        self.c = c
        r = c.pct_change()
        self.r = r
        self.ret252 = c.pct_change(252) * 100
        self.ret126 = c.pct_change(126) * 100
        self.ret21 = c.pct_change(21) * 100
        # 12-1 momentum: 12m return skipping the most recent month
        self.mom12_1 = (c.shift(21) / c.shift(252) - 1) * 100
        self.vol120 = r.rolling(120).std() * 100
        self.dd252 = c / c.rolling(252).max() - 1          # distance from 52w high
        # equal-weight universe index and its own 200d MA (regime)
        ew = (1 + r.mean(axis=1).fillna(0)).cumprod()
        self.ew = ew
        self.ew_sma200 = ew.rolling(200).mean()
        self.breadth = (c > c.rolling(200).mean()).mean(axis=1)

_X = {}
def xcache(data):
    k = id(data["Close"])
    if k not in _X:
        _X[k] = XCache(data)
    return _X[k]


def _rank(s):
    """Cross-sectional percentile rank, NaN-safe."""
    return s.rank(pct=True)


# ---------------- single factors, low turnover ----------------
def mom12_1(data, i, n=5):
    """Classic academic momentum: 12-month return skipping the last month,
    filtered to names in an uptrend."""
    k, x = S.cache(data), xcache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    return S._top(x.mom12_1.iloc[i], n, up)


def lowvol_long(data, i, n=5):
    """Low-volatility anomaly measured over 120d instead of 60d."""
    x = xcache(data)
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    v = x.vol120.iloc[i][up.reindex(x.vol120.columns).fillna(False)]
    return list(v.dropna().nsmallest(n).index)


# ---------------- multi-factor composites ----------------
def _composite(data, i, weights):
    """Weighted sum of cross-sectional percentile ranks -> a score Series."""
    k, x = S.cache(data), xcache(data)
    parts = {
        "mom6": k.ret120.iloc[i],
        "mom12_1": x.mom12_1.iloc[i],
        "lowvol": -x.vol120.iloc[i],
        "trend": k.c.iloc[i] / k.sma200.iloc[i],
        "nearhigh": x.dd252.iloc[i],
        "revers1m": -x.ret21.iloc[i],
    }
    score = None
    for key, w in weights.items():
        r = _rank(parts[key]) * w
        score = r if score is None else score.add(r, fill_value=np.nan)
    return score


def mf_mom_lowvol_trend(data, i, n=5):
    """Rank-combine 6m momentum + low vol + trend strength. Equal weights."""
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    score = _composite(data, i, {"mom6": 1.0, "lowvol": 1.0, "trend": 1.0})
    return S._top(score, n, up)


def mf_mom_lowvol(data, i, n=5):
    """Two-factor: momentum + low vol, no trend term."""
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    score = _composite(data, i, {"mom12_1": 1.0, "lowvol": 1.0})
    return S._top(score, n, up)


def mf_quadrant(data, i, n=5):
    """Four-factor: 12-1 momentum, low vol, near-52w-high, 1m reversal."""
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    score = _composite(data, i, {"mom12_1": 1.0, "lowvol": 1.0,
                                    "nearhigh": 1.0, "revers1m": 0.5})
    return S._top(score, n, up)


# ---------------- regime filters ----------------
def _regime_on(data, i, kind="index"):
    x = xcache(data)
    if kind == "index":
        a, b = x.ew.iloc[i], x.ew_sma200.iloc[i]
        return bool(np.isfinite(a) and np.isfinite(b) and a > b)
    if kind == "breadth":
        return bool(x.breadth.iloc[i] > 0.5)
    return True


def _wrap_regime(fn, kind="index"):
    def g(data, i, n=5):
        if not _regime_on(data, i, kind):
            return []
        return fn(data, i, n)
    g.__name__ = f"{fn.__name__}_regime_{kind}"
    g.__doc__ = f"{fn.__name__} but no new entries while the universe is in a downtrend ({kind})."
    return g


mom_regime = _wrap_regime(S.momentum, "index")
mf_regime = _wrap_regime(mf_mom_lowvol_trend, "index")
mom_regime_breadth = _wrap_regime(S.momentum, "breadth")


# ---------------- correlation / diversification ----------------
def _decorrelate(data, i, ranked, n, lookback=120, thresh=0.75):
    """Greedy: walk the ranked list, skip a name whose trailing correlation with
    an already-picked name exceeds thresh. Cuts the concentrated-sector bets
    that produce the -30% drawdowns."""
    x = xcache(data)
    win = x.r.iloc[max(0, i - lookback):i + 1]
    cand = [t for t in ranked if t in win.columns][:40]
    if not cand:
        return []
    cm = win[cand].corr()
    out = []
    for t in cand:
        if len(out) >= n:
            break
        if all(not np.isfinite(cm.loc[t, o]) or cm.loc[t, o] < thresh for o in out):
            out.append(t)
    return out


def mom_decorr(data, i, n=5):
    """Momentum, then drop picks correlated >0.75 with an existing pick."""
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    ranked = S._top(k.ret120.iloc[i], 40, up)
    return _decorrelate(data, i, ranked, n)


def mf_decorr(data, i, n=5):
    """Multi-factor composite, then decorrelated."""
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    score = _composite(data, i, {"mom6": 1.0, "lowvol": 1.0, "trend": 1.0})
    ranked = S._top(score, 40, up)
    return _decorrelate(data, i, ranked, n)


# ---------------- quality tilt (LOOK-AHEAD BIASED — see report) ----------------
_QUAL = {}
def _quality_scores():
    """CURRENT fundamental snapshot. Using it in a 2021-2024 backtest is
    look-ahead bias: these ratios were not knowable then, and survivorship of
    the snapshot favours firms that did well. Results are indicative only."""
    if "s" not in _QUAL:
        try:
            import fundamentals as F
            mf = F.metrics_frame()
        except Exception:
            mf = pd.DataFrame()
        if mf.empty:
            _QUAL["s"] = None
        else:
            def col(c):
                return mf[c] if c in mf.columns else pd.Series(np.nan, index=mf.index)
            parts = pd.DataFrame({
                "roe": _rank(col("returnOnEquity")),
                "margin": _rank(col("profitMargins")),
                "growth": _rank(col("revenueGrowth")),
                "lowdebt": 1 - _rank(col("debtToEquity")),
                "cheap": 1 - _rank(col("priceToBook")),
            })
            # mean of AVAILABLE ranks — requiring all five leaves only 15 names
            sc = parts.mean(axis=1, skipna=True).dropna()
            # price columns carry the ".NS" suffix; fundamentals do not
            sc.index = [t + ".NS" for t in sc.index]
            _QUAL["s"] = sc
    return _QUAL["s"]


def quality_mom(data, i, n=5):
    """LOOK-AHEAD: momentum among the top-half quality names (current snapshot)."""
    q = _quality_scores()
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    if q is None or q.empty:
        return S._top(k.ret120.iloc[i], n, up)
    good = set(q[q >= q.median()].index)
    up = up & pd.Series({t: t in good for t in up.index})
    return S._top(k.ret120.iloc[i], n, up)


REGISTRY = {
    "mom 12-1 >SMA200": mom12_1,
    "lowvol120 >SMA200": lowvol_long,
    "MF mom+lowvol+trend": mf_mom_lowvol_trend,
    "MF mom12-1+lowvol": mf_mom_lowvol,
    "MF 4-factor": mf_quadrant,
    "mom + regime(index)": mom_regime,
    "mom + regime(breadth)": mom_regime_breadth,
    "MF + regime(index)": mf_regime,
    "mom decorr<0.75": mom_decorr,
    "MF decorr<0.75": mf_decorr,
    "LOOKAHEAD quality+mom": quality_mom,
}
