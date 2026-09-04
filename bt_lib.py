"""Backtest library: realistic Groww costs + vectorised replica of engine.buy_score.

Nothing here modifies existing modules. Everything new is prefixed bt_.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

# ---------------------------------------------------------------- costs
def groww_cost(buy_value, sell_value):
    """Total round-trip cost in rupees for ONE scrip, Groww delivery (equity)."""
    brokerage = min(20.0, 0.001 * buy_value) + min(20.0, 0.001 * sell_value)
    stt = 0.001 * buy_value + 0.001 * sell_value
    exch = 0.0000297 * (buy_value + sell_value)
    sebi = 0.000001 * (buy_value + sell_value)
    stamp = 0.00015 * buy_value
    dp = 13.5
    gst = 0.18 * (brokerage + exch + dp)
    return brokerage + stt + exch + sebi + stamp + dp + gst


def net_return_pct(gross_ret_pct, ticket):
    """Net % return on a `ticket`-rupee position with gross_ret_pct gross move."""
    buy_v = ticket
    sell_v = ticket * (1 + gross_ret_pct / 100.0)
    c = groww_cost(buy_v, sell_v)
    return (sell_v - buy_v - c) / buy_v * 100.0


def cost_pct(ticket, gross_ret_pct=0.0):
    buy_v = ticket
    sell_v = ticket * (1 + gross_ret_pct / 100.0)
    return groww_cost(buy_v, sell_v) / buy_v * 100.0


# ---------------------------------------------------------------- data
def load_data():
    """Merge the 5y and 1y caches into one daily panel (5y ends 2026-08-24)."""
    a = pd.read_pickle(os.path.join(CACHE, "prices_5y.pkl"))
    b = pd.read_pickle(os.path.join(CACHE, "prices_1y.pkl"))
    cols = [c for c in a.columns if c in set(b.columns)]
    a, b = a[cols], b[cols]
    out = pd.concat([a, b[~b.index.isin(a.index)]]).sort_index()
    return out


# ---------------------------------------------------------------- vectorised score
_W = {"ret5": 0.15, "ret10": 0.15, "up10": 0.15, "streak": 0.10,
      "rsi": 0.15, "sma20": 0.10, "sma50": 0.10, "vol_surge": 0.05, "pos52w": 0.05}


def _rsi_sma(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _pos_streak(d):
    """Consecutive positive-diff streak ending at each bar."""
    pos = (d > 0).astype(int)
    grp = (pos == 0).cumsum()
    return pos.groupby(grp).cumsum()


def score_panel(data):
    """DataFrame (dates x tickers) of Buy % replicating engine.buy_score exactly.

    NaN where engine would have skipped the stock (<60 obs or non-finite RSI).
    """
    close, vol = data["Close"], data["Volume"]
    out = {}
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < 60:
            continue
        v = vol[t].reindex(s.index)
        d = s.diff()
        m_ret5 = (s / s.shift(5) - 1) * 100
        m_ret10 = (s / s.shift(10) - 1) * 100
        m_up10 = (d > 0).rolling(10).sum()
        m_streak = _pos_streak(d.iloc[1:]).reindex(s.index)
        m_rsi = _rsi_sma(s)
        m_sma20 = (s / s.rolling(20).mean() - 1) * 100
        m_sma50 = (s / s.rolling(50).mean() - 1) * 100
        vmean = v.rolling(20).mean()
        m_vs = (v / vmean).where(vmean > 0, 1.0)
        lo = s.rolling(252, min_periods=1).min()
        hi = s.rolling(252, min_periods=1).max()
        m_p52 = (s - lo) / (hi - lo).clip(lower=1e-9) * 100

        cl = lambda x: x.clip(0, 1)
        sub = {
            "ret5": cl(m_ret5 / 8 + 0.5),
            "ret10": cl(m_ret10 / 12 + 0.5),
            "up10": cl(m_up10 / 10),
            "streak": cl(m_streak / 5),
            "rsi": cl(1 - (m_rsi - 60).abs() / 30),
            "sma20": cl(m_sma20 / 6 + 0.5),
            "sma50": cl(m_sma50 / 10 + 0.5),
            "vol_surge": cl((m_vs - 0.5) / 2),
            "pos52w": cl(m_p52 / 100),
        }
        sc = sum(_W[k] * sub[k] for k in _W) / sum(_W.values()) * 100
        # engine requires >=60 observations of history
        sc.iloc[:59] = np.nan
        sc = sc.where(m_rsi.notna())
        out[t] = sc.reindex(close.index)
    return pd.DataFrame(out).round(1)


# ---------------------------------------------------------------- exits
def simulate_exit(high, low, close, i, hold, trail=None, tp=None, sl=None):
    """Return (gross_ret_pct, days_held, reason) for one position entered at close[i].

    Evaluated on daily bars. Trail/SL checked on CLOSES (conservative, no
    intrabar assumption); TP checked on HIGHs (fillable intraday).
    """
    entry = close[i]
    peak = entry
    n = len(close)
    end = min(i + hold, n - 1)
    for j in range(i + 1, end + 1):
        c, h = close[j], high[j]
        if not np.isfinite(c):
            continue
        if tp is not None and np.isfinite(h) and h >= entry * (1 + tp / 100):
            return (tp, j - i, "tp")
        if sl is not None and c <= entry * (1 - sl / 100):
            return ((c / entry - 1) * 100, j - i, "sl")
        if trail is not None and c <= peak * (1 - trail / 100):
            return ((c / entry - 1) * 100, j - i, "trail")
        peak = max(peak, c)
    px = close[end]
    return ((px / entry - 1) * 100, end - i, "time")
