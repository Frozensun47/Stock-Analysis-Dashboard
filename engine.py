"""Data fetch, indicators, weighted buy-score, and backtest engine."""
import json, os, time
import numpy as np
import pandas as pd
import yfinance as yf
from universe import TICKERS

CACHE = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE, exist_ok=True)

# ---------------- data ----------------
def fetch_prices(period="1y", max_age_min=30):
    """Download daily OHLCV for the universe, cached on disk."""
    f = os.path.join(CACHE, f"prices_{period}.pkl")
    if os.path.exists(f) and (time.time() - os.path.getmtime(f)) < max_age_min * 60:
        return pd.read_pickle(f)
    data = yf.download(TICKERS, period=period, interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    data.to_pickle(f)
    return data

# ---------------- indicators ----------------
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def metrics_asof(close, vol, i):
    """Compute metric dict for one stock using data up to index position i (inclusive)."""
    s = close.iloc[: i + 1].dropna()
    if len(s) < 60:
        return None
    d = s.diff().dropna()
    streak = 0
    for v in d.iloc[::-1]:
        if v > 0:
            streak += 1
        else:
            break
    v20 = vol.iloc[: i + 1].dropna().tail(20)
    m = {
        "ret5":    (s.iloc[-1] / s.iloc[-6] - 1) * 100,
        "ret10":   (s.iloc[-1] / s.iloc[-11] - 1) * 100,
        "ret20":   (s.iloc[-1] / s.iloc[-21] - 1) * 100,
        "up5":     (d.tail(5) > 0).sum(),
        "up10":    (d.tail(10) > 0).sum(),
        "streak":  streak,
        "rsi":     rsi(s).iloc[-1],
        "sma20":   (s.iloc[-1] / s.tail(20).mean() - 1) * 100,
        "sma50":   (s.iloc[-1] / s.tail(50).mean() - 1) * 100,
        "vol_surge": (v20.iloc[-1] / v20.mean()) if len(v20) == 20 and v20.mean() > 0 else 1.0,
        "pos52w":  (s.iloc[-1] - s.tail(252).min()) / max(s.tail(252).max() - s.tail(252).min(), 1e-9) * 100,
        "close":   s.iloc[-1],
    }
    return m

# ---------------- weighted score ----------------
# Each metric is mapped to a 0..1 sub-score, then combined by weights -> "buy %".
DEFAULT_WEIGHTS = {
    "ret5": 0.15, "ret10": 0.15, "up10": 0.15, "streak": 0.10,
    "rsi": 0.15, "sma20": 0.10, "sma50": 0.10, "vol_surge": 0.05, "pos52w": 0.05,
}

def _sub(m):
    clip = lambda x: float(np.clip(x, 0, 1))
    return {
        "ret5":  clip(m["ret5"] / 8 + 0.5),          # -4%..+4% -> 0..1
        "ret10": clip(m["ret10"] / 12 + 0.5),
        "up10":  clip(m["up10"] / 10),
        "streak": clip(m["streak"] / 5),
        "rsi":   clip(1 - abs(m["rsi"] - 60) / 30),  # sweet spot ~60 (momentum, not overbought)
        "sma20": clip(m["sma20"] / 6 + 0.5),
        "sma50": clip(m["sma50"] / 10 + 0.5),
        "vol_surge": clip((m["vol_surge"] - 0.5) / 2),
        "pos52w": clip(m["pos52w"] / 100),
    }

def buy_score(m, weights=None):
    w = weights or DEFAULT_WEIGHTS
    sub = _sub(m)
    return round(100 * sum(w[k] * sub[k] for k in w) / sum(w.values()), 1)

def scan(data, asof=-1, weights=None):
    """Score every stock as of a given row index. Returns DataFrame sorted by score."""
    close, vol = data["Close"], data["Volume"]
    idx = len(close) - 1 if asof == -1 else asof
    rows = []
    for t in close.columns:
        m = metrics_asof(close[t], vol[t], idx)
        if m is None or not np.isfinite(m["rsi"]):
            continue
        rows.append({"Symbol": t.replace(".NS", ""), "Buy %": buy_score(m, weights), **{k: round(float(v), 2) for k, v in m.items()}})
    return pd.DataFrame(rows).sort_values("Buy %", ascending=False).reset_index(drop=True)

# ---------------- backtest ----------------
def backtest(data, hold=7, top_n=5, step=5, weights=None, min_score=0):
    """Every `step` days: score all stocks, 'buy' top_n above min_score, sell after `hold` trading days."""
    close = data["Close"]
    trades = []
    start = 70
    for i in range(start, len(close) - hold, step):
        day = scan(data, asof=i, weights=weights)
        picks = day[day["Buy %"] >= min_score].head(top_n)
        for _, r in picks.iterrows():
            t = r["Symbol"] + ".NS"
            buy = close[t].iloc[i]
            sell = close[t].iloc[i + hold]
            if np.isfinite(buy) and np.isfinite(sell):
                trades.append({"date": close.index[i].date(), "Symbol": r["Symbol"],
                               "Buy %": r["Buy %"], "buy": buy, "sell": sell,
                               "ret": (sell / buy - 1) * 100})
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return tdf, {}
    stats = {"trades": len(tdf), "win_rate": round((tdf.ret > 0).mean() * 100, 1),
             "avg_ret": round(tdf.ret.mean(), 2), "median_ret": round(tdf.ret.median(), 2),
             "best": round(tdf.ret.max(), 2), "worst": round(tdf.ret.min(), 2)}
    return tdf, stats

# ---------------- virtual portfolio ----------------
PORT = os.path.join(os.path.dirname(__file__), "portfolio.json")

def load_port():
    if os.path.exists(PORT):
        return json.load(open(PORT))
    return {"cash": 1_000_000.0, "open": [], "closed": []}

def save_port(p):
    json.dump(p, open(PORT, "w"), indent=2, default=str)

def buy(p, symbol, price, amount, score, date):
    if amount > p["cash"]:
        return False
    qty = amount / price
    p["cash"] -= amount
    p["open"].append({"symbol": symbol, "qty": qty, "buy_price": price,
                      "buy_date": str(date), "score": score})
    save_port(p)
    return True

def evaluate(p, close, hold_days=7):
    """Close positions held >= hold_days trading sessions; mark the rest to market."""
    dates = close.index
    still = []
    for pos in p["open"]:
        t = pos["symbol"] + ".NS"
        held = (dates > pd.Timestamp(pos["buy_date"])).sum()
        px = close[t].dropna().iloc[-1]
        if held >= hold_days:
            p["cash"] += pos["qty"] * px
            p["closed"].append({**pos, "sell_price": float(px), "sell_date": str(dates[-1].date()),
                                "ret": (px / pos["buy_price"] - 1) * 100})
        else:
            pos["last_price"], pos["held_days"] = float(px), int(held)
            still.append(pos)
    p["open"] = still
    save_port(p)
    return p

# ---------------- news ----------------
def news_for(symbol, n=5):
    try:
        items = yf.Ticker(symbol + ".NS").news or []
        out = []
        for it in items[:n]:
            c = it.get("content", it)
            out.append({"title": c.get("title"), "publisher": (c.get("provider") or {}).get("displayName", ""),
                        "link": (c.get("canonicalUrl") or {}).get("url", it.get("link", ""))})
        return out
    except Exception:
        return []
