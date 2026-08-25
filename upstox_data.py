"""Fetch 15-minute OHLCV for the NSE universe from Upstox (historical-candle v3).

Token is read from UPSTOX_ACCESS_TOKEN (.env / Streamlit secrets) and never
written to disk except in .env, which is git-ignored.  Output: cache/upstox_15m.pkl,
a dict of wide DataFrames {Open, High, Low, Close, Volume} indexed by IST timestamp.
"""
import gzip, io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
from dotenv import load_dotenv
from universe import SYMBOLS

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")
CACHE = os.path.join(os.path.dirname(__file__), "cache"); os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(CACHE, "upstox_15m.pkl")
BASE = "https://api.upstox.com/v3/historical-candle"
HDR = {"Accept": "application/json", **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})}

def instrument_keys():
    r = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=60)
    d = json.load(gzip.open(io.BytesIO(r.content)))
    eq = {x["trading_symbol"]: x["instrument_key"] for x in d
          if x.get("segment") == "NSE_EQ" and x.get("instrument_type") == "EQ"}
    return {s: eq[s] for s in SYMBOLS if s in eq}

def _get(url, tries=4):
    for k in range(tries):
        r = requests.get(url, headers=HDR, timeout=30)
        if r.status_code == 200:
            return r.json().get("data", {}).get("candles", [])
        if r.status_code == 429:
            time.sleep(2 * (k + 1)); continue
        return []
    return []

def fetch_symbol(sym, key, start):
    """Monthly chunks (Upstox limit for 1-15 min intervals), newest first per chunk."""
    rows, t0 = [], pd.Timestamp(start)
    end = pd.Timestamp.today().normalize()
    while t0 <= end:
        t1 = min(t0 + pd.offsets.MonthEnd(0), end)
        url = f"{BASE}/{key.replace('|', '%7C')}/minutes/15/{t1:%Y-%m-%d}/{t0:%Y-%m-%d}"
        rows += _get(url)
        t0 = t1 + pd.Timedelta(days=1)
    if not rows:
        return sym, None
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume", "oi"])
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert("Asia/Kolkata")
    return sym, df.drop(columns="oi").drop_duplicates("ts").set_index("ts").sort_index()

def fetch_all(start="2022-01-01", workers=8):
    keys = instrument_keys()
    print(f"{len(keys)} instruments · 15m from {start}")
    frames = {}
    with ThreadPoolExecutor(workers) as ex:
        for n, (sym, df) in enumerate(ex.map(lambda kv: fetch_symbol(kv[0], kv[1], start), keys.items()), 1):
            if df is not None:
                frames[sym + ".NS"] = df
            if n % 10 == 0:
                print(f"  {n}/{len(keys)}", flush=True)
    panel = {f: pd.concat({s: d[f] for s, d in frames.items()}, axis=1).sort_index()
             for f in ["Open", "High", "Low", "Close", "Volume"]}
    pd.to_pickle(panel, OUT)
    c = panel["Close"]
    print(f"saved {OUT}: {c.shape[0]:,} bars × {c.shape[1]} stocks, {c.index[0]} → {c.index[-1]}")
    return panel

def load_15m():
    return pd.read_pickle(OUT)

if __name__ == "__main__":
    fetch_all(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01")
