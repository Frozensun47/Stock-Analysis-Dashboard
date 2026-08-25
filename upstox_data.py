"""Fetch 15-minute OHLCV for the NSE universe from Upstox (historical-candle v3).

Token is read from UPSTOX_ACCESS_TOKEN (.env / Streamlit secrets) and never
written to disk except in .env, which is git-ignored.  Output: cache/upstox_15m.pkl,
a dict of wide DataFrames {Open, High, Low, Close, Volume} indexed by IST timestamp.
"""
import gzip, io, json, os, sqlite3, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests
from dotenv import load_dotenv
from universe import SYMBOLS

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")
CACHE = os.path.join(os.path.dirname(__file__), "cache"); os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(CACHE, "upstox_15m.pkl")
BASE = "https://api.upstox.com/v3/historical-candle"
PART = os.path.join(CACHE, "upstox_15m_partial.sqlite")   # resume across runs

# Upstox sits behind Cloudflare, which IP-bans bursts with error 1015 (not a
# normal 429 you can just retry through). A shared token bucket keeps the whole
# thread pool under one global rate instead of each worker backing off alone.
class RateLimiter:
    def __init__(self, per_sec):
        self.interval, self.lock, self.next_at = 1.0 / per_sec, threading.Lock(), 0.0
    def wait(self):
        with self.lock:
            now = time.monotonic()
            self.next_at = max(now, self.next_at) + self.interval
            due = self.next_at - self.interval
        if due > now:
            time.sleep(due - now)

LIMIT = RateLimiter(float(os.getenv("UPSTOX_RPS", "3")))
HDR = {"Accept": "application/json", **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})}

def instrument_keys():
    r = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=60)
    d = json.load(gzip.open(io.BytesIO(r.content)))
    eq = {x["trading_symbol"]: x["instrument_key"] for x in d
          if x.get("segment") == "NSE_EQ" and x.get("instrument_type") == "EQ"}
    return {s: eq[s] for s in SYMBOLS if s in eq}

def _get(url, tries=6):
    for k in range(tries):
        LIMIT.wait()
        try:
            r = requests.get(url, headers=HDR, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** k); continue
        if r.status_code == 200:
            return r.json().get("data", {}).get("candles", [])
        if r.status_code in (429, 503):
            time.sleep(min(60, 5 * 2 ** k))   # Cloudflare 1015 needs a real pause
            continue
        return []
    return []

CHUNK_MONTHS = int(os.getenv("UPSTOX_CHUNK_MONTHS", "1"))  # API caps 15m requests at one month

def fetch_symbol(sym, key, start):
    """Fetch in CHUNK_MONTHS-wide date windows (fewer requests = fewer 1015s)."""
    rows, t0 = [], pd.Timestamp(start)
    end = pd.Timestamp.today().normalize()
    while t0 <= end:
        t1 = min(t0 + pd.offsets.MonthEnd(CHUNK_MONTHS - 1), end)
        url = f"{BASE}/{key.replace('|', '%7C')}/minutes/15/{t1:%Y-%m-%d}/{t0:%Y-%m-%d}"
        rows += _get(url)
        t0 = t1 + pd.Timedelta(days=1)
    if not rows:
        return sym, None
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume", "oi"])
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_convert("Asia/Kolkata")
    return sym, df.drop(columns="oi").drop_duplicates("ts").set_index("ts").sort_index()

def _part():
    con = sqlite3.connect(PART, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS c (sym TEXT, ts INTEGER, o REAL, h REAL, "
                "l REAL, cl REAL, v REAL, PRIMARY KEY (sym, ts)) WITHOUT ROWID")
    return con

def fetch_all(start="2022-01-01", workers=3, resume=True):
    """Resumable:each symbol is written to a partial SQLite as soon as it lands, so a
    Cloudflare ban or Ctrl-C costs only the symbols still outstanding."""
    keys = instrument_keys()
    con = _part()
    done = {r[0] for r in con.execute("SELECT DISTINCT sym FROM c")} if resume else set()
    todo = {s: k for s, k in keys.items() if s not in done}
    print(f"{len(keys)} instruments · {len(done)} already cached · fetching {len(todo)} · from {start}")
    with ThreadPoolExecutor(workers) as ex:
        for n, (sym, df) in enumerate(ex.map(lambda kv: fetch_symbol(kv[0], kv[1], start), todo.items()), 1):
            if df is not None and not df.empty:
                secs = df.index.tz_convert("UTC").tz_localize(None).astype("datetime64[s]").astype("int64")
                con.executemany("INSERT OR REPLACE INTO c VALUES (?,?,?,?,?,?,?)",
                                [(sym, int(t), *map(float, r)) for t, r in zip(secs, df.values)])
                con.commit()
            print(f"  {n}/{len(todo)} {sym}: {0 if df is None else len(df):,} bars", flush=True)
    frames = {}
    for sym, in con.execute("SELECT DISTINCT sym FROM c"):
        d = pd.read_sql("SELECT ts, o, h, l, cl, v FROM c WHERE sym=? ORDER BY ts", con, params=(sym,))
        d.index = pd.to_datetime(d.pop("ts"), unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
        frames[sym + ".NS"] = d.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                                "cl": "Close", "v": "Volume"})
    con.close()
    if not frames:
        print("nothing fetched"); return {}
    panel = {f: pd.concat({s: d[f] for s, d in frames.items()}, axis=1).sort_index()
             for f in ["Open", "High", "Low", "Close", "Volume"]}
    pd.to_pickle(panel, OUT)
    c = panel["Close"]
    print(f"saved {OUT}: {c.shape[0]:,} bars × {c.shape[1]} stocks, {c.index[0]} → {c.index[-1]}")
    return panel

SPLIT_THRESHOLD = 0.25    # a 15-minute bar never moves this much on real trading
DEMERGER_THRESHOLD = 0.30 # overnight gap this large is a corporate action, not a crash
BLANK_BARS = 120          # bars blanked either side of an unadjustable action

def adjust_splits(panel, threshold=SPLIT_THRESHOLD, verbose=True):
    """Back-adjust for splits and bonus issues.

    Upstox serves RAW candles: a 1:2 split shows up as a genuine-looking -50%
    bar, which poisons both the training labels and the backtest (it cost one
    VBL trade -50% in the first full run). Any bar-to-bar move beyond the
    threshold that is close to a simple ratio (1:2, 1:5, 2:1 …) is treated as a
    corporate action, and every bar BEFORE it is divided by that ratio so the
    series becomes continuous — the same convention as an adjusted close.
    """
    panel = {k: v.copy() for k, v in panel.items()}
    c = panel["Close"]
    r = c.pct_change()
    fixed = []
    for sym in c.columns:
        jumps = r[sym][r[sym].abs() > threshold].dropna()
        if jumps.empty:
            continue
        factor = pd.Series(1.0, index=c.index)
        for ts, move in jumps.items():
            ratio = 1 + move
            # only treat it as a corporate action if it lands near a clean ratio
            if not any(abs(ratio - t) < 0.06 for t in (0.5, 0.2, 0.1, 0.25, 1/3, 2.0, 5.0, 10.0, 3.0, 4.0)):
                continue
            factor.loc[:ts] *= ratio
            factor.loc[ts] = factor.loc[ts] / ratio  # the jump bar is already post-action
            fixed.append((sym, ts.date(), round(ratio, 3)))
        for f in ["Open", "High", "Low", "Close"]:
            panel[f][sym] = panel[f][sym] * factor
        panel["Volume"][sym] = panel["Volume"][sym] / factor.replace(0, 1)
    if verbose and fixed:
        print("split/bonus adjustments applied:")
        for sym, d, ratio in fixed:
            print(f"  {sym} {d} ×{ratio}")

    # Demergers cannot be back-adjusted by a ratio — the company genuinely
    # changed shape. Neither Upstox nor Yahoo adjusts them, so Vedanta's April
    # 2026 demerger reads as a -65% four-day "return". Rather than invent a
    # factor, blank a window around each event so it cannot become a label or a
    # trade. Real crashes (Adani/Hindenburg 2023) fall over several sessions and
    # are deliberately left intact — only single overnight gaps are treated as
    # corporate actions.
    c = panel["Close"]
    blanked = []
    for sym in c.columns:
        # per-symbol, on the observed bars only: a halted or late-opening session
        # leaves NaNs that would otherwise hide the gap (Vedanta's demerger day
        # opened at 10:00, so the 09:45 slot was empty and pct_change returned NaN)
        s_ = c[sym].dropna()
        if s_.empty:
            continue
        d_ = s_.index.normalize()
        gap = s_.pct_change()[d_ != pd.Series(d_).shift(1).values]
        for ts in gap[gap.abs() > DEMERGER_THRESHOLD].dropna().index:
            k = c.index.get_loc(ts)
            lo, hi = max(0, k - BLANK_BARS), min(len(c) - 1, k + BLANK_BARS)
            for f in ["Open", "High", "Low", "Close", "Volume"]:
                panel[f].iloc[lo:hi + 1, panel[f].columns.get_loc(sym)] = float("nan")
            blanked.append((sym, ts.date(), round(gap.loc[ts] * 100, 1)))
    if verbose and blanked:
        print("unadjustable corporate actions — window blanked:")
        for sym, d, mv in blanked:
            print(f"  {sym} {d} ({mv:+.1f}% overnight gap)")
    return panel


def load_15m(adjust=True):
    panel = pd.read_pickle(OUT)
    return adjust_splits(panel) if adjust else panel

if __name__ == "__main__":
    fetch_all(sys.argv[1] if len(sys.argv) > 1 else "2022-01-01",
              workers=int(os.getenv("UPSTOX_WORKERS", "3")))
