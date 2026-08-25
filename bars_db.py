"""Local SQLite store of 1-minute NSE bars, topped up daily from yfinance.

yfinance only serves ~7 calendar days of 1-minute history per request, so the
value here is cumulative: run `python bars_db.py sync` once a day (cron) and the
database grows into a private minute-level archive that outlives the API window.

Schema
    bars(symbol TEXT, ts INTEGER epoch-seconds UTC, open, high, low, close, volume)
    PRIMARY KEY (symbol, ts)   -- INSERT OR REPLACE makes every sync idempotent

Usage
    python bars_db.py sync            # fetch the last 7d of 1m bars, upsert
    python bars_db.py sync 2          # only the last 2 days (fast daily top-up)
    python bars_db.py stats           # rows, symbols, date span
    python bars_db.py export 15       # write cache/db_15m.pkl resampled to 15m
"""
import os, sqlite3, sys
import pandas as pd, yfinance as yf
from universe import TICKERS

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.getenv("BARS_DB", os.path.join(HERE, "cache", "bars.sqlite"))
IST = "Asia/Kolkata"

DDL = """
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT NOT NULL, ts INTEGER NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_bars_ts ON bars(ts);
"""

def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(DDL)
    return con

def sync(days=7, batch=40):
    """Download 1m bars for the universe and upsert them. Safe to re-run."""
    days = max(1, min(int(days), 7))          # yfinance 1m limit
    con, total = connect(), 0
    for k in range(0, len(TICKERS), batch):
        chunk = TICKERS[k:k + batch]
        df = yf.download(chunk, period=f"{days}d", interval="1m",
                         auto_adjust=False, progress=False, threads=True, group_by="column")
        if df is None or df.empty:
            continue
        idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        secs = idx.tz_localize(None).astype("datetime64[s]").astype("int64")
        rows = []
        for t in chunk:
            try:
                sub = pd.DataFrame({f: df[(f, t)] for f in ["Open", "High", "Low", "Close", "Volume"]})
            except KeyError:
                continue
            sub = sub.dropna(how="all")
            if sub.empty:
                continue
            pos = idx.get_indexer(sub.index)
            rows += [(t, int(secs[p]), *map(_f, r)) for p, r in zip(pos, sub.itertuples(index=False))]
        con.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)", rows)
        con.commit()
        total += len(rows)
        print(f"  {min(k+batch, len(TICKERS))}/{len(TICKERS)} symbols · {total:,} bars upserted", flush=True)
    con.close()
    print(f"sync done: {total:,} rows written to {DB}")
    return total

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def load(symbols=None, start=None, end=None, interval=None):
    """Read bars back as a dict of wide DataFrames (IST index), optionally resampled."""
    con = connect()
    q, args = "SELECT symbol, ts, open, high, low, close, volume FROM bars WHERE 1=1", []
    if symbols:
        q += f" AND symbol IN ({','.join('?' * len(symbols))})"; args += list(symbols)
    for col, val, op in [("ts", start, ">="), ("ts", end, "<=")]:
        if val is not None:
            q += f" AND {col} {op} ?"; args.append(int(pd.Timestamp(val, tz=IST).timestamp()))
    df = pd.read_sql(q, con, params=args)
    con.close()
    if df.empty:
        return {}
    df["ts"] = pd.to_datetime(df.ts, unit="s", utc=True).dt.tz_convert(IST)
    panel = {f.capitalize(): df.pivot_table(index="ts", columns="symbol", values=f, aggfunc="last").sort_index()
             for f in ["open", "high", "low", "close", "volume"]}
    if interval:
        how = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        panel = {f: v.resample(interval, label="left", closed="left", origin="start_day",
                               offset="9h15min").agg(how[f]).dropna(how="all")
                 for f, v in panel.items()}
    return panel

def stats():
    con = connect()
    n, s, a, b = con.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(ts), MAX(ts) FROM bars").fetchone()
    con.close()
    if not n:
        print(f"{DB}: empty — run `python bars_db.py sync`"); return
    fmt = lambda t: pd.to_datetime(t, unit="s", utc=True).tz_convert(IST)
    size = os.path.getsize(DB) / 1e6
    print(f"{DB}\n  {n:,} bars · {s} symbols · {fmt(a)} → {fmt(b)} · {size:.1f} MB")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sync"
    if cmd == "sync":
        sync(sys.argv[2] if len(sys.argv) > 2 else 7); stats()
    elif cmd == "stats":
        stats()
    elif cmd == "export":
        mins = sys.argv[2] if len(sys.argv) > 2 else "15"
        panel = load(interval=f"{mins}min")
        out = os.path.join(HERE, "cache", f"db_{mins}m.pkl")
        pd.to_pickle(panel, out)
        print(f"wrote {out}: {panel['Close'].shape[0]:,} bars × {panel['Close'].shape[1]} stocks")
    else:
        print(__doc__)
