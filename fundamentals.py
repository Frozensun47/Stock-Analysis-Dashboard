"""Company financial statements + key metrics -> SQLite, free via yfinance.

yfinance exposes annual and quarterly income statement, balance sheet and cash
flow for NSE tickers, plus a `info` blob of ratios. Both are stored long-format
so new line items never require a migration.

Schema
    statements(symbol, statement, period, freq, item, value)  PK(all but value)
        statement ∈ income|balance|cashflow   freq ∈ A|Q   period = 'YYYY-MM-DD'
    metrics(symbol, asof, item, value)        PK(symbol, asof, item)

Usage
    python fundamentals.py sync              # whole universe
    python fundamentals.py sync RELIANCE TCS
    python fundamentals.py stats
    python fundamentals.py show RELIANCE
"""
import os, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, yfinance as yf
from universe import SYMBOLS

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.getenv("FUNDA_DB", os.path.join(HERE, "cache", "fundamentals.sqlite"))

DDL = """
CREATE TABLE IF NOT EXISTS statements (
  symbol TEXT, statement TEXT, freq TEXT, period TEXT, item TEXT, value REAL,
  PRIMARY KEY (symbol, statement, freq, period, item)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS metrics (
  symbol TEXT, asof TEXT, item TEXT, value REAL,
  PRIMARY KEY (symbol, asof, item)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_st_sym ON statements(symbol, item);
"""

# ratios worth keeping from .info; everything else in that blob is noise or text
KEEP = ["marketCap", "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        "enterpriseValue", "enterpriseToEbitda", "enterpriseToRevenue", "profitMargins",
        "grossMargins", "operatingMargins", "ebitdaMargins", "returnOnEquity", "returnOnAssets",
        "debtToEquity", "currentRatio", "quickRatio", "totalDebt", "totalCash", "freeCashflow",
        "operatingCashflow", "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        "trailingEps", "forwardEps", "bookValue", "dividendYield", "payoutRatio", "beta",
        "heldPercentInsiders", "heldPercentInstitutions", "shortRatio", "pegRatio"]

def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(DDL)
    return con

def _melt(sym, df, statement, freq):
    """yfinance returns items x period-columns; flatten to tidy rows."""
    if df is None or getattr(df, "empty", True):
        return []
    rows = []
    for period in df.columns:
        p = pd.Timestamp(period).strftime("%Y-%m-%d")
        for item, val in df[period].items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if pd.notna(v):
                rows.append((sym, statement, freq, p, str(item), v))
    return rows

def fetch_one(sym):
    t = yf.Ticker(sym + ".NS")
    rows, mrows = [], []
    for statement, a, q in [("income", "income_stmt", "quarterly_income_stmt"),
                            ("balance", "balance_sheet", "quarterly_balance_sheet"),
                            ("cashflow", "cashflow", "quarterly_cashflow")]:
        for freq, attr in [("A", a), ("Q", q)]:
            try:
                rows += _melt(sym, getattr(t, attr), statement, freq)
            except Exception:
                pass
    try:
        info = t.info or {}
        asof = pd.Timestamp.today().strftime("%Y-%m-%d")
        for k in KEEP:
            v = info.get(k)
            if isinstance(v, (int, float)) and pd.notna(v):
                mrows.append((sym, asof, k, float(v)))
    except Exception:
        pass
    return rows, mrows

def sync(symbols=None, workers=6):
    symbols = symbols or SYMBOLS
    con, ns, nm = connect(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        for i, (rows, mrows) in enumerate(ex.map(fetch_one, symbols), 1):
            if rows:
                con.executemany("INSERT OR REPLACE INTO statements VALUES (?,?,?,?,?,?)", rows); ns += len(rows)
            if mrows:
                con.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?)", mrows); nm += len(mrows)
            con.commit()
            if i % 10 == 0:
                print(f"  {i}/{len(symbols)} · {ns:,} statement rows · {nm:,} metrics", flush=True)
    con.close()
    print(f"fundamentals: {ns:,} statement rows, {nm:,} metric rows")
    return ns, nm

def statement(symbol, which="income", freq="A"):
    """Wide DataFrame: items x periods, newest period first."""
    con = connect()
    df = pd.read_sql("SELECT period, item, value FROM statements "
                     "WHERE symbol=? AND statement=? AND freq=?", con, params=(symbol, which, freq))
    con.close()
    if df.empty:
        return df
    return df.pivot(index="item", columns="period", values="value").sort_index(axis=1, ascending=False)

def metrics_frame(symbols=None):
    """Latest ratio snapshot per symbol — ready to merge into the ML feature table."""
    con = connect()
    q = ("SELECT m.symbol, m.item, m.value FROM metrics m JOIN "
         "(SELECT symbol, MAX(asof) a FROM metrics GROUP BY symbol) x "
         "ON m.symbol=x.symbol AND m.asof=x.a")
    df = pd.read_sql(q, con)
    con.close()
    if df.empty:
        return df
    out = df.pivot(index="symbol", columns="item", values="value")
    return out.loc[out.index.intersection(symbols)] if symbols else out

def stats():
    con = connect()
    a = con.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), COUNT(DISTINCT item) FROM statements").fetchone()
    b = con.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM metrics").fetchone()
    print(f"{DB}\n  statements: {a[0]:,} rows · {a[1]} symbols · {a[2]} distinct line items")
    print(f"  metrics:    {b[0]:,} rows · {b[1]} symbols · {os.path.getsize(DB)/1e6:.1f} MB")
    con.close()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sync"
    if cmd == "sync":
        sync(sys.argv[2:] or None); stats()
    elif cmd == "stats":
        stats()
    elif cmd == "show":
        sym = sys.argv[2] if len(sys.argv) > 2 else "RELIANCE"
        df = statement(sym, "income", "A")
        print(df.head(15).to_string() if not df.empty else f"no data for {sym} — run sync")
    else:
        print(__doc__)
