"""Free stock-news collector -> SQLite corpus, with finance-lexicon sentiment.

No API key and no paid tier: every source below is a public RSS/Atom feed.
  * Google News RSS, one query per NSE symbol  -> per-stock coverage
  * Publisher market feeds (Moneycontrol, Economic Times, Livemint, Business
    Standard, Hindu BusinessLine, Zerodha Z-Connect, Trendlyne blog) -> market
    context and analysis articles, symbol-matched against the universe
  * yfinance's own news payload as a fallback per symbol

Why RSS over the keyed "free tiers": NewsAPI's free plan forbids production use
and lags 24h, GNews caps at 100 req/day, and Finnhub's company-news is US-only.
RSS has no cap, no key to leak, and no commercial restriction.

Schema
    news(id TEXT PK, symbol TEXT, ts INTEGER, title, summary, link, source,
         sentiment REAL, npos INT, nneg INT)
    id = sha1(symbol|link) so re-running never duplicates a story.

Usage
    python news_db.py sync            # every symbol + all market feeds
    python news_db.py sync RELIANCE TCS
    python news_db.py feeds           # market-wide feeds only (fast)
    python news_db.py stats
    python news_db.py recent RELIANCE
"""
import hashlib, os, re, sqlite3, sys, time
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET
import requests
from universe import SYMBOLS

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.getenv("NEWS_DB", os.path.join(HERE, "cache", "news.sqlite"))
UA = {"User-Agent": "Mozilla/5.0 (compatible; stockdash/1.0)"}

# Market-wide feeds: broad trend/analysis material, symbol-matched after fetch.
FEEDS = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "moneycontrol_results": "https://www.moneycontrol.com/rss/results.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "livemint_companies": "https://www.livemint.com/rss/companies",
    "bs_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "bs_companies": "https://www.business-standard.com/rss/companies-101.rss",
    "hindu_businessline": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
    "zerodha_zconnect": "https://zerodha.com/z-connect/feed",
    "trendlyne_blog": "https://trendlyne.com/blog/feed/",
    "yahoo_india": "https://finance.yahoo.com/news/rssindex",
}

def gnews_url(sym):
    q = requests.utils.quote(f'"{sym}" (NSE OR share OR stock) when:7d')
    return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

# ---- sentiment: compact Loughran-McDonald-style finance lexicon ----
POS = set("""beat beats beating surge surged surges rally rallied jump jumped soar soared
gain gains gained rise rises rose upgrade upgraded outperform buy bullish profit profitable
record high strong strength growth grow grew expand expansion approval approved wins won
order orders contract deal acquisition acquires stake boost boosted robust healthy recovery
rebound optimistic upbeat dividend bonus buyback breakout momentum topline margin-expansion""".split())
NEG = set("""miss misses missed fall falls fell drop dropped plunge plunged slump slumped
decline declined declines sink sank downgrade downgraded underperform sell bearish loss losses
weak weakness slowdown slow cut cuts cutting probe investigation fraud scam penalty fine
default debt lawsuit sue resign resigned exit layoff layoffs shut halt halted recall ban banned
warning warns risk risky concern concerns pressure headwind selloff crash correction""".split())
NEGATORS = {"not", "no", "never", "without", "fails", "fail", "failed"}
TOKEN = re.compile(r"[a-z][a-z'-]+")

def score(text):
    """Returns (sentiment in [-1,1], n_pos, n_neg) with simple negation flipping."""
    w = TOKEN.findall((text or "").lower())
    p = n = 0
    for k, t in enumerate(w):
        flip = k and w[k - 1] in NEGATORS
        if t in POS:
            n += 1 if flip else 0; p += 0 if flip else 1
        elif t in NEG:
            p += 1 if flip else 0; n += 0 if flip else 1
    return ((p - n) / (p + n) if p + n else 0.0), p, n

DDL = """
CREATE TABLE IF NOT EXISTS news (
  id TEXT PRIMARY KEY, symbol TEXT, ts INTEGER, title TEXT, summary TEXT,
  link TEXT, source TEXT, sentiment REAL, npos INTEGER, nneg INTEGER);
CREATE INDEX IF NOT EXISTS idx_news_sym_ts ON news(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_news_ts ON news(ts);
"""

def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(DDL)
    return con

def _strip(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()

def parse_feed(xml):
    """RSS 2.0 and Atom -> [(title, summary, link, epoch_seconds)]."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    items = root.iter("item") if root.find(".//item") is not None else root.iter("{http://www.w3.org/2005/Atom}entry")
    for it in items:
        g = lambda *tags: next((it.findtext(t) for t in tags if it.findtext(t)), "")
        title = _strip(g("title", "a:title"))
        link = g("link") or (it.find("a:link", ns).get("href") if it.find("a:link", ns) is not None else "")
        summary = _strip(g("description", "summary", "a:summary", "content"))
        date = g("pubDate", "published", "updated", "a:published", "a:updated")
        ts = None
        try:
            ts = int(parsedate_to_datetime(date).timestamp())
        except Exception:
            try:
                import datetime as _dt
                ts = int(_dt.datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp())
            except Exception:
                ts = int(time.time())
        if title and link:
            out.append((title, summary, link.strip(), ts))
    return out

def _fetch(url, tries=2):
    for _ in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(1)
    return b""

def _rows(symbol, source, items):
    rows = []
    for title, summary, link, ts in items:
        s, p, n = score(f"{title}. {summary}")
        rid = hashlib.sha1(f"{symbol}|{link}".encode()).hexdigest()
        rows.append((rid, symbol, ts, title, summary[:2000], link, source, s, p, n))
    return rows

def sync_symbols(symbols=None, workers=8):
    """Google News RSS per symbol (+ yfinance news as a top-up)."""
    symbols = symbols or SYMBOLS
    con, total = connect(), 0
    def one(sym):
        rows = _rows(sym, "google_news", parse_feed(_fetch(gnews_url(sym))))
        try:
            import yfinance as yf
            for a in (yf.Ticker(sym + ".NS").news or []):
                c = a.get("content", a)
                link = (c.get("canonicalUrl") or {}).get("url") or c.get("link", "")
                title, summ = c.get("title", ""), _strip(c.get("summary", ""))
                pub = c.get("pubDate") or c.get("providerPublishTime")
                try:
                    import datetime as _dt
                    ts = int(pub) if isinstance(pub, (int, float)) else int(
                        _dt.datetime.fromisoformat(str(pub).replace("Z", "+00:00")).timestamp())
                except Exception:
                    ts = int(time.time())
                if title and link:
                    rows += _rows(sym, "yfinance", [(title, summ, link, ts)])
        except Exception:
            pass
        return rows
    with ThreadPoolExecutor(workers) as ex:
        for n, rows in enumerate(ex.map(one, symbols), 1):
            if rows:
                con.executemany("INSERT OR REPLACE INTO news VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
                con.commit(); total += len(rows)
            if n % 20 == 0:
                print(f"  {n}/{len(symbols)} symbols · {total:,} articles", flush=True)
    con.close()
    print(f"per-symbol news: {total:,} articles upserted")
    return total

def sync_feeds(workers=6):
    """Market-wide publisher feeds; each article is tagged with any symbol it names."""
    con, total, untagged = connect(), 0, 0
    pat = {s: re.compile(rf"\b{re.escape(s)}\b", re.I) for s in SYMBOLS}
    def one(kv):
        name, url = kv
        return name, parse_feed(_fetch(url))
    with ThreadPoolExecutor(workers) as ex:
        for name, items in ex.map(one, FEEDS.items()):
            rows = []
            for title, summary, link, ts in items:
                blob = f"{title} {summary}"
                hits = [s for s, p in pat.items() if p.search(blob)]
                for sym in (hits or ["_MARKET"]):
                    rows += _rows(sym, name, [(title, summary, link, ts)])
                untagged += 0 if hits else 1
            if rows:
                con.executemany("INSERT OR REPLACE INTO news VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
                con.commit(); total += len(rows)
            print(f"  {name}: {len(items)} items", flush=True)
    con.close()
    print(f"market feeds: {total:,} rows ({untagged} kept as _MARKET context)")
    return total

def sentiment_panel(days=30):
    """Daily mean sentiment + article count per symbol — join this onto price features."""
    import pandas as pd
    con = connect()
    df = pd.read_sql(
        "SELECT symbol, date(ts,'unixepoch','+5 hours','+30 minutes') d, "
        "AVG(sentiment) sent, COUNT(*) n FROM news "
        "WHERE ts > strftime('%s','now',?) AND symbol != '_MARKET' GROUP BY symbol, d",
        con, params=[f"-{int(days)} days"])
    con.close()
    return df

def stats():
    con = connect()
    n, s, a, b = con.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(ts), MAX(ts) FROM news").fetchone()
    if not n:
        print(f"{DB}: empty — run `python news_db.py sync`"); con.close(); return
    import datetime as dt
    f = lambda t: dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")
    print(f"{DB}\n  {n:,} articles · {s} symbols · {f(a)} → {f(b)} · {os.path.getsize(DB)/1e6:.1f} MB")
    print("  by source:")
    for src, c in con.execute("SELECT source, COUNT(*) c FROM news GROUP BY source ORDER BY c DESC"):
        print(f"    {src:24s} {c:,}")
    print("  most-covered symbols:")
    for sym, c, s_ in con.execute("SELECT symbol, COUNT(*) c, AVG(sentiment) FROM news "
                                  "GROUP BY symbol ORDER BY c DESC LIMIT 8"):
        print(f"    {sym:12s} {c:5,}  avg sentiment {s_:+.2f}")
    con.close()

def recent(symbol, n=15):
    con = connect()
    for ts, t, src, s in con.execute(
            "SELECT ts, title, source, sentiment FROM news WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, n)):
        import datetime as dt
        print(f"  {dt.datetime.fromtimestamp(ts):%Y-%m-%d %H:%M} [{s:+.2f}] {t[:95]}  ({src})")
    con.close()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sync"
    if cmd == "sync":
        sync_feeds(); sync_symbols(sys.argv[2:] or None); stats()
    elif cmd == "feeds":
        sync_feeds(); stats()
    elif cmd == "stats":
        stats()
    elif cmd == "recent":
        recent(sys.argv[2] if len(sys.argv) > 2 else "RELIANCE")
    else:
        print(__doc__)
