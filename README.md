# NSE Stock Analysis Dashboard

Streamlit dashboard + backtested trading research for the NSE Nifty-100/midcap universe.

- **Daily strategy** (`strategy.py`) — mean-reversion in an uptrend (price > SMA50, RSI(14) < 40,
  lowest RSI wins), next-day-open entry, 4% trailing stop on closes, 15-session time exit.
  Walk-forward tested over 5 years net of ~0.35% round-trip costs.
- **15-minute intraday model** (`model_15m.py`) — gradient-boosted regressor predicting the
  forward 8-bar (~2h) return from intraday features (VWAP deviation, RSI, ATR, volume surge,
  time-of-day, market breadth). Walk-forward trained (expanding window, monthly retrain),
  entry at the *next* bar's open, trailing stop, forced flat by the close of each session.
- **Data** — daily bars from yfinance (`engine.py`), 15-minute bars from the
  Upstox historical-candle v3 API (`upstox_data.py`).

## Data collection (local SQLite corpus)

| Script | Store | What it collects |
|---|---|---|
| `bars_db.py` | `cache/bars.sqlite` | 1-minute OHLCV from yfinance. The API only serves a 7-day window, so a daily `sync` accumulates an archive that outlives it. `export 15` resamples to any interval. |
| `news_db.py` | `cache/news.sqlite` | Free RSS only — no key, no quota: Google News per symbol, plus Moneycontrol, Economic Times, Livemint, Business Standard, BusinessLine, Zerodha Z-Connect, Trendlyne and Yahoo. Each article is scored with a finance lexicon (negation-aware); `sentiment_panel()` returns a daily per-symbol series ready to join onto price features. |
| `fundamentals.py` | `cache/fundamentals.sqlite` | Annual + quarterly income statement, balance sheet and cash flow, plus ~30 ratios, long-format so new line items need no migration. |
| `upstox_data.py` | `cache/upstox_15m.pkl` | 15-minute history from the Upstox v3 API. |
| `storage.py` | R2 / S3 / B2 | Pushes the SQLite files to object storage once they outgrow the laptop. |

```bash
./daily_sync.sh        # bars + news + fundamentals, then optional object-storage push
```

Cron it at 20:00 IST on weekdays:
```
0 20 * * 1-5 /path/to/stockdash/daily_sync.sh >> /tmp/stockdash.log 2>&1
```

**Why RSS instead of a news API:** NewsAPI's free plan bans production use and
delays articles 24h, GNews caps at 100 requests/day, and Finnhub's company-news
is US-only. RSS has no cap, no key to leak and no commercial restriction.

**Why R2 over Firestore:** Firestore bills per document write, which is the wrong
shape for append-only bar and news blobs. R2 is S3-compatible with a 10 GB free
tier and zero egress, and the same `boto3` code works against S3 or B2.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # then paste your Upstox access token
.venv/bin/python upstox_data.py 2022-01-01   # fetch 15m history -> cache/
.venv/bin/python model_15m.py                # walk-forward test + train + save model
.venv/bin/streamlit run app.py
```

## Configuration

`UPSTOX_ACCESS_TOKEN` is read from `.env` locally, or from Streamlit secrets /
environment variables when deployed. **It is never committed** — `.env` is git-ignored.
Upstox tokens are daily-expiring and some endpoints are restricted to a static IP
configured in your Upstox account.

## Deploy

Streamlit Community Cloud: point it at this repo and `app.py`, then add
`UPSTOX_ACCESS_TOKEN` under *Settings → Secrets*. A `Dockerfile` is included for
any container host (Fly.io, Render, Cloud Run).

## Disclaimer

Research and education only. Backtested results are not a promise of future returns;
nothing here is investment advice.
