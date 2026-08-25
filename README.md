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

## What the 15-minute model actually found

A rank-IC sweep over prediction horizons (out-of-sample, walk-forward) settled the
design. The edge is real but small, and it only clears trading costs once the
holding period spans days:

| Horizon | rank-IC | t | Top-decile fwd return (gross) |
|---|---|---|---|
| 4 bars (1h) | +0.003 | 0.5 | +0.04% |
| 8 bars (2h) | +0.015 | 2.7 | +0.03% |
| 25 bars (1 session) | +0.033 | 6.8 | +0.33% |
| 50 bars (2 sessions) | +0.040 | 7.7 | +0.60% |
| 100 bars (4 sessions) | +0.031 | 5.4 | +1.03% |

At a 2-hour horizon the top decile earns 3bps gross against ~15bps of intraday
costs — a guaranteed loser, and the intraday backtest confirmed it (-0.17%/trade).
So the model keeps the *15-minute features* (VWAP deviation, intraday range
position, time-of-day, volume surge) but predicts a **4-session** forward return.

Trailing stops were swept at 1.5/2.5/4/6% and none: every stop width reduced
returns (1.5% trail → -0.19%/trade; no stop → +0.22%/trade, Sharpe 1.09), so the
horizon exit alone is what the evidence supports.

**Caveat:** those numbers come from ~3 months of yfinance 15m data, which is all
that free API serves. The Upstox pipeline exists to re-run this on multi-year
history — treat the single-quarter result as directional, not validated.

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
