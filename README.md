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

Measured on 3.5 years of Upstox 15-minute bars (22,476 bars × 124 stocks),
walk-forward, out-of-sample from Oct 2024 to Aug 2026 (1.81 years).

**The benchmark that matters:** holding the equal-weight universe over the same
window returned **+10.2% total, +5.5% CAGR** with zero trading. Any strategy has
to beat that after costs, or it is an expensive way to buy the market.

| | net per trade | trades | Sharpe | CAGR |
|---|---|---|---|---|
| Buy & hold equal-weight | — | 0 | — | +5.5% |
| Rule: RSI<35 above SMA78 | -0.272% | 3,129 | -4.33 | loses |
| ML, first version | +0.004% | 2,692 | 0.04 | flat |
| **ML, current** | **+0.201%** | **605** | **1.04** | **+12.7%** |

Three findings drove the difference between "flat" and "beats the market":

**1. The alpha is far smaller than it looks.** Demeaning each bar's forward
returns cross-sectionally — subtracting what the whole universe did — separates
stock selection from market drift:

| Horizon | Universe drift | Top decile raw | Top decile **alpha** |
|---|---|---|---|
| 1 session | +0.027% | +0.337% | +0.034% |
| 8 sessions | +0.204% | +0.990% | +0.121% |
| 15 sessions | +0.425% | +0.953% | +0.197% |

Most of the top decile's return is beta. Real stock-selection edge peaks around
+0.20% — still below the 0.348% round-trip cost. **Cost, not signal, is the
binding constraint.**

**2. So turnover is the main lever.** Trading every bar pays 0.348% to harvest at
most 0.12%. Rebalancing once per session instead cut turnover ~25× and took the
strategy from flat to profitable. Fewer, longer, higher-conviction trades.

**3. Cross-sectional rank features earn their place.** In a ranking problem what
matters is a stock's value *relative to the universe right now*, not its absolute
level. Adding per-bar percentile ranks alongside the raw features flipped
out-of-sample rank-IC from -0.002 to +0.004 and net return from -0.011%/trade to
+0.070%/trade. Ranks *instead of* raw values did worse — both are needed.

Things that were tested and **did not** work, kept here so they are not retried:

- **Intraday horizons.** At 2 hours the top decile earns 3bps gross against 15bps
  of costs. Structurally unprofitable; the backtest confirmed -0.17%/trade.
- **Trailing stops**, swept at 1.5/2.5/4/6%. Every width reduced returns versus a
  plain horizon exit (1.5% trail → -0.19%/trade).
- **Long-short.** The shorted bottom decile *rose* +0.16% over the holding
  period — the model separates high-volatility names from low-volatility ones,
  not winners from losers. Adding the short leg doubled costs and took
  -0.23%/trade to -0.37%/trade.
- **Long-lookback momentum / relative-strength features.** They raise average
  rank-IC (+0.0065, t=5.5) but halve the decile spread and turn net return
  sharply negative — better mid-distribution ordering, worse extreme picks.

### Honest limitations

- The out-of-sample window is **1.81 years and 605 trades**. Suggestive, not proven.
- The edge is **fragile in width**: top-5 returns +12.7% CAGR, but top-10 at a
  lower threshold *loses* (-6.8% CAGR). A result that flips sign with a mild
  parameter change is not yet robust.
- Win rate is 47.8% — profit comes from asymmetry, not accuracy, and the worst
  single trade was -15.4%.
- Costs are modelled at a flat 0.348% with no market-impact or slippage term.

## Configuration## Configuration

`UPSTOX_ACCESS_TOKEN` is read from `.env` locally, or from Streamlit secrets /
environment variables when deployed. **It is never committed** — `.env` is git-ignored.
Upstox tokens are daily-expiring and some endpoints are restricted to a static IP
configured in your Upstox account.

## Deploy

See **[DEPLOY.md](DEPLOY.md)** for Streamlit Community Cloud, Docker and Fly.io,
including how the deployed app gets its data (it reads the corpus from R2 that
the nightly GitHub Action writes).

## Disclaimer

Research and education only. Backtested results are not a promise of future returns;
nothing here is investment advice.
