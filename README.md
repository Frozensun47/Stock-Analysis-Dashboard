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
