# Deploying the dashboard

The app is a standard Streamlit application. The only wrinkle is data: everything
under `cache/` (SQLite corpus, the 112 MB 15-minute panel, the trained model) is
git-ignored, so a fresh deploy starts with none of it.

## Option A — Streamlit Community Cloud (free, easiest)

1. Go to https://share.streamlit.io → **New app** → pick
   `Frozensun47/Stock-Analysis-Dashboard`, branch `main`, file `app.py`.
2. **Advanced settings → Secrets**, paste:
   ```toml
   UPSTOX_ACCESS_TOKEN = "..."
   S3_ENDPOINT_URL = "https://<accountid>.r2.cloudflarestorage.com"
   S3_BUCKET = "stockdash"
   AWS_ACCESS_KEY_ID = "..."
   AWS_SECRET_ACCESS_KEY = "..."
   ```
3. Deploy. The Scanner and Backtest tabs work immediately (they pull daily bars
   from yfinance at runtime). The 15m Model, News Corpus and Fundamentals tabs
   need the corpus — they show a "run the sync" message until it is present.
4. To populate them, run `python storage.py push` locally once, then have the app
   pull on boot. Streamlit Cloud's disk is ephemeral, so the pull re-runs on each
   cold start.

**Caveat:** Community Cloud gives ~1 GB RAM. The full 15-minute panel is 112 MB on
disk and several times that in memory, so keep `live_signals()` (which reads only
the last bar) on that instance and run backtests locally.

## Option B — Docker, anywhere (Fly.io, Render, Cloud Run)

```bash
docker build -t stockdash .
docker run -p 8501:8501 --env-file .env stockdash
```

Fly.io, with a volume so the corpus survives restarts:
```bash
fly launch --no-deploy
fly volumes create data --size 3
fly secrets set UPSTOX_ACCESS_TOKEN=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
fly deploy
```
Mount the volume at `/app/cache` in `fly.toml`.

## Data refresh in production

`.github/workflows/daily-sync.yml` runs at 20:00 IST on weekdays: it pulls the
existing corpus from R2, collects the day's bars and news (fundamentals on
Fridays), and pushes the corpus back. The deployed app reads the same bucket, so
it never has to scrape anything itself.

## Token expiry

Upstox access tokens expire daily (~03:30 IST) and some endpoints are restricted
to a static IP set in your Upstox account — the historical-candle endpoint used
here works without either, but `/user/profile` and live quotes do not. Refresh
the token in Streamlit secrets (or the GitHub secret) when live data is needed.
