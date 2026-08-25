#!/usr/bin/env bash
# Daily data collection. Add to cron, e.g. 20:00 IST on weekdays:
#   0 20 * * 1-5 /Volumes/ExFAT/my-projects/analysis/stockdash/daily_sync.sh >> /tmp/stockdash.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY bars_db.py sync 2          # 1-minute bars -> cache/bars.sqlite (yfinance 7d window)
$PY news_db.py sync            # RSS news corpus -> cache/news.sqlite
$PY fundamentals.py sync       # statements + ratios -> cache/fundamentals.sqlite
[ -n "${AWS_ACCESS_KEY_ID:-}" ] && $PY storage.py push || true
echo "daily sync complete: $(date)"
