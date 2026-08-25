"""Backtest: 1 purchase/day (best-scoring stock), 1% dynamic trailing stop.
Exit when Low breaches trail (high-since-entry * 0.99); gap-downs exit at Open.
Win = exit price > entry price."""
import numpy as np
import pandas as pd
from engine import fetch_prices, scan

def run(data, weights=None, min_score=60, trail_pct=1.0, max_hold=15, start=70):
    close, high, low, opn = data["Close"], data["High"], data["Low"], data["Open"]
    trades = []
    for i in range(start, len(close) - 1):
        day = scan(data, asof=i, weights=weights)
        picks = day[day["Buy %"] >= min_score]
        if picks.empty:
            continue
        r = picks.iloc[0]                      # max 1 purchase per day: the top pick
        t = r["Symbol"] + ".NS"
        entry = close[t].iloc[i]
        if not np.isfinite(entry):
            continue
        peak, exit_px, exit_day = entry, None, None
        for j in range(i + 1, min(i + 1 + max_hold, len(close))):
            o, h, l = opn[t].iloc[j], high[t].iloc[j], low[t].iloc[j]
            if not np.isfinite(l):
                continue
            stop = peak * (1 - trail_pct / 100)
            if o <= stop:                       # gapped below the trail
                exit_px, exit_day = o, j
                break
            if l <= stop:
                exit_px, exit_day = stop, j
                break
            peak = max(peak, h)
        if exit_px is None:                     # time-exit at last close
            exit_day = min(i + max_hold, len(close) - 1)
            exit_px = close[t].iloc[exit_day]
        trades.append({"date": close.index[i].date(), "Symbol": r["Symbol"],
                       "Buy %": r["Buy %"], "entry": round(float(entry), 2),
                       "exit": round(float(exit_px), 2), "days_held": exit_day - i,
                       "ret": (exit_px / entry - 1) * 100})
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return tdf, {}
    stats = {"trades": len(tdf), "win_rate": round((tdf.ret > 0).mean() * 100, 1),
             "avg_ret": round(tdf.ret.mean(), 2), "total_ret_sum": round(tdf.ret.sum(), 1),
             "avg_days": round(tdf.days_held.mean(), 1)}
    return tdf, stats

if __name__ == "__main__":
    data = fetch_prices()
    for ms in (60, 70, 75, 80):
        tdf, s = run(data, min_score=ms)
        print(f"min_score={ms}: {s}")
