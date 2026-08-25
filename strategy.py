"""Final validated strategy (5y backtest, next-day-open entry, net of Groww costs):
+0.57%/trade avg over 843 trades, profitable in every calendar year 2022-2026.
Buy at NEXT DAY'S OPEN after the signal. No Nifty regime filter (it hurt),
no take-profit, no gap filter, top-1 pick per day (all tested; all worse).
ML models (gradient-boosted regressor and classifier, walk-forward) were tested
and LOST to these rules — see ml_model.py.

Rules:
- Entry (max 1 purchase/day): the stock deepest in a dip within an uptrend —
  price above its 50-day SMA, RSI(14) < 40, lowest RSI wins.
- Exit: take-profit at +0.6%, OR 1% dynamic trailing stop (evaluated on
  closing prices — stop = highest close since entry * 0.99), OR time exit
  after 15 sessions.
- Win = exit above entry.
"""
import numpy as np
import pandas as pd
from engine import fetch_prices, rsi

# Walk-forward validated over 5y net of Groww costs (~0.35% round trip):
# no take-profit (let winners run), 4% trailing stop on closes, 15-day max hold.
TP_PCT, TRAIL_PCT, MAX_HOLD, RSI_MAX = None, 4.0, 15, 40

def pick(close, i, rsi_max=RSI_MAX):
    """Best mean-reversion candidate as of row i: uptrend + most oversold."""
    best, bestrsi = None, 100
    for t in close.columns:
        s = close[t].iloc[: i + 1].dropna()
        if len(s) < 60:
            continue
        r = rsi(s).iloc[-1]
        if np.isfinite(r) and s.iloc[-1] > s.tail(50).mean() and r < rsi_max and r < bestrsi:
            best, bestrsi = t.replace(".NS", ""), r
    return best, round(bestrsi, 1) if best else None

def backtest(data, tp=TP_PCT, trail=TRAIL_PCT, max_hold=MAX_HOLD, start=70):
    close, high, opn = data["Close"], data["High"], data["Open"]
    trades = []
    for i in range(start, len(close) - 1):
        sym, r = pick(close, i)
        if sym is None:
            continue
        t = sym + ".NS"
        entry = close[t].iloc[i]
        if not np.isfinite(entry):
            continue
        peak, ep, ed, reason = entry, None, None, "time"
        for j in range(i + 1, min(i + 1 + max_hold, len(close))):
            h, c = high[t].iloc[j], close[t].iloc[j]
            if not np.isfinite(c):
                continue
            if h >= entry * (1 + tp / 100):
                ep, ed, reason = entry * (1 + tp / 100), j, "take-profit"
                break
            if c <= peak * (1 - trail / 100):
                ep, ed, reason = c, j, "trail-stop"
                break
            peak = max(peak, c)
        if ep is None:
            ed = min(i + max_hold, len(close) - 1)
            ep = close[t].iloc[ed]
        trades.append({"date": close.index[i].date(), "Symbol": sym, "rsi": r,
                       "entry": round(float(entry), 2), "exit": round(float(ep), 2),
                       "days": ed - i, "reason": reason, "ret": (ep / entry - 1) * 100})
    tdf = pd.DataFrame(trades)
    stats = {} if tdf.empty else {
        "trades": len(tdf), "win_rate": round((tdf.ret > 0).mean() * 100, 1),
        "avg_ret": round(tdf.ret.mean(), 3), "total_ret_sum": round(tdf.ret.sum(), 1),
        "avg_days": round(tdf.days.mean(), 1)}
    return tdf, stats

def todays_pick(data):
    close = data["Close"]
    sym, r = pick(close, len(close) - 1)
    return sym, r

if __name__ == "__main__":
    data = fetch_prices()
    tdf, stats = backtest(data)
    print("Backtest:", stats)
    print(tdf.reason.value_counts().to_string())
    sym, r = todays_pick(data)
    print(f"\nToday's pick: {sym} (RSI {r})" if sym else "\nToday's pick: none (no uptrend stock with RSI<40)")
