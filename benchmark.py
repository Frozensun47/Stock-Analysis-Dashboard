"""The frozen benchmark every strategy is scored against.

Nothing here may be tuned. A strategy that changes the harness to look good is
cheating, so the split dates, the cost model, the capital and the benchmark are
constants — the only thing an experiment may vary is the strategy function.

CONTRACT
    A strategy is a function  f(close, high, low, open_, volume, i) -> list[str]
    returning the tickers to hold as of bar i. It may only read data up to and
    including i; the harness enters at bar i+1's OPEN, which is the first price
    actually obtainable after a signal.

SPLITS (frozen 2026-09-04)
    TRAIN  2021-09-06 .. 2024-12-31   for fitting and for choosing parameters
    TEST   2025-01-01 .. 2026-08-24   touched only to report a final number
    A result quoted on TRAIN is a hypothesis. Only TEST counts as a result.

COSTS  real Groww delivery, per scrip round trip. Several charges are flat
    rupees, so cost as a percentage depends on ticket size — a Rs 2,000 ticket
    pays 1.26% while a Rs 50,000 ticket pays 0.35%. Backtests that assume a flat
    percentage silently flatter small accounts.

BENCHMARK  equal-weight buy & hold of the same universe over the same window,
    paying the same entry cost once. Beating zero is not the bar; beating this is.
"""
import json, os, sqlite3, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "cache", "experiments.sqlite")
LOG_MD = os.path.join(HERE, "EXPERIMENTS.md")

TRAIN = ("2021-09-06", "2024-12-31")
TEST = ("2025-01-01", "2026-08-24")
CAPITAL = 200_000.0     # large enough that flat charges are not the whole story
N_POS = 5
REBALANCE = 5           # trading days between rebalances

def groww_cost(buy_value, sell_value):
    """Real delivery cost in rupees for one scrip, one round trip."""
    bro = min(20.0, 0.001 * buy_value) + min(20.0, 0.001 * sell_value)
    stt = 0.001 * buy_value + 0.001 * sell_value
    exch = 0.0000297 * (buy_value + sell_value)
    sebi = 0.000001 * (buy_value + sell_value)
    stamp = 0.00015 * buy_value
    dp = 13.5
    gst = 0.18 * (bro + exch + dp)
    return bro + stt + exch + sebi + stamp + dp + gst


def load(period="5y"):
    d = pd.read_pickle(os.path.join(HERE, "cache", f"prices_{period}.pkl"))
    return {k: d[k] for k in ["Open", "High", "Low", "Close", "Volume"]}


def _window(idx, split):
    a, b = pd.Timestamp(split[0]), pd.Timestamp(split[1])
    return np.where((idx >= a) & (idx <= b))[0]


def run(strategy, data, split, capital=CAPITAL, n_pos=N_POS, rebalance=REBALANCE,
        max_hold=20, trail=4.0, warmup=250):
    """Score one strategy on a real, time-indexed portfolio.

    Accounting note that matters: an earlier version compounded each trade's
    return on capital/n_pos and multiplied them together. That rewards strategies
    purely for TRADING MORE — 524 random trades "compounded" to +41.8% CAGR and
    beat every real strategy. Returns must accrue over TIME, not per trade.

    So this holds n_pos slots against real cash. A slot is filled only if cash is
    available, positions are marked to market daily, and the equity curve is the
    portfolio's value on each date. Trading more cannot manufacture return.
    """
    close, opn = data["Close"], data["Open"]
    idx = close.index
    bars = _window(idx, split)
    bars = bars[bars >= warmup]
    if len(bars) == 0:
        raise ValueError("empty window")
    lo, hi = int(bars[0]), int(bars[-1])

    cash = capital
    slots = []          # dicts: symbol, qty, entry, entry_bar, peak, exit_bar
    trades, equity = [], []
    rebal_bars = set(bars[::rebalance].tolist())

    for i in range(lo, hi + 1):
        px_now = close.iloc[i]

        # --- exits, evaluated on today's close ---
        keep = []
        for p in slots:
            c = px_now.get(p["symbol"], np.nan)
            if not np.isfinite(c):
                keep.append(p); continue
            p["peak"] = max(p["peak"], c)
            held = i - p["entry_bar"]
            why = None
            if trail and c <= p["peak"] * (1 - trail / 100):
                why = "trail"
            elif held >= max_hold:
                why = "time"
            if why:
                bv, sv = p["qty"] * p["entry"], p["qty"] * c
                cost = groww_cost(bv, sv)
                cash += sv - groww_cost(0.0, sv)     # sell-side charges only here
                trades.append(dict(symbol=p["symbol"], entry=p["entry"], exit=float(c),
                                   qty=p["qty"], days=held, reason=why,
                                   gross=sv - bv, cost=cost, net=sv - bv - cost,
                                   net_pct=(sv - bv - cost) / bv * 100))
            else:
                keep.append(p)
        slots = keep

        # --- entries at tomorrow's open, decided on today's close ---
        if i in rebal_bars and len(slots) < n_pos and i + 1 <= hi:
            have = {p["symbol"] for p in slots}
            try:
                picks = [t for t in strategy(data, i) if t not in have]
            except Exception as e:
                raise RuntimeError(f"strategy failed at bar {i}: {e}") from e
            for t in picks:
                if len(slots) >= n_pos:
                    break
                e = opn[t].iloc[i + 1] if t in opn.columns else np.nan
                if not np.isfinite(e) or e <= 0:
                    continue
                budget = min(cash, capital / n_pos)
                qty = int(budget // e)
                if qty < 1:
                    continue
                bv = qty * e
                cash -= bv + groww_cost(bv, 0.0)     # buy-side charges only here
                slots.append(dict(symbol=t, qty=qty, entry=float(e), entry_bar=i + 1,
                                  peak=float(e)))

        mtm = sum(p["qty"] * close[p["symbol"]].iloc[i] for p in slots
                  if np.isfinite(close[p["symbol"]].iloc[i]))
        equity.append(cash + mtm)

    eq = pd.Series(equity, index=idx[lo:hi + 1]).ffill()
    td = pd.DataFrame(trades)
    years = len(eq) / 250
    total = (eq.iloc[-1] / capital - 1) * 100
    dd = float((eq / eq.cummax() - 1).min() * 100)
    dr = eq.pct_change().dropna()

    # benchmark: hold the equal-weight universe over the identical window, one entry cost
    w = close.iloc[lo:hi + 1]
    bh = (1 + w.pct_change().mean(axis=1).fillna(0)).cumprod()
    bh_total = (bh.iloc[-1] - 1) * 100 - groww_cost(capital, capital * bh.iloc[-1]) / capital * 100
    bh_dd = float((bh / bh.cummax() - 1).min() * 100)

    m = dict(
        trades=len(td),
        net_pct=float(td.net_pct.mean()) if len(td) else np.nan,
        total_pct=float(total),
        cagr=float(((eq.iloc[-1] / capital) ** (1 / years) - 1) * 100),
        sharpe=float(dr.mean() / dr.std() * np.sqrt(250)) if dr.std() else 0.0,
        win=float((td.net_pct > 0).mean() * 100) if len(td) else np.nan,
        worst=float(td.net_pct.min()) if len(td) else np.nan,
        max_dd=dd,
        cost_drag=float(td.cost.sum() / capital * 100) if len(td) else 0.0,
        exposure=float(np.mean([1 - (c / capital) for c in [cash]])) if False else np.nan,
        bench_total_pct=float(bh_total),
        bench_cagr=float(((1 + bh_total / 100) ** (1 / years) - 1) * 100),
        bench_max_dd=bh_dd,
        alpha_total_pct=float(total - bh_total),
        years=float(years), split=f"{split[0]}..{split[1]}")
    m["equity"] = eq
    m["bench_curve"] = bh
    return m, td


# ---------------- experiment log ----------------
DDL = """
CREATE TABLE IF NOT EXISTS experiments (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, name TEXT, split TEXT,
  hypothesis TEXT, trades INT, net_pct REAL, total_pct REAL, cagr REAL,
  bench_cagr REAL, alpha_total_pct REAL, sharpe REAL, win REAL, worst REAL,
  cost_drag REAL, max_dd REAL, bench_max_dd REAL, verdict TEXT, notes TEXT);
"""

def log(name, m, hypothesis="", verdict="", notes=""):
    """Record an experiment — including the ones that fail. That is the point."""
    con = sqlite3.connect(DB); con.executescript(DDL)
    con.execute(
        "INSERT INTO experiments (ts,name,split,hypothesis,trades,net_pct,total_pct,"
        "cagr,bench_cagr,alpha_total_pct,sharpe,win,worst,cost_drag,max_dd,"
        "bench_max_dd,verdict,notes) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M"), name, m.get("split"), hypothesis,
         m.get("trades"), m.get("net_pct"), m.get("total_pct"), m.get("cagr"),
         m.get("bench_cagr"), m.get("alpha_total_pct"), m.get("sharpe"),
         m.get("win"), m.get("worst"), m.get("cost_drag"), m.get("max_dd"),
         m.get("bench_max_dd"), verdict, notes))
    con.commit(); con.close()

def table():
    con = sqlite3.connect(DB); con.executescript(DDL)
    df = pd.read_sql("SELECT * FROM experiments ORDER BY id", con); con.close()
    return df

def report(m, name):
    print(f"\n{name}  [{m['split']}]")
    if not m.get("trades"):
        print("  no trades"); return
    print(f"  trades {m['trades']:<5} net/trade {m['net_pct']:+.3f}%  "
          f"win {m['win']:.1f}%  worst {m['worst']:+.1f}%  cost drag {m['cost_drag']:.2f}% of capital")
    print(f"  strategy  {m['total_pct']:+7.1f}% total  {m['cagr']:+6.1f}% CAGR  "
          f"Sharpe {m['sharpe']:5.2f}  maxDD {m['max_dd']:.1f}%")
    print(f"  benchmark {m['bench_total_pct']:+7.1f}% total  {m['bench_cagr']:+6.1f}% CAGR  "
          f"{'':13} maxDD {m['bench_max_dd']:.1f}%")
    verdict = "BEATS benchmark" if m["alpha_total_pct"] > 0 else "loses to benchmark"
    print(f"  alpha {m['alpha_total_pct']:+.1f}%  -> {verdict}")
