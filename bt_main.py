"""Honest walk-forward test of the Scanner's Buy % signal, with real Groww costs.

Sections:
  1. Walk-forward over EVERY viable start date in the last 10 months (hold to today)
  2. Benchmark: equal-weight buy & hold of the whole universe, identical windows
  3. Does Buy % predict anything? forward returns by score bucket, raw and alpha
  4. Exit-rule grid on identical entries
  5. Capital sensitivity
"""
import sys
import numpy as np
import pandas as pd
from bt_lib import load_data, score_panel, groww_cost, cost_pct, simulate_exit

pd.set_option("display.width", 200)

TOP_N = 5
TICKET_DEFAULT = 50_000.0   # per position; == the repo's old flat 0.348% assumption

data = load_data()
close = data["Close"]
high = data["High"]
low = data["Low"]
dates = close.index
SC = score_panel(data)
print(f"Panel: {close.shape[1]} tickers, {len(dates)} sessions "
      f"{dates[0].date()} -> {dates[-1].date()}\n")

C = {t: close[t].values for t in close.columns}
H = {t: high[t].values for t in close.columns}

# 10-month window
last = dates[-1]
win_start = last - pd.DateOffset(months=10)
win_idx = [i for i, d in enumerate(dates) if d >= win_start and i < len(dates) - 1]
print(f"10-month window: {dates[win_idx[0]].date()} -> {dates[win_idx[-1]].date()} "
      f"({len(win_idx)} viable start dates)\n")


def picks_at(i, n=TOP_N, min_score=0.0):
    row = SC.iloc[i].dropna()
    row = row[row >= min_score]
    return list(row.sort_values(ascending=False).head(n).index)


def net_pct(gross, ticket):
    bv, sv = ticket, ticket * (1 + gross / 100)
    return (sv - bv - groww_cost(bv, sv)) / bv * 100


# =====================================================================
# 1 + 2. WALK-FORWARD: buy top-N on date i, hold to the LAST session.
# =====================================================================
print("=" * 78)
print("1+2. BUY TOP-5 ON EVERY DATE IN THE LAST 10 MONTHS, HOLD TO 2026-09-04")
print("=" * 78)
rows = []
end = len(dates) - 1
for i in win_idx:
    ps = picks_at(i)
    if len(ps) < TOP_N:
        continue
    grets = []
    for t in ps:
        b, s = C[t][i], C[t][end]
        if np.isfinite(b) and np.isfinite(s):
            grets.append((s / b - 1) * 100)
    if len(grets) < TOP_N:
        continue
    strat_gross = float(np.mean(grets))
    strat_net = float(np.mean([net_pct(g, TICKET_DEFAULT) for g in grets]))
    # benchmark: equal-weight the whole universe, same window, same costs
    bh = []
    for t in close.columns:
        b, s = C[t][i], C[t][end]
        if np.isfinite(b) and np.isfinite(s):
            bh.append((s / b - 1) * 100)
    bench_gross = float(np.mean(bh))
    bench_net = net_pct(bench_gross, TICKET_DEFAULT)
    rows.append({"date": dates[i].date(), "months": (last - dates[i]).days / 30.44,
                 "strat_gross": strat_gross, "strat_net": strat_net,
                 "bench_gross": bench_gross, "bench_net": bench_net,
                 "alpha_net": strat_net - bench_net})
W = pd.DataFrame(rows)


def dist(s, label):
    return {"metric": label, "mean": s.mean(), "median": s.median(),
            "%>0": (s > 0).mean() * 100, "worst": s.min(), "best": s.max(),
            "std": s.std()}


summary = pd.DataFrame([
    dist(W.strat_net, "Top-5 Scanner, NET"),
    dist(W.bench_net, "Equal-wt universe B&H, NET"),
    dist(W.alpha_net, "Alpha (strat - bench)"),
])
print(f"\nN start dates = {len(W)}   ticket Rs {TICKET_DEFAULT:,.0f}/position "
      f"(cost {cost_pct(TICKET_DEFAULT):.3f}% round trip)")
print(summary.round(2).to_string(index=False))
print(f"\n% of start dates where Scanner BEAT buy&hold: "
      f"{(W.alpha_net > 0).mean()*100:.1f}%")

# a single date, shown only to demonstrate it is noise
one = W.sample(1, random_state=7).iloc[0]
print(f"\n[NOISE DEMO] one 'random date' {one['date']}: strat {one.strat_net:+.2f}% "
      f"vs bench {one.bench_net:+.2f}%. Range across all dates is "
      f"{W.strat_net.min():+.1f}% to {W.strat_net.max():+.1f}% -- a single date is meaningless.")

print("\nBy holding length (months from entry to today):")
W["bucket"] = pd.cut(W.months, [0, 2, 4, 6, 8, 11],
                     labels=["1-2m", "2-4m", "4-6m", "6-8m", "8-10m"])
print(W.groupby("bucket", observed=True).agg(
    n=("strat_net", "size"), strat_net=("strat_net", "mean"),
    bench_net=("bench_net", "mean"), alpha=("alpha_net", "mean"),
    pct_strat_pos=("strat_net", lambda s: (s > 0).mean() * 100)).round(2).to_string())

# =====================================================================
# 3. DOES Buy % PREDICT ANYTHING?
# =====================================================================
print("\n" + "=" * 78)
print("3. PREDICTIVE POWER OF Buy %  (full 5y, all stocks, all days)")
print("=" * 78)
for H_DAYS in (5, 10, 20):
    fwd = (close.shift(-H_DAYS) / close - 1) * 100
    sc = SC
    m = sc.notna() & fwd.notna()
    x = sc.where(m).stack()
    y = fwd.where(m).stack()
    # cross-sectional demean -> alpha (strip market drift)
    y_alpha = (fwd.where(m).sub(fwd.where(m).mean(axis=1), axis=0)).stack()
    b = pd.cut(x, [-1, 50, 60, 70, 80, 101],
               labels=["<50", "50-60", "60-70", "70-80", "80+"])
    t = pd.DataFrame({"score": x, "raw": y, "alpha": y_alpha, "b": b})
    g = t.groupby("b", observed=True).agg(n=("raw", "size"), raw_mean=("raw", "mean"),
                                          raw_med=("raw", "median"),
                                          win=("raw", lambda s: (s > 0).mean() * 100),
                                          alpha_mean=("alpha", "mean"))
    ic = t.score.corr(t.raw, method="spearman")
    ica = t.score.corr(t.alpha, method="spearman")
    print(f"\n--- forward {H_DAYS} sessions --- (rank-IC raw {ic:+.4f}, alpha {ica:+.4f})")
    print(g.round(3).to_string())
    # monotonicity check on alpha
    a = g.alpha_mean.values
    print("  alpha monotonic increasing with score?", bool(np.all(np.diff(a) > 0)))

# decile view for the same, 10d
print("\nDecile view, forward 10 sessions (10 = highest Buy %):")
fwd = (close.shift(-10) / close - 1) * 100
m = SC.notna() & fwd.notna()
x = SC.where(m).stack(); y = fwd.where(m).stack()
ya = (fwd.where(m).sub(fwd.where(m).mean(axis=1), axis=0)).stack()
dec = pd.qcut(x, 10, labels=False, duplicates="drop") + 1
t = pd.DataFrame({"dec": dec, "raw": y, "alpha": ya, "score": x})
print(t.groupby("dec").agg(n=("raw", "size"), score=("score", "mean"),
                           raw=("raw", "mean"), alpha=("alpha", "mean")).round(3).to_string())

# =====================================================================
# 4. EXIT RULES ON IDENTICAL ENTRIES
# =====================================================================
print("\n" + "=" * 78)
print("4. EXIT-RULE GRID -- identical entries (top-5 by Buy %, every 5th session, 5y)")
print("=" * 78)
STEP = 5
entry_idx = [i for i in range(70, len(dates) - 21, STEP)]
entries = []
for i in entry_idx:
    for t in picks_at(i):
        if np.isfinite(C[t][i]):
            entries.append((i, t))
print(f"{len(entries)} entries over {len(entry_idx)} rebalance dates "
      f"({dates[entry_idx[0]].date()} -> {dates[entry_idx[-1]].date()})")

# market drift benchmark: mean universe return over the same holding periods
def bench_for(i, hold):
    j = min(i + hold, len(dates) - 1)
    r = []
    for t in close.columns:
        b, s = C[t][i], C[t][j]
        if np.isfinite(b) and np.isfinite(s):
            r.append((s / b - 1) * 100)
    return float(np.mean(r))


def run_exit(hold, trail=None, tp=None, sl=None, ticket=TICKET_DEFAULT):
    g, d, reasons, alp = [], [], [], []
    for i, t in entries:
        gr, dd, rs = simulate_exit(H[t], None, C[t], i, hold, trail, tp, sl)
        if not np.isfinite(gr):
            continue
        g.append(gr); d.append(dd); reasons.append(rs)
        alp.append(gr - bench_for(i, dd))
    g = np.array(g)
    net = np.array([net_pct(v, ticket) for v in g])
    return {"gross%": g.mean(), "net%": net.mean(), "net_med%": np.median(net),
            "win%": (net > 0).mean() * 100, "avg_days": float(np.mean(d)),
            "alpha_gross%": float(np.mean(alp)), "worst%": g.min(), "best%": g.max(),
            "n": len(g), "sharpe": net.mean() / net.std() * np.sqrt(252 / max(np.mean(d), 1))}


results = []
print("\n(a) FIXED HOLD")
for h in (5, 7, 10, 15, 20):
    r = run_exit(h); r["rule"] = f"hold {h}d"; results.append(r)
print("\n(b) TRAILING STOP (on closes, 20d max hold)")
for tr in (2, 3, 4, 6):
    r = run_exit(20, trail=tr); r["rule"] = f"trail {tr}% /20d"; results.append(r)
print("\n(c) STOP-LOSS (fixed, on closes, 20d max hold)")
for s in (2, 3, 5, 8):
    r = run_exit(20, sl=s); r["rule"] = f"SL {s}% /20d"; results.append(r)
print("\n(d) TAKE-PROFIT (on highs, 20d max hold)")
for p in (2, 3, 5, 8):
    r = run_exit(20, tp=p); r["rule"] = f"TP {p}% /20d"; results.append(r)
print("\n(e) COMBINED")
for p, s in ((5, 3), (5, 5), (8, 5), (3, 3)):
    r = run_exit(20, tp=p, sl=s); r["rule"] = f"TP{p}/SL{s} /20d"; results.append(r)

R = pd.DataFrame(results).set_index("rule")
R = R[["n", "gross%", "net%", "net_med%", "alpha_gross%", "win%", "avg_days", "sharpe", "worst%", "best%"]]
print("\n" + R.round(3).to_string())
print(f"\n(net at Rs {TICKET_DEFAULT:,.0f}/position = {cost_pct(TICKET_DEFAULT):.3f}% round trip)")
print("alpha_gross% = trade return minus the equal-weight universe over the SAME days held.")

# =====================================================================
# 5. CAPITAL SENSITIVITY
# =====================================================================
print("\n" + "=" * 78)
print("5. CAPITAL SENSITIVITY (best fixed-hold rule vs buy&hold)")
print("=" * 78)
base = run_exit(20)
gross_mean = base["gross%"]
print(f"\nUsing the 20-day hold entries: mean GROSS {gross_mean:+.3f}%/trade, "
      f"avg {base['avg_days']:.1f} days held\n")
cap_rows = []
for cap in (10_000, 50_000, 200_000, 1_000_000):
    for n in (3, 5, 10):
        tk = cap / n
        c = cost_pct(tk)
        cap_rows.append({"capital": cap, "N_pos": n, "ticket": tk,
                         "cost%_roundtrip": c, "gross%": gross_mean,
                         "net%_per_trade": gross_mean - c,
                         "net_Rs_per_trade": (gross_mean - c) / 100 * tk})
CS = pd.DataFrame(cap_rows)
print(CS.round(3).to_string(index=False))

W.to_csv("bt_walkforward.csv", index=False)
R.to_csv("bt_exits.csv")
print("\nwrote bt_walkforward.csv, bt_exits.csv")
