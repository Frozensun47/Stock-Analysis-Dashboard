"""What ₹10,000 would have done over the last month following the model.

Trained on everything up to the start date, then run forward — no peeking. Costs
are modelled twice: the flat 0.348% used in the backtest, and the REAL Groww
delivery cost at this ticket size, which is very different for small amounts
because several charges are flat rupees rather than percentages.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import model_15m as M
from upstox_data import load_15m

CAPITAL, K = 10_000.0, 5
panel = load_15m(); c, o = panel["Close"], panel["Open"]
end = c.index[-1]
start = end - pd.Timedelta(days=30)

def groww_costs(value, sell_value):
    """Groww delivery equity, per round trip on one scrip (₹, as of 2026 rates)."""
    bro = min(20.0, 0.001 * value) + min(20.0, 0.001 * sell_value)  # 0.1% or ₹20 cap
    stt = 0.001 * value + 0.001 * sell_value
    exch = 0.0000297 * (value + sell_value)
    sebi = 0.000001 * (value + sell_value)
    stamp = 0.00015 * value                     # buy side only
    dp = 13.5                                   # flat, per sell per scrip
    gst = 0.18 * (bro + exch + dp)
    return bro + stt + exch + sebi + stamp + dp + gst

F, fwd = M.build_features(panel)
tab = M.flatten(F, fwd, c)
split = c.index.get_loc(c.index[c.index <= start][-1])
train = tab[(tab.i < split - M.HORIZON) & tab.fwd.notna()]
print(f"train: {len(train):,} rows up to {c.index[split]:%Y-%m-%d} (nothing after)")
model = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.06,
                                      random_state=0).fit(train[M.FEATS], train.fwd)
test = tab[tab.i >= split].copy()
test["pred"] = model.predict(test[M.FEATS])

# same rule as the backtest: rebalance every 25 bars, top-5, pred > 1.0%
ent = []
for i, g in test[test.i % M.REBALANCE_EVERY == 0].groupby("i"):
    for _, r in g.nlargest(K, "pred").query("pred > @M.MIN_PRED").iterrows():
        ent.append((int(r.i), c.columns[int(r.j)]))
print(f"signals in the window: {len(ent)}")

N = len(c); rows = []
busy = {}
for i, t in sorted(ent):
    if i < busy.get(t, -1):
        continue
    e = o[t].iloc[i + 1] if i + 1 < N else np.nan
    if not np.isfinite(e):
        continue
    stop = min(i + M.HORIZON, N - 1)
    x = c[t].iloc[stop]
    if not np.isfinite(x):
        continue
    busy[t] = stop
    qty = int((CAPITAL / K) // e)
    if qty < 1:
        rows.append(dict(sym=t.replace(".NS",""), entry=e, exit=x, qty=0, gross=0.0,
                         cost=0.0, net=0.0, note="unaffordable"))
        continue
    buy_v, sell_v = qty * e, qty * x
    cost = groww_costs(buy_v, sell_v)
    rows.append(dict(sym=t.replace(".NS",""), entry=round(e,1), exit=round(x,1), qty=qty,
                     gross=sell_v - buy_v, cost=cost, net=sell_v - buy_v - cost,
                     pct=(x/e-1)*100, note=""))
d = pd.DataFrame(rows)
print(f"\nwindow: {c.index[split]:%Y-%m-%d} → {end:%Y-%m-%d}")
if d.empty:
    print("no completed trades"); raise SystemExit
done = d[d.qty > 0]
print(done[["sym","entry","exit","qty","pct","gross","cost","net"]].round(2).to_string(index=False))
print(f"\n--- ₹{CAPITAL:,.0f} capital, {K} positions of ₹{CAPITAL/K:,.0f} ---")
print(f"gross P&L        ₹{done.gross.sum():+,.2f}")
print(f"real Groww costs ₹{-done.cost.sum():,.2f}   ({done.cost.sum()/ (CAPITAL) *100:.2f}% of capital)")
print(f"NET P&L          ₹{done.net.sum():+,.2f}   ({done.net.sum()/CAPITAL*100:+.2f}%)")
ideal = done.gross.sum() - (done.qty*done.entry).sum()*M.COST/100
print(f"\nfor comparison, at the backtest's flat {M.COST}% cost: ₹{ideal:+,.2f} "
      f"({ideal/CAPITAL*100:+.2f}%)")
print(f"cost per trade: real ₹{done.cost.mean():.1f} on a ₹{(done.qty*done.entry).mean():,.0f} "
      f"ticket = {(done.cost/(done.qty*done.entry)).mean()*100:.2f}% vs {M.COST}% assumed")
# benchmark
oos = c.loc[c.index[split]:]
bh = (1 + oos.pct_change().mean(axis=1).fillna(0)).cumprod().iloc[-1]
print(f"\nbuy & hold the same ₹{CAPITAL:,.0f} in the equal-weight universe: "
      f"₹{CAPITAL*(bh-1):+,.2f} ({(bh-1)*100:+.2f}%)")

# ---- how does this scale with capital? ----
print("\n\n=== the same 18 trades at different capital levels ===")
print(f"{'capital':>10} {'per pos':>9} {'gross':>9} {'costs':>9} {'net ₹':>9} {'net %':>8} {'cost/trade':>11}")
for cap in [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]:
    g = t_cost = 0.0
    for _, r in d.iterrows():
        e, x = r.entry, r.exit
        qty = int((cap / K) // e)
        if qty < 1:
            continue
        bv, sv = qty * e, qty * x
        g += sv - bv
        t_cost += groww_costs(bv, sv)
    net = g - t_cost
    tickets = [int((cap/K)//r.entry)*r.entry for _, r in d.iterrows() if int((cap/K)//r.entry) >= 1]
    pct = t_cost / sum(tickets) * 100 if tickets else float("nan")
    print(f"{cap:>10,} {cap/K:>9,.0f} {g:>+9,.0f} {-t_cost:>9,.0f} {net:>+9,.0f} "
          f"{net/cap*100:>+8.2f} {pct:>10.2f}%")
print("\nThe strategy's measured edge is ~0.20%/trade net at the backtest's assumed cost.")
print("Below that, cost/trade must fall under the gross edge of "
      f"{(d.gross.sum()/sum((d.qty*d.entry).where(d.qty>0).dropna()))*100:.2f}% for this window.")
