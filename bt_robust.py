"""Robustness: significance of the bucket alphas, a random-pick control,
full-5y walk-forward, and non-overlapping start dates."""
import numpy as np, pandas as pd
from scipy import stats
from bt_lib import load_data, score_panel, groww_cost, cost_pct, simulate_exit

data = load_data(); close = data["Close"]; high = data["High"]; dates = close.index
SC = score_panel(data)
C = {t: close[t].values for t in close.columns}
H = {t: high[t].values for t in close.columns}
TICKET = 50_000.0
net = lambda g, tk=TICKET: (tk*(1+g/100) - tk - groww_cost(tk, tk*(1+g/100)))/tk*100

print("="*78); print("A. IS THE BUCKET ALPHA STATISTICALLY DISTINGUISHABLE FROM ZERO?")
print("="*78)
print("(t-stats are OVERSTATED: overlapping windows + cross-sectional correlation.")
print(" Treat |t|<3 as indistinguishable from zero.)\n")
for HD in (5, 10, 20):
    fwd = (close.shift(-HD)/close - 1)*100
    m = SC.notna() & fwd.notna()
    a = fwd.where(m).sub(fwd.where(m).mean(axis=1), axis=0)
    x = SC.where(m).stack(); ya = a.stack()
    b = pd.cut(x, [-1,50,60,70,80,101], labels=["<50","50-60","60-70","70-80","80+"])
    t = pd.DataFrame({"alpha": ya, "b": b})
    g = t.groupby("b", observed=True).alpha.agg(["size","mean","std"])
    g["t_stat"] = g["mean"]/(g["std"]/np.sqrt(g["size"]))
    # top minus bottom spread
    hi = t[t.b=="80+"].alpha; lo = t[t.b=="<50"].alpha
    ts, pv = stats.ttest_ind(hi, lo, equal_var=False)
    print(f"-- fwd {HD}d --"); print(g.round(3).to_string())
    print(f"   80+ minus <50 spread = {hi.mean()-lo.mean():+.4f}%  t={ts:+.2f} p={pv:.3f}\n")

print("="*78); print("B. RANDOM-PICK CONTROL: does picking 5 stocks at random do worse?")
print("="*78)
STEP = 5
eidx = [i for i in range(70, len(dates)-21, STEP)]
def picks(i, n=5):
    r = SC.iloc[i].dropna(); return list(r.sort_values(ascending=False).head(n).index)

def run(ents, hold=20):
    g = []
    for i, t in ents:
        gr, dd, rs = simulate_exit(H[t], None, C[t], i, hold)
        if np.isfinite(gr): g.append(gr)
    g = np.array(g); nt = np.array([net(v) for v in g])
    return g.mean(), nt.mean(), (nt>0).mean()*100, len(g)

sig = [(i, t) for i in eidx for t in picks(i) if np.isfinite(C[t][i])]
print(f"Top-5 by Buy %:      gross {run(sig)[0]:+.3f}%  net {run(sig)[1]:+.3f}%  win {run(sig)[2]:.1f}%  n={run(sig)[3]}")
rng = np.random.default_rng(42); cols = list(close.columns); rand_res = []
for trial in range(30):
    ents = []
    for i in eidx:
        for t in rng.choice(cols, 5, replace=False):
            if np.isfinite(C[t][i]): ents.append((i, t))
    rand_res.append(run(ents))
rr = np.array(rand_res)
print(f"Random 5 (30 trials): gross {rr[:,0].mean():+.3f}% (sd {rr[:,0].std():.3f})  "
      f"net {rr[:,1].mean():+.3f}%  win {rr[:,2].mean():.1f}%")
z = (run(sig)[0] - rr[:,0].mean())/rr[:,0].std()
print(f"Top-5 is {z:+.2f} standard deviations from the random-pick mean.")
# bottom-5 too
def bpicks(i, n=5):
    r = SC.iloc[i].dropna(); return list(r.sort_values().head(n).index)
bot = [(i, t) for i in eidx for t in bpicks(i) if np.isfinite(C[t][i])]
print(f"Bottom-5 by Buy %:   gross {run(bot)[0]:+.3f}%  net {run(bot)[1]:+.3f}%  win {run(bot)[2]:.1f}%")

print("\n" + "="*78); print("C. FULL-5Y WALK-FORWARD (fixed 20-session hold, not hold-to-today)")
print("="*78)
rows = []
for i in eidx:
    ps = picks(i); j = min(i+20, len(dates)-1)
    gs = [(C[t][j]/C[t][i]-1)*100 for t in ps if np.isfinite(C[t][i]) and np.isfinite(C[t][j])]
    bh = [(C[t][j]/C[t][i]-1)*100 for t in close.columns if np.isfinite(C[t][i]) and np.isfinite(C[t][j])]
    if len(gs) < 5: continue
    sn = np.mean([net(v) for v in gs]); bn = net(np.mean(bh))
    rows.append({"date": dates[i].date(), "year": dates[i].year, "strat_net": sn,
                 "bench_net": bn, "alpha": sn-bn})
F = pd.DataFrame(rows)
print(f"n={len(F)} rebalance dates, 2021-2026")
print(f"  strat net  mean {F.strat_net.mean():+.3f}%  median {F.strat_net.median():+.3f}%  %>0 {(F.strat_net>0).mean()*100:.1f}%")
print(f"  bench net  mean {F.bench_net.mean():+.3f}%  median {F.bench_net.median():+.3f}%  %>0 {(F.bench_net>0).mean()*100:.1f}%")
print(f"  alpha      mean {F.alpha.mean():+.3f}%  %>0 {(F.alpha>0).mean()*100:.1f}%  "
      f"t={F.alpha.mean()/(F.alpha.std()/np.sqrt(len(F))):+.2f}")
print("\nPer calendar year:")
print(F.groupby("year").agg(n=("alpha","size"), strat=("strat_net","mean"),
                            bench=("bench_net","mean"), alpha=("alpha","mean")).round(3).to_string())

print("\n" + "="*78); print("D. NON-OVERLAPPING 10-MONTH START DATES (effective sample size)")
print("="*78)
W = pd.read_csv("bt_walkforward.csv")
print(f"All 210 overlapping start dates share the SAME exit day -> they are ~1 observation.")
for k in (21, 42):
    s = W.iloc[::k]
    print(f"  every {k}th date (n={len(s)}): strat {s.strat_net.mean():+.2f}%  "
          f"bench {s.bench_net.mean():+.2f}%  alpha {s.alpha_net.mean():+.2f}%  "
          f"%strat>0 {(s.strat_net>0).mean()*100:.0f}%")
