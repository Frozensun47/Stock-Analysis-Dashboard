"""Follow-ups: regime diagnostics, momentum-lookback stability, quality tilt."""
import functools, sys
import numpy as np, pandas as pd
import benchmark as B, strategies as S, strat_explore as X

data = B.load("5y")
x = X.xcache(data)
lo = data["Close"].index.searchsorted(pd.Timestamp(B.TRAIN[0]))
hi = data["Close"].index.searchsorted(pd.Timestamp(B.TRAIN[1]))
on_i = (x.ew.iloc[lo:hi] > x.ew_sma200.iloc[lo:hi]).mean()
on_b = (x.breadth.iloc[lo:hi] > 0.5).mean()
print(f"regime(index) ON {on_i*100:.1f}% of TRAIN days; regime(breadth) ON {on_b*100:.1f}%")


def momN(lb):
    def f(data, i, n=5):
        k = S.cache(data)
        up = k.c.iloc[i] > k.sma200.iloc[i]
        return S._top(k.c.iloc[i] / k.c.iloc[i - lb] - 1, n, up)
    return f

CAND = {f"mom {lb}d >SMA200": momN(lb) for lb in (60, 90, 120, 180, 252)}
CAND["mom120 + breadth"] = X._wrap_regime(momN(120), "breadth")
CAND["LOOKAHEAD quality+mom"] = X.quality_mom
CAND["LOOKAHEAD quality only"] = None

def quality_only(data, i, n=5):
    q = X._quality_scores()
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    return S._top(q.reindex(k.c.columns), n, up)
CAND["LOOKAHEAD quality only"] = quality_only

hdr = (f"{'strategy':<24} {'npos':>4} {'hold':>5} {'trd':>4} {'cost%':>6} "
       f"{'total%':>8} {'CAGR%':>7} {'Shrp':>6} {'maxDD%':>7} {'alpha%':>8}")
print(hdr)
res = {}
for name, fn in CAND.items():
    alphas = []
    for npos in (3, 5, 10):
        for hold in (60, 120, 250):
            f = functools.partial(fn, n=npos) if npos != 5 else fn
            m, _ = B.run(f, data, B.TRAIN, n_pos=npos, max_hold=hold, trail=None)
            alphas.append(m["alpha_total_pct"])
            print(f"{name:<24} {npos:>4} {hold:>5} {m['trades']:>4} {m['cost_drag']:>6.2f} "
                  f"{m['total_pct']:>8.1f} {m['cagr']:>7.1f} {m['sharpe']:>6.2f} "
                  f"{m['max_dd']:>7.1f} {m['alpha_total_pct']:>8.1f}", flush=True)
            if "--log" in sys.argv:
                B.log(f"{name} n{npos} hold{hold} trail=None", m,
                      hypothesis="lookback / quality / breadth-regime follow-up",
                      verdict="beats" if m["alpha_total_pct"] > 0 else "loses",
                      notes="TRAIN sweep run_explore2.py")
    res[name] = alphas

print("\n=== stability ===")
for name, a in res.items():
    print(f"{name:<24} wins {sum(v>0 for v in a)}/9  median {sorted(a)[4]:+7.1f}  "
          f"min {min(a):+7.1f}  max {max(a):+7.1f}")

# random control across seeds, at the 'best' cell, to size the noise band
print("\n=== random5 control, 20 seeds at n=10 hold=250 (noise band) ===")
al = []
for seed in range(20):
    f = functools.partial(S.random5, n=10, seed=seed)
    m, _ = B.run(f, data, B.TRAIN, n_pos=10, max_hold=250, trail=None)
    al.append(m["alpha_total_pct"])
al = np.array(al)
print(f"alpha mean {al.mean():+.1f}%  sd {al.std():.1f}%  min {al.min():+.1f}  max {al.max():+.1f}")
if "--log" in sys.argv:
    B.log("random5 control x20 seeds n10 hold250", dict(split=f"{B.TRAIN[0]}..{B.TRAIN[1]}",
          alpha_total_pct=float(al.mean())),
          hypothesis="how big is the noise band on alpha at this turnover?",
          verdict="noise band", notes=f"sd {al.std():.1f}% across 20 seeds")
