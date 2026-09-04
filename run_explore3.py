"""Overfit checks: quality tilt (look-ahead), and TRAIN sub-period stability.

Sub-periods are halves of TRAIN only. TEST is never touched.
"""
import functools, sys
import numpy as np
import benchmark as B, strategies as S, strat_explore as X

data = B.load("5y")
H1 = ("2021-09-06", "2023-04-30")
H2 = ("2023-05-01", "2024-12-31")


def momN(lb):
    def f(data, i, n=5):
        k = S.cache(data)
        up = k.c.iloc[i] > k.sma200.iloc[i]
        return S._top(k.c.iloc[i] / k.c.iloc[i - lb] - 1, n, up)
    return f

CAND = {
    "mom 60d >SMA200": momN(60),
    "mom 90d >SMA200": momN(90),
    "mom 120d >SMA200": momN(120),
    "mom 180d >SMA200": momN(180),
    "MF mom+lowvol+trend": X.mf_mom_lowvol_trend,
    "LOOKAHEAD quality only": None,
    "LOOKAHEAD quality+mom": X.quality_mom,
}

def quality_only(data, i, n=5):
    q = X._quality_scores()
    k = S.cache(data)
    up = k.c.iloc[i] > k.sma200.iloc[i]
    return S._top(q.reindex(k.c.columns), n, up)
CAND["LOOKAHEAD quality only"] = quality_only

print("=== quality tilt on full TRAIN (LOOK-AHEAD BIASED: current fundamental "
      "snapshot used for 2021-24; invalid as a result, indicative only) ===")
hdr = (f"{'strategy':<24} {'npos':>4} {'hold':>5} {'trd':>4} {'cost%':>6} "
       f"{'total%':>8} {'CAGR%':>7} {'Shrp':>6} {'maxDD%':>7} {'alpha%':>8}")
print(hdr)
for name in ["LOOKAHEAD quality only", "LOOKAHEAD quality+mom"]:
    for npos in (3, 5, 10):
        for hold in (60, 120, 250):
            fn = CAND[name]
            f = functools.partial(fn, n=npos) if npos != 5 else fn
            m, _ = B.run(f, data, B.TRAIN, n_pos=npos, max_hold=hold, trail=None)
            print(f"{name:<24} {npos:>4} {hold:>5} {m['trades']:>4} {m['cost_drag']:>6.2f} "
                  f"{m['total_pct']:>8.1f} {m['cagr']:>7.1f} {m['sharpe']:>6.2f} "
                  f"{m['max_dd']:>7.1f} {m['alpha_total_pct']:>8.1f}", flush=True)
            if "--log" in sys.argv:
                B.log(f"{name} n{npos} hold{hold} trail=None", m,
                      hypothesis="LOOK-AHEAD current-snapshot quality tilt",
                      verdict="beats" if m["alpha_total_pct"] > 0 else "loses",
                      notes="INVALID: look-ahead fundamentals, indicative only")

print("\n=== TRAIN split in half: does the alpha survive in BOTH halves? ===")
print(f"{'strategy':<24} {'npos':>4} {'hold':>5} {'H1 alpha':>9} {'H2 alpha':>9} {'both>0':>7}")
for name, fn in CAND.items():
    for npos in (5, 10):
        for hold in (120, 250):
            f = functools.partial(fn, n=npos) if npos != 5 else fn
            a = []
            for sp, tag in ((H1, "H1"), (H2, "H2")):
                m, _ = B.run(f, data, sp, n_pos=npos, max_hold=hold, trail=None)
                a.append(m["alpha_total_pct"])
                if "--log" in sys.argv:
                    B.log(f"{name} n{npos} hold{hold} {tag}", m,
                          hypothesis="does alpha survive in each TRAIN half?",
                          verdict="beats" if m["alpha_total_pct"] > 0 else "loses",
                          notes="TRAIN sub-period; not TEST")
            print(f"{name:<24} {npos:>4} {hold:>5} {a[0]:>9.1f} {a[1]:>9.1f} "
                  f"{str(a[0] > 0 and a[1] > 0):>7}", flush=True)
