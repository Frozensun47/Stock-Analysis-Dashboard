"""Low-turnover selection sweep on TRAIN. Never touches TEST."""
import functools, sys
import benchmark as B, strategies as S, strat_explore as X

data = B.load("5y")
HOLDS = [60, 120, 250]
NPOS = [3, 5, 10]

CANDIDATES = dict(S.REGISTRY)
CANDIDATES.update(X.REGISTRY)
BASE = ["random5 (control)", "momentum 6m >SMA200", "lowvol >SMA200",
        "momentum/vol >SMA200", "breakout near 52w high"]
NAMES = BASE + list(X.REGISTRY)

hdr = (f"{'strategy':<24} {'npos':>4} {'hold':>5} {'trd':>4} {'cost%':>6} "
       f"{'total%':>8} {'CAGR%':>7} {'Shrp':>6} {'maxDD%':>7} {'alpha%':>8}")
print("=== TRAIN, trail=None (low turnover). benchmark +82.6% / +29.7% CAGR / -12.4% DD ===")
print(hdr)
rows = []
for name in NAMES:
    fn = CANDIDATES[name]
    for npos in NPOS:
        f = functools.partial(fn, n=npos) if npos != 5 else fn
        for hold in HOLDS:
            m, td = B.run(f, data, B.TRAIN, n_pos=npos, max_hold=hold, trail=None)
            rows.append((name, npos, hold, m))
            print(f"{name:<24} {npos:>4} {hold:>5} {m['trades']:>4} {m['cost_drag']:>6.2f} "
                  f"{m['total_pct']:>8.1f} {m['cagr']:>7.1f} {m['sharpe']:>6.2f} "
                  f"{m['max_dd']:>7.1f} {m['alpha_total_pct']:>8.1f}", flush=True)
            if "--log" in sys.argv:
                B.log(f"{name} n{npos} hold{hold} trail=None", m,
                      hypothesis="low-turnover selection signal beats equal-weight hold",
                      verdict="beats" if m["alpha_total_pct"] > 0 else "loses",
                      notes="TRAIN sweep run_explore.py")

# --- robustness summary: a strategy must win across ALL cells to count ---
print("\n=== robustness: alpha% by (npos,hold), and how many of 9 cells beat bench ===")
for name in NAMES:
    sub = [r for r in rows if r[0] == name]
    a = [r[3]["alpha_total_pct"] for r in sub]
    wins = sum(x > 0 for x in a)
    print(f"{name:<24} wins {wins}/9  median alpha {sorted(a)[len(a)//2]:+7.1f}  "
          f"min {min(a):+7.1f}  max {max(a):+7.1f}")
