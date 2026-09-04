"""Experiment: is the exit rule destroying the returns?

Cost drag of 11-17% of capital over 3.3 years says the strategies are being
churned by their own stop, not by their signal. Sweep holding period and trail
width on the best baseline and see what turnover is actually worth.
"""
import benchmark as B, strategies as S
data = B.load("5y")
print(f"=== exit sweep on TRAIN · {B.CAPITAL:,.0f} capital · {B.N_POS} positions ===")
print(f"benchmark: hold the universe = +82.6% total, +29.7% CAGR, maxDD -12.4%\n")
print(f"{'strategy':<22} {'trail':>6} {'hold':>5} {'trades':>7} {'cost%':>7} "
      f"{'total%':>8} {'CAGR%':>7} {'Sharpe':>7} {'maxDD%':>7} {'alpha%':>8}")
rows = []
for sname in ["momentum/vol >SMA200", "lowvol >SMA200", "momentum 6m >SMA200"]:
    fn = S.REGISTRY[sname]
    for trail, hold in [(4.0, 20), (6.0, 40), (8.0, 60), (10.0, 120), (None, 60),
                        (None, 120), (None, 250)]:
        m, td = B.run(fn, data, B.TRAIN, max_hold=hold, trail=trail)
        rows.append((sname, trail, hold, m))
        print(f"{sname:<22} {str(trail):>6} {hold:>5} {m['trades']:>7} "
              f"{m['cost_drag']:>7.2f} {m['total_pct']:>+8.1f} {m['cagr']:>+7.1f} "
              f"{m['sharpe']:>7.2f} {m['max_dd']:>7.1f} {m['alpha_total_pct']:>+8.1f}")
best = max(rows, key=lambda r: r[3]["alpha_total_pct"])
print(f"\nbest: {best[0]}  trail={best[1]} hold={best[2]}  "
      f"alpha {best[3]['alpha_total_pct']:+.1f}%")
for sname, trail, hold, m in rows:
    B.log(f"{sname} trail={trail} hold={hold}", m,
          hypothesis="turnover, not signal, is destroying returns",
          verdict="beats" if m["alpha_total_pct"] > 0 else "loses")
