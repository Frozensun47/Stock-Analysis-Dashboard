import sys
import benchmark as B, strategies as S
data = B.load("5y")
split = B.TEST if "--test" in sys.argv else B.TRAIN
names = [a for a in sys.argv[1:] if not a.startswith("--")]
reg = {k: v for k, v in S.REGISTRY.items() if not names or any(n in k for n in names)}
print(f"=== {'TEST' if split is B.TEST else 'TRAIN'} {split[0]}..{split[1]} · "
      f"Rs {B.CAPITAL:,.0f} · {B.N_POS} positions · rebalance {B.REBALANCE}d ===")
for name, fn in reg.items():
    m, td = B.run(fn, data, split)
    B.report(m, name)
    if "--log" in sys.argv:
        B.log(name, m, verdict="beats" if m.get("alpha_total_pct", 0) > 0 else "loses")
