"""Forward paper-test: the only honest out-of-sample evidence left.

TRAIN was mined across ~126 configurations and TEST has been spent (see
FINDINGS.md), so no untouched historical window remains. Any further claim about
a strategy has to be earned on data that did not exist when the rule was written.

This records dated picks each day and scores them later. Nothing here can be
tuned after the fact: a pick row is immutable once written, and scoring only ever
reads prices dated strictly after the pick.

    python forward_test.py record     # append today's picks for every strategy
    python forward_test.py score      # score all matured picks vs buy-and-hold
"""
import os, sqlite3, sys, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.getenv("FORWARD_DB", os.path.join(HERE, "cache", "forward.sqlite"))

# Every rule recorded is fixed here, in advance, with its horizon. Adding a new
# arm is fine; editing or deleting an existing one destroys the evidence.
ARMS = {
    "momentum_60_h120": ("momentum_60", 120),
    "equal_weight_bench": ("_bench", 120),
    "random5_control": ("_random", 120),
}


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS picks(
        pick_date TEXT, arm TEXT, symbol TEXT, entry REAL, horizon INT,
        recorded_at TEXT, PRIMARY KEY(pick_date, arm, symbol))""")
    return con


def record(data=None):
    import benchmark as B, strategies as S, random
    data = data or B.load("5y")
    close = data["Close"]
    i = len(close) - 1
    day = str(close.index[i].date())
    con = connect()
    now = dt.datetime.now().isoformat(timespec="seconds")
    for arm, (fn, hz) in ARMS.items():
        live = [c for c in close.columns if close[c].iloc[i] == close[c].iloc[i]]
        if fn == "_bench":
            picks = live
        elif fn == "_random":
            picks = random.Random(day).sample(live, min(5, len(live)))
        else:
            picks = getattr(S, fn)(data, i)
        for s in picks:
            con.execute("INSERT OR IGNORE INTO picks VALUES(?,?,?,?,?,?)",
                        (day, arm, s, float(close[s].iloc[i]), hz, now))
        print(f"  {arm:20s} {len(picks):3d} picks on {day}")
    con.commit()
    con.close()


def score():
    import benchmark as B
    close = B.load("5y")["Close"]
    con = connect()
    rows = con.execute("SELECT * FROM picks").fetchall()
    if not rows:
        print("No picks recorded yet. Run `record` daily first.")
        return
    by_arm = {}
    for r in rows:
        # score only if the horizon has fully elapsed -- never peek
        idx = close.index.searchsorted(r["pick_date"])
        end = idx + r["horizon"]
        if end >= len(close) or r["symbol"] not in close.columns:
            continue
        px = close[r["symbol"]].iloc[end]
        if px != px:
            continue
        by_arm.setdefault(r["arm"], []).append((px / r["entry"] - 1) * 100)
    if not by_arm:
        first = min(r["pick_date"] for r in rows)
        print(f"{len(rows)} picks recorded since {first}; none matured yet.")
        print("Earliest verdict needs 120 trading days (~6 months) after the first pick.")
        return
    print(f"{'arm':22s} {'n':>4s} {'mean %':>9s} {'median %':>9s} {'win %':>7s}")
    for arm, v in sorted(by_arm.items()):
        n = len(v)
        mean = sum(v) / n
        med = sorted(v)[n // 2]
        win = sum(1 for x in v if x > 0) / n * 100
        print(f"{arm:22s} {n:4d} {mean:+9.2f} {med:+9.2f} {win:7.1f}")
    if "momentum_60_h120" in by_arm and "equal_weight_bench" in by_arm:
        a = sum(by_arm["momentum_60_h120"]) / len(by_arm["momentum_60_h120"])
        b = sum(by_arm["equal_weight_bench"]) / len(by_arm["equal_weight_bench"])
        print(f"\nforward alpha vs buy-and-hold: {a - b:+.2f} pts "
              f"({len(by_arm['momentum_60_h120'])} matured picks)")
        print("Treat as evidence only once n is large and several months have passed.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    {"record": record, "score": score}[cmd]()
