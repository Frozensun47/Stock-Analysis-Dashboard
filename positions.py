"""Track real purchases and tell you when to sell.

The virtual portfolio in engine.py closed every position blindly after 7
sessions regardless of price — that is a timer, not a sell rule. This module
records positions you actually bought and evaluates a real exit rule against
them on every refresh, so the same logic that decides "sell" in the app is the
one a backtest can replay.

EXIT RULE (each position carries its own copy, so changing the default never
silently rewrites the rule an open position was bought under):
  - stop loss      : hard floor on loss from entry
  - trailing stop  : from the highest CLOSE seen since entry, armed only after
                     the position is up by TRAIL_ARM (a trail that is live from
                     day one just converts normal noise into an exit)
  - take profit    : optional ceiling
  - time exit      : maximum holding period
Signals are evaluated on daily closes, which is what the rules were fitted on.

Schema
    positions(id INTEGER PK, symbol, qty, buy_price, buy_date, rule TEXT(json),
              status, sell_price, sell_date, sell_reason, note)

Usage
    python positions.py add RELIANCE 10 1420.5          # qty and price
    python positions.py add RELIANCE 10 1420.5 2026-08-01
    python positions.py check                            # what to sell today
    python positions.py list
    python positions.py close 3 1465.0                   # record a real sale
"""
import json, os, sqlite3, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.getenv("POSITIONS_DB", os.path.join(HERE, "cache", "positions.sqlite"))

# These defaults are exactly the "trail 4% / 20-session" rule from the exit grid
# in README.md — the only rule family there with non-negative alpha. The trail
# arms immediately (trail_arm 0) because that is the variant that was measured;
# arming it later is untested. A 4% trail from entry already acts as the stop
# loss, so a separate stop_loss would double up on an untested rule.
#
# What the grid actually says: no exit rule creates an edge, because the ENTRY
# signal has none (see README). This one gives up ~0.56%/trade versus a plain
# 20-day hold to cut the worst trade from -59.5% to -13.3%. That is the trade
# worth making on a signal with no demonstrated edge.
DEFAULT_RULE = {"stop_loss": None, "trail": 4.0, "trail_arm": 0.0,
                "take_profit": None, "max_hold": 20}

DDL = """
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol TEXT NOT NULL, qty REAL NOT NULL, buy_price REAL NOT NULL,
  buy_date TEXT NOT NULL, rule TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open',
  sell_price REAL, sell_date TEXT, sell_reason TEXT, note TEXT);
CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(status);
"""

def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(DDL)
    return con

def add(symbol, qty, buy_price, buy_date=None, rule=None, note=""):
    """Mark a stock as purchased. buy_date defaults to today."""
    buy_date = str(buy_date or pd.Timestamp.today().date())
    rule = {**DEFAULT_RULE, **(rule or {})}
    con = connect()
    cur = con.execute(
        "INSERT INTO positions (symbol, qty, buy_price, buy_date, rule, note) "
        "VALUES (?,?,?,?,?,?)",
        (symbol.upper().replace(".NS", ""), float(qty), float(buy_price),
         buy_date, json.dumps(rule), note))
    con.commit(); pid = cur.lastrowid; con.close()
    return pid

def close(pid, sell_price, sell_date=None, reason="manual"):
    con = connect()
    con.execute("UPDATE positions SET status='closed', sell_price=?, sell_date=?, "
                "sell_reason=? WHERE id=?",
                (float(sell_price), str(sell_date or pd.Timestamp.today().date()), reason, pid))
    con.commit(); con.close()

def open_positions():
    con = connect()
    rows = [dict(r) for r in con.execute("SELECT * FROM positions WHERE status='open' ORDER BY buy_date")]
    con.close()
    for r in rows:
        r["rule"] = json.loads(r["rule"])
    return rows


def evaluate_position(pos, close_series):
    """Apply the position's own exit rule to its price history since purchase.

    Returns a dict with the current state and, if triggered, the sell reason.
    Uses only closes at or after the buy date, so it never sees data the holder
    could not have seen either.
    """
    rule = pos["rule"]
    s = close_series.dropna()
    s = s[s.index >= pd.Timestamp(pos["buy_date"])]
    if s.empty:
        return {**pos, "signal": "no data", "action": "HOLD"}
    entry, last = float(pos["buy_price"]), float(s.iloc[-1])
    peak = float(s.max())
    held = len(s) - 1
    pnl = (last / entry - 1) * 100
    peak_gain = (peak / entry - 1) * 100
    trail_level = peak * (1 - rule["trail"] / 100)

    action, why = "HOLD", ""
    if rule.get("stop_loss") and pnl <= -rule["stop_loss"]:
        action, why = "SELL", f"stop loss: {pnl:+.1f}% ≤ -{rule['stop_loss']}%"
    elif rule.get("take_profit") and pnl >= rule["take_profit"]:
        action, why = "SELL", f"take profit: {pnl:+.1f}% ≥ +{rule['take_profit']}%"
    elif peak_gain >= rule.get("trail_arm", 0) and last <= trail_level:
        action, why = "SELL", (f"trailing stop: fell to {last:,.1f} from a peak of "
                               f"{peak:,.1f} (-{rule['trail']}%)")
    elif rule.get("max_hold") and held >= rule["max_hold"]:
        action, why = "SELL", f"time exit: held {held} sessions ≥ {rule['max_hold']}"
    elif peak_gain >= rule.get("trail_arm", 0):
        why = f"trail armed — sell below {trail_level:,.1f}"
    else:
        why = (f"trail arms at +{rule.get('trail_arm', 0)}% "
               f"(peak so far {peak_gain:+.1f}%); stop at -{rule['stop_loss']}%")

    return {**pos, "last_price": last, "pnl_pct": pnl, "peak_gain_pct": peak_gain,
            "held_sessions": held, "trail_level": trail_level,
            "action": action, "signal": why}


def check(close_df=None):
    """Evaluate every open position. close_df: wide daily closes, columns '<SYM>.NS'."""
    if close_df is None:
        from engine import fetch_prices
        close_df = fetch_prices()["Close"]
    out = []
    for pos in open_positions():
        col = pos["symbol"] + ".NS"
        if col not in close_df.columns:
            out.append({**pos, "action": "HOLD", "signal": "not in universe"}); continue
        out.append(evaluate_position(pos, close_df[col]))
    return out


def summary(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [c for c in ["id", "symbol", "qty", "buy_price", "buy_date", "last_price",
                        "pnl_pct", "held_sessions", "action", "signal"] if c in df]
    return df[keep]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "add":
        if len(sys.argv) < 5:
            print("usage: positions.py add SYMBOL QTY BUY_PRICE [BUY_DATE]"); sys.exit(1)
        pid = add(sys.argv[2], sys.argv[3], sys.argv[4],
                  sys.argv[5] if len(sys.argv) > 5 else None)
        print(f"tracked position #{pid}: {sys.argv[2].upper()} x{sys.argv[3]} @ {sys.argv[4]}")
    elif cmd == "close":
        close(int(sys.argv[2]), float(sys.argv[3]))
        print(f"closed position #{sys.argv[2]}")
    elif cmd in ("check", "list"):
        rows = check()
        if not rows:
            print("no open positions — add one with `positions.py add SYMBOL QTY PRICE`")
        else:
            df = summary(rows)
            print(df.to_string(index=False))
            sells = [r for r in rows if r["action"] == "SELL"]
            print(f"\n{len(sells)} SELL signal(s)" if sells else "\nnothing to sell today")
            for r in sells:
                print(f"  SELL {r['symbol']}: {r['signal']}")
    else:
        print(__doc__)
