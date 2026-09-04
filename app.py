import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from engine import (fetch_prices, scan, backtest, load_port, save_port, buy,
                    evaluate, news_for, DEFAULT_WEIGHTS)

# Streamlit secrets -> environment, so the data modules (which read os.getenv
# and a local .env) work unchanged when deployed. Only scalars are copied;
# st.secrets raises if no secrets file exists at all, hence the guard.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

st.set_page_config(page_title="NSE Stock Dashboard", layout="wide")
st.title("📈 NSE Momentum Dashboard")

# ---- sidebar: weights ----
st.sidebar.header("Score weights")
weights = {}
for k, v in DEFAULT_WEIGHTS.items():
    weights[k] = st.sidebar.slider(k, 0.0, 0.5, v, 0.05)
if st.sidebar.button("Refresh data (clear cache)"):
    fetch_prices(max_age_min=0)

data = fetch_prices()
close = data["Close"]
st.caption(f"Data through **{close.index[-1].date()}** · {close.shape[1]} stocks · cached ≤30 min")

(tab_scan, tab_hold, tab_intra, tab_port, tab_bt,
 tab_news, tab_corpus, tab_fund) = st.tabs(
    ["🔍 Scanner", "📌 My Positions", "⚡ 15m Model", "💰 Virtual Portfolio",
     "🧪 Backtest", "📰 News", "🗞️ News Corpus", "📑 Fundamentals"])

# ---- scanner ----
with tab_scan:
    df = scan(data, weights=weights)
    c1, c2, c3 = st.columns(3)
    c1.metric("Top pick", df.iloc[0]["Symbol"], f'{df.iloc[0]["Buy %"]}% buy score')
    c2.metric("Stocks scoring ≥ 70", int((df["Buy %"] >= 70).sum()))
    c3.metric("Positive 10d returns", int((df["ret10"] > 0).sum()))
    st.dataframe(df, use_container_width=True, height=450)

    sym = st.selectbox("Chart", df["Symbol"])
    s = close[sym + ".NS"].dropna()
    fig = go.Figure(go.Scatter(x=s.index, y=s, name=sym))
    fig.add_scatter(x=s.index, y=s.rolling(20).mean(), name="SMA20")
    fig.add_scatter(x=s.index, y=s.rolling(50).mean(), name="SMA50")
    fig.update_layout(height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ---- portfolio ----
with tab_port:
    p = load_port()
    p = evaluate(p, close)  # auto-close positions held 7+ sessions
    open_val = sum(pos["qty"] * pos.get("last_price", pos["buy_price"]) for pos in p["open"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", f'₹{p["cash"]:,.0f}')
    c2.metric("Open positions value", f'₹{open_val:,.0f}')
    c3.metric("Total equity", f'₹{p["cash"] + open_val:,.0f}',
              f'{(p["cash"] + open_val) / 1_000_000 * 100 - 100:+.2f}%')
    closed = pd.DataFrame(p["closed"])
    c4.metric("Closed trades win rate",
              f'{(closed.ret > 0).mean() * 100:.0f}%' if not closed.empty else "—")

    st.subheader("Buy (virtual)")
    df = scan(data, weights=weights)
    col1, col2, col3 = st.columns([2, 1, 1])
    sym = col1.selectbox("Stock", df["Symbol"], key="buysym")
    amt = col2.number_input("Amount ₹", 1000, 500000, 50000, 1000)
    row = df[df["Symbol"] == sym].iloc[0]
    if col3.button(f"Buy @ ₹{row['close']}"):
        buy(p, sym, float(row["close"]), amt, float(row["Buy %"]), close.index[-1].date())
        st.rerun()
    if st.button(f"Auto-buy top 5 (₹50k each, score ≥ 65)"):
        for _, r in df[df["Buy %"] >= 65].head(5).iterrows():
            buy(p, r["Symbol"], float(r["close"]), 50000, float(r["Buy %"]), close.index[-1].date())
        st.rerun()

    if p["open"]:
        st.subheader("Open positions (auto-sold after 7 trading days)")
        op = pd.DataFrame(p["open"])
        op["P&L %"] = (op.get("last_price", op["buy_price"]) / op["buy_price"] - 1) * 100
        st.dataframe(op, use_container_width=True)
    if not closed.empty:
        st.subheader("Closed trades — strategy evaluation")
        st.dataframe(closed.sort_values("sell_date", ascending=False), use_container_width=True)
        st.write(f"Avg return: **{closed.ret.mean():.2f}%** · Win rate: **{(closed.ret>0).mean()*100:.0f}%**")
    if st.button("Reset portfolio to ₹10L"):
        save_port({"cash": 1_000_000.0, "open": [], "closed": []})
        st.rerun()

# ---- backtest ----
with tab_bt:
    c1, c2, c3, c4 = st.columns(4)
    hold = c1.number_input("Hold (trading days)", 3, 30, 7)
    top_n = c2.number_input("Picks per rebalance", 1, 20, 5)
    step = c3.number_input("Rebalance every N days", 1, 20, 5)
    min_score = c4.number_input("Min buy score", 0, 100, 60)
    if st.button("Run backtest"):
        with st.spinner("Backtesting over the past year…"):
            tdf, stats = backtest(data, hold=hold, top_n=top_n, step=step,
                                  weights=weights, min_score=min_score)
        if tdf.empty:
            st.warning("No trades matched — lower the min score.")
        else:
            c = st.columns(6)
            for col, (k, v) in zip(c, stats.items()):
                col.metric(k.replace("_", " "), v)
            tdf["bucket"] = pd.cut(tdf["Buy %"], [0, 50, 60, 70, 80, 100])
            st.write("**Return by score bucket** — does a higher score actually pay?")
            st.dataframe(tdf.groupby("bucket", observed=True).ret.agg(["count", "mean", lambda x: (x > 0).mean() * 100])
                         .rename(columns={"<lambda_0>": "win %"}).round(2))
            st.dataframe(tdf.sort_values("date", ascending=False), use_container_width=True, height=300)

# ---- news ----
with tab_news:
    sym = st.selectbox("Stock", scan(data, weights=weights)["Symbol"], key="newssym")
    for it in news_for(sym) or [{"title": "No news found (yfinance free feed)", "publisher": "", "link": ""}]:
        st.markdown(f"- [{it['title']}]({it['link']}) — *{it['publisher']}*")


# ---- intraday 15m model ----
with tab_intra:
    st.subheader("15-minute ML model (Upstox data)")
    st.caption("Gradient-boosted regressor predicting the next ~2h (8 bars) return. "
               "Trained walk-forward; entry at the next bar's open, trailing stop, flat by session close.")
    try:
        from upstox_data import load_15m
        import model_15m as M
        panel = load_15m()
        c = panel["Close"]
        st.caption(f"15m data through **{c.index[-1]:%Y-%m-%d %H:%M}** · {c.shape[1]} stocks · {c.shape[0]:,} bars")
        if not os.path.exists(M.MODEL_PATH):
            st.warning("No trained model yet — run `python model_15m.py` to train and save one.")
        else:
            k = st.slider("Top picks", 1, 20, 5)
            picks = M.live_signals(panel, top_k=k)
            st.dataframe(picks.style.format({"pred_%": "{:+.3f}", "price": "{:,.2f}"}),
                         use_container_width=True)
            sym = st.selectbox("Intraday chart", [t.replace(".NS", "") for t in c.columns],
                               index=0, key="intrasym")
            s15 = c[sym + ".NS"].dropna().tail(400)
            f = go.Figure(go.Scatter(x=s15.index, y=s15, name=f"{sym} 15m"))
            f.add_scatter(x=s15.index, y=s15.rolling(26).mean(), name="SMA26")
            f.update_layout(height=380, margin=dict(t=20, b=20))
            st.plotly_chart(f, use_container_width=True)
    except FileNotFoundError:
        st.info("No 15-minute cache found. Run `python upstox_data.py 2022-01-01` "
                "with UPSTOX_ACCESS_TOKEN set to download it.")
    except Exception as e:
        st.error(f"Intraday tab unavailable: {e}")


# ---- collected news corpus (SQLite, refreshed daily) ----
with tab_corpus:
    st.subheader("News corpus")
    st.caption("Keyless RSS: Google News per symbol plus Moneycontrol, Economic Times, "
               "Livemint, Business Standard, BusinessLine, Zerodha and Trendlyne. "
               "Sentiment is a negation-aware finance lexicon, not a language model.")
    try:
        import news_db as ND
        con = ND.connect()
        n, nsym, tmax = con.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MAX(ts) FROM news").fetchone()
        if not n:
            st.info("Corpus is empty — run `python news_db.py sync`.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Articles", f"{n:,}")
            c2.metric("Symbols covered", nsym)
            c3.metric("Latest article", pd.to_datetime(tmax, unit="s").strftime("%d %b %H:%M"))

            syms = [r[0] for r in con.execute(
                "SELECT symbol, COUNT(*) c FROM news GROUP BY symbol ORDER BY c DESC")]
            sym = st.selectbox("Symbol", syms, key="corpsym")
            days = st.slider("Look back (days)", 1, 90, 30, key="corpdays")
            arts = pd.read_sql(
                "SELECT ts, title, source, sentiment, link FROM news "
                "WHERE symbol=? AND ts > strftime('%s','now',?) ORDER BY ts DESC",
                con, params=(sym, f"-{days} days"))
            if arts.empty:
                st.info("No articles in that window.")
            else:
                arts["when"] = pd.to_datetime(arts.ts, unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
                d1, d2 = st.columns(2)
                d1.metric("Articles", len(arts))
                d2.metric("Mean sentiment", f"{arts.sentiment.mean():+.2f}")
                daily = arts.set_index("when").sentiment.resample("D").agg(["mean", "count"]).dropna()
                if len(daily) > 1:
                    fg = go.Figure(go.Bar(x=daily.index, y=daily["mean"],
                                          marker_color=["#2e7d32" if v >= 0 else "#c62828" for v in daily["mean"]]))
                    fg.update_layout(height=220, margin=dict(t=10, b=10), yaxis_title="daily sentiment")
                    st.plotly_chart(fg, use_container_width=True)
                for _, r in arts.head(40).iterrows():
                    tone = "🟢" if r.sentiment > 0.1 else ("🔴" if r.sentiment < -0.1 else "⚪")
                    st.markdown(f"{tone} **[{r.title}]({r.link})** · *{r.source}* · "
                                f"{r.when:%d %b %H:%M} · `{r.sentiment:+.2f}`")
        con.close()
    except Exception as e:
        st.error(f"News corpus unavailable: {e}")

# ---- fundamentals ----
with tab_fund:
    st.subheader("Financial statements")
    st.caption("Annual and quarterly statements plus ratio snapshots, collected from yfinance.")
    try:
        import fundamentals as FU
        con = FU.connect()
        syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM statements ORDER BY symbol")]
        con.close()
        if not syms:
            st.info("No fundamentals yet — run `python fundamentals.py sync`.")
        else:
            c1, c2, c3 = st.columns(3)
            sym = c1.selectbox("Symbol", syms, key="fundsym")
            which = c2.selectbox("Statement", ["income", "balance", "cashflow"], key="fundwhich")
            freq = c3.selectbox("Frequency", ["A", "Q"], key="fundfreq",
                                format_func=lambda x: "Annual" if x == "A" else "Quarterly")
            m = FU.metrics_frame([sym])
            if not m.empty:
                row = m.iloc[0]
                cols = st.columns(5)
                for col, k in zip(cols, ["trailingPE", "priceToBook", "returnOnEquity",
                                         "debtToEquity", "profitMargins"]):
                    v = row.get(k)
                    col.metric(k, "—" if pd.isna(v) else f"{v:,.2f}")
            df = FU.statement(sym, which, freq)
            if df.empty:
                st.info("No rows for that combination.")
            else:
                st.dataframe((df / 1e7).round(1).rename_axis("₹ crore"), use_container_width=True, height=520)
                st.caption("Values in ₹ crore (raw values ÷ 10⁷); per-share items are scaled too.")
    except Exception as e:
        st.error(f"Fundamentals unavailable: {e}")


# ---- real positions: mark what you bought, get a sell signal ----
with tab_hold:
    import positions as POS
    st.subheader("Positions you actually hold")
    st.caption("Mark a stock as purchased and it is tracked against a real exit rule — "
               "stop loss, a trailing stop that arms only once you are up, take profit, "
               "and a time limit. The same rule is what the backtest replays.")

    rows = POS.check(close)
    sells = [r for r in rows if r.get("action") == "SELL"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Open positions", len(rows))
    c2.metric("Sell signals today", len(sells))
    if rows:
        inv = sum(r["qty"] * r["buy_price"] for r in rows)
        now = sum(r["qty"] * r.get("last_price", r["buy_price"]) for r in rows)
        c3.metric("Unrealised P&L", f"₹{now - inv:,.0f}",
                  f"{(now / inv - 1) * 100:+.2f}%" if inv else None)

    if sells:
        st.error("**Sell signals:**\n" + "\n".join(
            f"- **{r['symbol']}** ({r['pnl_pct']:+.2f}%) — {r['signal']}" for r in sells))

    with st.expander("➕ Mark a stock as purchased", expanded=not rows):
        df_scan = scan(data, weights=weights)
        f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
        psym = f1.selectbox("Stock", df_scan["Symbol"], key="possym")
        live = float(df_scan[df_scan["Symbol"] == psym].iloc[0]["close"])
        pprice = f2.number_input("Buy price ₹", 0.01, 1e6, live, key="posprice")
        pqty = f3.number_input("Quantity", 1, 100000, 1, key="posqty")
        pdate = f4.date_input("Buy date", close.index[-1].date(), key="posdate")
        r1, r2, r3, r4 = st.columns(4)
        rule = {
            "stop_loss": r1.number_input("Stop loss %", 1.0, 30.0,
                                         float(POS.DEFAULT_RULE["stop_loss"]), 0.5),
            "trail": r2.number_input("Trailing stop %", 1.0, 20.0,
                                     float(POS.DEFAULT_RULE["trail"]), 0.5),
            "trail_arm": r3.number_input("Arm trail after +%", 0.0, 20.0,
                                         float(POS.DEFAULT_RULE["trail_arm"]), 0.5),
            "max_hold": int(r4.number_input("Max hold (sessions)", 1, 120,
                                            int(POS.DEFAULT_RULE["max_hold"]))),
            "take_profit": None,
        }
        if st.button("Track this purchase", type="primary"):
            POS.add(psym, pqty, pprice, pdate, rule)
            st.success(f"Tracking {psym} × {pqty} @ ₹{pprice:,.2f}")
            st.rerun()

    if rows:
        st.subheader("Open")
        st.dataframe(POS.summary(rows), use_container_width=True)
        st.caption("`action` is the rule's verdict on today's close. "
                   "A stop loss cannot protect against an overnight gap — "
                   "Paytm's Feb-2024 gap blew through a 6% stop at -20%.")
        cid, cpx, cbtn = st.columns([1, 1, 1])
        pid = cid.selectbox("Position", [r["id"] for r in rows], key="closeid")
        cur = next(r.get("last_price", r["buy_price"]) for r in rows if r["id"] == pid)
        spx = cpx.number_input("Sell price ₹", 0.01, 1e6, float(cur), key="closepx")
        if cbtn.button("Record sale"):
            POS.close(pid, spx)
            st.rerun()

    con = POS.connect()
    hist = pd.read_sql("SELECT symbol, qty, buy_price, buy_date, sell_price, sell_date, "
                       "sell_reason FROM positions WHERE status='closed' "
                       "ORDER BY sell_date DESC", con)
    con.close()
    if not hist.empty:
        hist["ret %"] = (hist.sell_price / hist.buy_price - 1) * 100
        st.subheader("Closed")
        st.dataframe(hist.round(2), use_container_width=True)
        st.write(f"Realised: **{hist['ret %'].mean():+.2f}%** average over "
                 f"{len(hist)} trades · win rate **{(hist['ret %'] > 0).mean() * 100:.0f}%**")
