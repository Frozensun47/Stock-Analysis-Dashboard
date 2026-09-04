# Findings — what actually works, and what does not

Written after spending the TEST split. **TEST is now used. Nothing may be tuned
against it.** Any future idea needs a new out-of-sample window.

## The headline result is negative

The best hypothesis found on TRAIN — 3-month momentum above SMA200, 5 positions,
120-day hold, no stop — **failed out-of-sample.**

| | TRAIN 2021-09→2024-12 | TEST 2025-01→2026-08 |
|---|---|---|
| strategy | +48.3% CAGR | **-3.2% CAGR** |
| buy & hold | +32.1% CAGR | **+7.8% CAGR** |
| alpha | +58.6% | **-18.4%** |
| maxDD | -19.8% | -25.6% |

Predicted in advance by two independent checks, both of which were right:

1. **Half-sample split.** On TRAIN, H1 alpha was +1.9 and H2 was +74.1. Essentially
   all the edge came from one 20-month regime, not from a stable signal.
2. **Multiple-testing.** ~126 configurations were tried. Random-5 alpha has sd ≈ 21
   points, so the expected best-of-126 noise draw is ~65 points. The +58.6 "edge"
   sat *inside* that envelope.

Momentum did beat all 60 random-5 draws (z = +3.96), which is why it was worth
spending TEST on. It still did not survive.

## A bug in the scoring harness (found and fixed)

`benchmark.run` compared against `(1 + pct_change().mean(axis=1)).cumprod()` — a
daily-**rebalanced** equal-weight index, not buy-and-hold. On TRAIN that bar was
23.0% CAGR instead of the correct 25.2%, so every historical alpha in this repo was
**overstated by ~12 points**. Now uses a true buy-and-hold basket.

This is the second harness bug found. The first rewarded strategies for trading more
(random picks "beat" everything). Both inflated results. Check the scorer first.

## What is confirmed true

- **Turnover, not signal quality, drove every earlier result.** Cost drag: 17.5% of
  capital at trail=4%/hold=20 vs 1.7% at trail=None/hold=120. The flat Rs 13.5 DP
  charge dominates small tickets.
- **No selection signal tested has out-of-sample alpha.** Momentum, low-vol,
  breakout, mean-reversion, multi-factor composites, the 15m ML model, and the
  dashboard's Buy % (rank-IC -0.0038, t = -0.20) all fail.
- **Per-trade stops cost money at every width tested.**
- **Intraday is structurally unprofitable here:** ~3bps gross edge vs ~15bps cost.
- **Concentration is a penalty, not an edge.** Random-5 averages -24.9% alpha vs the
  equal-weight basket; 5-name books carry ~15-18% tracking error.

## Known biases not yet corrected

- **Survivorship:** the 124-name universe was chosen in 2026 and priced back to 2021.
  This inflates both the benchmark and every strategy.
- **Look-ahead in the quality tilt:** `fundamentals.metrics_frame()` is a *current*
  snapshot. The quality results (stable across both TRAIN halves) are invalid.
  Point-in-time fundamentals from the dated `statements` table are the one genuinely
  promising follow-up.

## The honest recommendation

Over TEST the equal-weight basket returned **+13.2% (+7.8% CAGR)** while active
selection returned -5.2%. On this evidence the net-positive outcome is the passive
one. For a Rs 2L account, a Nifty200 Momentum 30 / Alpha Low Vol 30 index fund
delivers the same factor tilt at ~0.3% TER with no DP charges.

The scanner remains useful as a screen. It should not be read as a return forecast.

## What happens next: forward paper-testing

TRAIN was mined across ~126 configurations and TEST is spent, so **no untouched
historical window remains**. The only honest evidence left is data that did not
exist when the rule was written.

`forward_test.py` records dated picks every day (wired into `daily_sync.sh`) for
three arms — the momentum rule, the equal-weight benchmark, and a random-5 control —
and scores them only after the full 120-day horizon has elapsed. Pick rows are
immutable and scoring never reads a price dated on or before the pick.

    .venv/bin/python forward_test.py score

First meaningful verdict: roughly 6 months of accrual. Until then the honest
position is the one above — the passive basket is what has actually made money.

### Why point-in-time fundamentals were not pursued

The `statements` table is dated and could be made point-in-time, but annual
coverage is only FY2022 through FY2026 — five report dates, about three usable
inside TRAIN. Three annual rebalances cannot distinguish skill from luck against a
21-point noise sd. Building it would produce a number, not evidence.
