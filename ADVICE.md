# Advice on the NSE factor / selection research

Written 2026-09-04 after reading `benchmark.py`, `strategies.py`, `README.md` and the
measured results supplied. Opinions are mine; numbers I did not measure here are
marked as literature ranges.

## 0. The short version

- **Stock selection at 5 names out of 124 is not a factor strategy, it is a lottery ticket.**
  Nothing tested so far — scanner, 15m ML, exit-rule grid, the +56% momentum cell — has
  produced evidence that survives an honest null. The +56% alpha is roughly 2 sigma before
  correcting for ~126 tried combinations, which is exactly what noise looks like.
- **Costs are not the binding constraint once you rebalance quarterly.** At Rs 40k tickets a
  round trip is 0.38%; 70% one-way annual turnover costs ~0.3%/yr. The 11-17%-of-capital
  drag came from weekly/20-day churn, not from the factor.
- **There is a durable, published premium in India: momentum, and to a lesser degree low
  volatility.** Long-only, broad (30+ names), rebalanced quarterly-to-semi-annually, the
  realistic gross edge over cap-weight is 3-6%/yr for momentum with -35% to -55% drawdowns,
  and 0-3%/yr for low-vol with materially lower drawdowns. Neither is capturable with 5 names.
- **For a Rs 2 lakh delivery account the rational product is an index fund**: Nifty Midcap 150
  or Nifty200 Momentum 30 / Nifty100 Low Vol 30 at ~0.2-0.4% TER, zero DP charges, no
  20-trade backtests. If the goal is to make money rather than to build a system, stop here.
- If the goal is to build a system anyway, section 6 gives the tests that could actually
  change the conclusion. They are few, and most of them are about measurement, not signals.

## 1. Is beating equal-weight buy & hold realistic?

Mostly no, and the framing hides three things.

**a) The benchmark in `benchmark.run` is daily-rebalanced equal weight, not buy & hold.**
`w.pct_change().mean(axis=1)` is the return of a portfolio reset to equal weight every day.
That earns a diversification/rebalancing premium of roughly 1-3%/yr over true buy-and-hold in
a high-dispersion midcap universe and would itself cost a fortune to run. The bar is set
slightly too high, and it is not a portfolio anyone can hold. Report true B&H (fixed shares
from day one) as well.

**b) The universe is survivorship-biased.** 124 "large/midcaps" defined in 2026, priced back
to 2021, contain the names that made it. That inflates the benchmark and inflates every
momentum result (the winners are, by construction, in the sample). Whatever you do, the
2021-2024 +29.7% CAGR is not the expected return of this universe; ~12-15% nominal is the
long-run Indian midcap number.

**c) "Alpha" in the harness is total return minus benchmark total return, not beta-adjusted.**
`exposure` is hard-coded to `nan`. Momentum names in a bull market carry beta 1.2-1.4 versus
the equal-weight basket. 0.3 extra beta on a 30% CAGR benchmark is ~9%/yr of "alpha" that is
just leverage. Until you regress the strategy's daily returns on the benchmark and report the
intercept and its t-stat, none of the alpha numbers mean what they say.

The honest position: in a 3.3-year sample of a bull market, with 5 concentrated names, you
cannot distinguish selection skill from noise plus beta. The best achievable target for a
retail long-only book is "capture the midcap/momentum premium at 0.3%/yr cost with a
drawdown you can hold through". That is worth doing; it is not stock picking.

## 2. Factor evidence in emerging / Indian markets

Ranked by how much I would trust the premium going forward in India, long-only, vs cap-weight.

| Factor | Evidence | Realistic gross alpha (long-only tilt vs index) | Turnover (one-way, annual) | Survives 0.35-0.7% RT at quarterly? |
|---|---|---|---|---|
| **Momentum (12-1 or 6-1)** | Strongest EM factor. Fama-French (2017) international tests; Cakici-Fabozzi-Tan (2013) EM incl. India; Griffin-Ji-Martin (2003); Indian academic work (Sehgal-Balakrishnan 2002, Ansari-Khan 2012, IIMA Agarwalla-Jacob-Varma factor library). Live: Nifty200 Momentum 30 (2020-), backfill to 2005 ~+5-7%/yr over Nifty 200; live since launch roughly +3-5%/yr with a -30%+ drawdown in 2025. | **+3 to +6%/yr**, fat left tail: momentum crashes of -20 to -40% relative in reversal years (2009, 2020, 2025 H1). | 60-120% at semi-annual/quarterly rebalance with buffer rules; 200%+ monthly. | **Yes.** At 80% turnover and 0.5% RT: ~0.4%/yr. Cost is not the problem, crashes are. |
| **Low volatility / low beta** | Robust across EM (Blitz-Pang-van Vliet 2013 "low-volatility effect in emerging markets"). India: Nifty100 Low Vol 30 and Nifty Low Vol 50 have matched or beaten Nifty 100 since 2005 backfill with ~25-30% lower vol and shallower drawdowns; underperforms sharply in high-beta rallies (2021, 2023-24 midcap run). | **0 to +3%/yr** return, but Sharpe improves ~0.2-0.3 and max DD cut by 20-35%. Think of it as a drawdown tool, not a return tool. | 30-50%. | **Yes**, easily. |
| **Quality (ROE, accruals, leverage)** | Asness-Frazzini-Pedersen QMJ holds in EM; India Nifty200 Quality 30 backfill +2-4%/yr but live 2020-2024 underperformed the beta-driven rally badly. | **+1 to +3%/yr**, slow, regime-dependent (works in downturns). | 20-40%. | Yes. Needs fundamentals — you have `fundamentals.sqlite`. |
| **Value (B/M, E/P)** | Positive long-run in EM (Fama-French 2017), but in India it is mostly a PSU/cyclicals bet: dead 2017-2020, spectacular 2021-2024 (Nifty500 Value 50). Enormous regime risk. | **+2 to +4%/yr** long-run mean with 5+ year losing streaks. | 30-50%. | Yes, but you cannot hold through the regime. |
| **Size** | Exists in India but is largely a liquidity / impact premium. Irrelevant: your universe has no size spread. | n/a | n/a | n/a |

Two caveats on all of the above:

1. Those alphas are for **quintile/decile portfolios of 30-50 names**. A 5-name portfolio has
   roughly the same expected factor loading but 3-4x the tracking error, so the factor is
   invisible under the noise. This is the single biggest structural problem with the harness
   (`N_POS = 5`).
2. Indian factor indices are net of nothing; add 0.3-0.5%/yr for a home-built version, and
   note factor premia have compressed globally since publication (McLean-Pontiff: ~30-50%
   post-publication decay).

Cost arithmetic with your own `groww_cost`:

| ticket | RT cost | at 80% turnover/yr | at 200%/yr (monthly) |
|---|---|---|---|
| Rs 2,000 | 1.25% | 1.0%/yr | 2.5%/yr |
| Rs 6,700 (Rs 2L / 30 names) | 0.70% | 0.56%/yr | 1.4%/yr |
| Rs 20,000 | 0.54% | 0.43%/yr | 1.1%/yr |
| Rs 40,000 | 0.38% | 0.30%/yr | 0.76%/yr |

So: a 30-name book on Rs 2 lakh rebalanced quarterly costs ~0.6%/yr, against a 3-6% momentum
premium. The premium survives. What does not survive is trading weekly.

## 3. Rebalancing frequency and holding period

- **Momentum**: signal half-life is ~6-9 months. Rebalance **quarterly**, use a 12-1 (or 6-1)
  lookback, and add a **buffer rule** (enter top 20% of ranks, exit only when a name falls
  below the top 40%) — that halves turnover for almost no loss of exposure. Semi-annual (what
  NSE indices do) gives up ~1%/yr of premium for a further ~30% turnover cut; at your costs
  quarterly is the better trade. Monthly rebalancing adds ~1%/yr of gross alpha in academic
  data but doubles turnover, roughly break-even at 0.5% RT; not worth it on small tickets.
- **Low-vol / quality**: signals are slow. **Semi-annual**, buffer rule, done.
- **Effective holding period**: 9-18 months per name for momentum with buffers, 2+ years for
  low-vol. Your `max_hold=20` and `trail=4` defaults enforce a 1-month horizon on a
  6-12-month signal; the README's own exit grid shows every stop costing money. **Drop stops
  entirely for factor portfolios.** A factor portfolio's stop is the rebalance.
- The `run()` harness refills slots only when empty and uses `rebalance=5` days. For factor
  tests you need a different loop: full re-rank every N days, sell what leaves the top set,
  buy what enters, no per-position exit logic.

## 4. Statistical traps and how much to discount +56%

**Noise floor of a 5-name portfolio.** Indian midcap single-name vol ~30-35% annualised,
pairwise correlation ~0.3. A 5-name equal-weight book has ~13-15% idiosyncratic vol on top of
beta; tracking error vs the 124-name basket is ~15-18%/yr. Over 3.3 years the standard error
of *total* alpha is ~15-18% x sqrt(3.3) = **27-33 percentage points**. The +56% you observed
is therefore **~1.7-2.0 sigma** as a single draw.

**Multiple testing.** You swept 21 exit cells x 6 strategies (~126 combinations, more if you
count the earlier scanner and ML work) and took the max. The expected maximum of 126 pure-noise
draws is ~sqrt(2 ln 126) = **3.1 sigma**. A 2-sigma winner from a 126-cell grid is *below*
what luck alone would produce. Harvey-Liu-Zhu's threshold for a new factor is t > 3 *before*
worrying about your specific grid.

**Beta.** As in section 1c, momentum-above-SMA200 names in a 30%/yr bull carry extra beta
worth maybe 5-10%/yr of the gap. That is another ~20-30 points of the 56.

**Twenty trades.** Twenty is the number of coin flips; whatever the t-stat on per-trade net,
it cannot support any claim. Also 4 of those 20 are probably the same 2-3 names re-bought.

**Regime.** Momentum-with-trend-filter is long beta when the market is above SMA200 and in
cash when below. In a 2021-24 sample that is almost always long. The TEST window (2025 H1
midcap correction) is where this rule either earns its keep or gets whipsawed; the TRAIN
window cannot tell you which.

**Discount**: I would treat the +56% as **consistent with zero** true alpha. If forced to put
a number on the expected forward alpha of that exact rule: 0-3%/yr gross, minus costs, with a
-30% drawdown attached.

**Sample needed.** To detect a true alpha of a at tracking error TE with t=2:
years = (2 x TE / a)^2.

| portfolio | TE | alpha to detect | years needed |
|---|---|---|---|
| 5 names | 16% | 4%/yr | **64** |
| 5 names | 16% | 10%/yr | 10 |
| 30 names | 6% | 4%/yr | **9** |
| 30 names | 6% | 2%/yr | 36 |

You have 3.3 years of TRAIN and 1.6 of TEST. The only test design that can ever get you a
significant number with this dataset is a broad (30+ name) portfolio with a large expected
premium — i.e. momentum — and even that needs the full 5 years and a beta-adjusted
regression, and it will come in at t ~ 1.5. Everything else is unmeasurable here. Accept
that, and lean on the published evidence for the prior rather than on your backtest.

**Correct null to use.** You already have `random5`. Run it with 200-500 seeds under the
*identical* harness (same rebalance, hold, stops, window) and report the percentile of every
strategy against that distribution. That is a permutation test that automatically handles
the beta, the bull market and the survivorship bias. If "momentum 6m > SMA200" is not above
the 95th percentile of random-5 alpha, it is nothing. I would expect it to land around the
80th.

## 5. Cutting the -30% drawdown without giving up the premium

In order of bang for buck:

1. **Breadth: 5 -> 25-30 names.** This alone takes the idiosyncratic drawdown out. A 30-name
   momentum book's drawdown will track the factor's (which in India is still -30 to -40% in a
   crash; you cannot diversify that away, only hedge it). Costs go from 0.38% to 0.70% per
   round trip at Rs 2L — acceptable at quarterly turnover.
2. **Index-level regime filter, not stock-level.** Nifty Midcap 150 (or your EW basket) vs its
   own 200-day SMA, evaluated monthly, applied to the *whole* book (go 50% or 100% to
   liquid ETF / cash). Faber-style rules cut max DD by roughly a third to a half in
   backtests, cost ~1-2%/yr in bull markets from whipsaw, and are the only retail-feasible
   momentum-crash hedge. Evaluate monthly, not daily — daily evaluation is the whipsaw.
   The stock-level `> SMA200` filter you use now does something different and worse: it
   concentrates you further into the surviving names at exactly the wrong time.
3. **Volatility targeting** (Barroso-Santa-Clara 2015, Moreira-Muir 2017): scale exposure by
   target_vol / realised_vol (say 15% / trailing-60d vol), capped at 1. For momentum this
   roughly doubles Sharpe in US data and removes most of the crash. Retail implementation:
   vary the number of names / cash fraction at each rebalance rather than trading daily.
   Cheap and well supported; combine with (2).
4. **Blend momentum with low-vol** (rank on ret120 / vol60, which `mom_lowvol` already does,
   or a 50/50 rank average). Low-vol carries a natural short-crash tilt; the blend has
   materially better drawdown than either leg in Indian index data (Nifty Alpha Low-Vol 30
   backfills at about the momentum return with ~2/3 of the drawdown).
5. **Sector caps**: max 25% per sector, max 2 names per sector at 30 names. Necessary
   hygiene for Indian momentum, which periodically becomes 60% PSU banks / defence /
   capital goods. Small cost to the premium, large cut in the concentrated crash.
6. **Equal-risk (inverse-vol) weighting** within the book: modest DD improvement, basically
   free at quarterly rebalancing. Do it.

What does *not* work, and you already have the data: per-trade trailing stops. They convert
a factor bet into a series of short-horizon coin flips and pay a round trip each time.

## 6. What to test next, in priority order

Things already tried and closed (do not reopen): scanner score, 15m ML, exit-rule grids,
trailing stops, long-short, intraday horizons, per-trade take-profits.

1. **Fix the measurement before any new signal.**
   - Report true buy-and-hold benchmark alongside daily-EW.
   - Regress strategy daily returns on benchmark daily returns; report beta, intercept
     (annualised), t-stat of intercept, and information ratio. Make this the headline metric
     instead of `alpha_total_pct`.
   - Random-N null with 300 seeds under identical settings; report percentile.
   - Check how many of the 124 names existed / were mid-cap in 2021; if you can, rebuild the
     universe from a point-in-time index constituent list (NSE publishes historical Nifty
     Midcap 150 changes) to remove survivorship.
   Cost: an afternoon. Value: it probably ends the 5-name research programme by itself.

2. **Broad quarterly momentum, no stops.** 12-1 momentum, top 25-30 of the universe,
   inverse-vol weights, quarterly rebalance with an enter-top-20% / exit-below-40% buffer,
   sector cap 25%. Measure on TRAIN with the corrected metrics and the random-30 null. This
   is the only strategy family in the repo with a credible prior. Expect: beta ~1.1, gross
   alpha 2-6%/yr, t ~1-1.5, max DD comparable to the benchmark.

3. **Same book with (a) index-level SMA200 monthly regime filter and (b) 15% vol targeting.**
   Compare drawdown and alpha give-up. Then combine. This is the drawdown question in
   section 5, answered with your own data.

4. **Broad low-vol (30 names, semi-annual) and the 50/50 momentum + low-vol rank blend.**
   Cheap given the code you have. Expect low-vol to *lose* to the benchmark on TRAIN (it was
   a beta market) and win on drawdown; that is the correct result, not a failure.

5. **Quality from `fundamentals.sqlite`**: ROE, accruals, debt/equity, point-in-time (lag
   filings 60-90 days or you will leak). Rank blend with momentum. Only after 2-4 are done
   and only if the fundamentals data is genuinely point-in-time; otherwise skip.

6. **Run TEST exactly once**, on the one configuration you would actually trade, after
   pre-registering its expected beta, alpha and DD. If it is inside the random-30 90%
   interval on TEST, the conclusion is section 0, bullet 4.

What I would not spend time on: more ML on short horizons, sentiment from RSS as a
selection signal (horizon mismatch with delivery costs; the news that moves midcaps is
priced in the opening auction), anything intraday, anything with more than ~4 free
parameters, anything evaluated with fewer than 25 names.

## 7. The honest bottom line

At Rs 2 lakh, delivery costs, 124 names, 5 years of data (3.3 of them a one-directional bull
market) there is **no measurable stock-selection edge in this repository, and no realistic
prospect of measuring one**. The 15m model, the scanner and the exit grid are all consistent
with zero. The +56% momentum cell is consistent with zero plus beta plus a 126-way max.

What *is* real, and the evidence for it is external rather than in your backtest: a
long-only, broad, quarterly-rebalanced Indian momentum tilt, ideally with a low-vol blend and
an index-level regime or vol overlay. Expected value over cap-weight: a few percent a year,
with drawdowns you must be prepared to sit through and a t-stat you will not be able to
verify for a decade. You can get most of it from Nifty200 Momentum 30 / Nifty Alpha Low Vol
30 index funds for 0.3%/yr and no DP charges. Building it yourself is a defensible hobby and
a way to hold a sector-capped, vol-targeted version the index funds do not offer; it is not a
way to beat the market by the margins the TRAIN numbers suggest.
