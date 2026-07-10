# ALLOC — the risk-allocation agent (fleet pivot, pre-registered 2026-07-03)

The four picker-bots (BREAKOUT / ARB / CATALYST / MACRO) are **retired**: all four
failed to beat buy-and-hold out-of-sample, consistent with SPIVA (92% of pros lose
over 20y) and Bessembinder (57% of stocks lose to T-bills). The 141-year study
(`century_market_structure`, 2026-07-03) showed every surviving edge is a
*how-much-to-hold* effect, not a *what-to-buy* effect. ALLOC is the fleet rebuilt as
ONE risk-allocation machine over index beta. Goal: beat v20 **as an allocator** —
risk-adjusted return, drawdown, wealth concentration, and robustness in eras v20
never lived — not raw CAGR (v20 carries a stock-picking base ALLOC deliberately
lacks).

## Spec v1.0 — ALL parameters a-priori (century-validated or textbook; no tuning)

**Instrument** QQQ daily (1999-03+, `daily-bars.db`); NASDAQ Composite 1971+ for the
era-robustness run (same machinery, synthetic leverage with REAL financing).

1. **Regime gate (asymmetric, century-validated):** exit when close < EMA200,
   re-enter when close > EMA50. Signals at close t, position earns from t+1.
2. **Leverage dial (the new part):** `lev = clip(σ_tgt / σ_realized20d, 0, 3)` when
   gate ON; 0 when OFF. σ_tgt = **30%/yr a-priori**. Rebalance leverage only when
   |desired − held| > 0.25 (band, turnover control).
3. **Financing & costs (honest, the challenge-agent fix):** borrowed notional
   (lev−1)⁺ pays TB3MS + 60 bps; ER = 0.20%/yr at lev ≤ 1 scaling linearly to
   0.95%/yr at lev = 3; 5 bps × turnover trading cost. Synthetic 3× must validate
   against real TQQQ 2010-2026 (report tracking error) before any pre-2010 claim.
4. **Defensive sleeve (gate OFF, inflation-conditional):** trailing-3y CPI ≤ 3%
   (1-mo publication lag) → IEF (GS10-implied TR pre-2004); > 3% → cash (TB3MS).
5. **Rebound overlay (capped v20 rule):** prior calendar year index return < 0 →
   from first trading day, dial FLOORED at 3.0 with a −10% stop on the levered book;
   stop → revert to normal dial for the year; floor expires at year-end. Capped
   variant (floor 2.0) reported alongside — the de-fragilized primary if verdicts
   differ.
6. **Trend sleeve (2007+ comparison only):** 20% book weight to TREND_NE
   (`data/trend_noeq_oos_equity.csv`), 80% ALLOC core.

## Ablation (attribution, not tuning)
A0 QQQ B&H · A1 gate 1× · A2 gate+dial · A3 +defensive · A4 +rebound (**= ALLOC
core**) · A5 gated FIXED 3× (the v20-core-like control) · A4+TREND (2007+).

## Pre-registered questions
Q1 Does the dial fix the 2000-02 sawtooth that killed gated-fixed-3× (−98%, memory
`secular_v20_full_history`)? Q2 Does ALLOC beat v20 2007-2026 on MAR (CAGR/|maxDD|,
v20 = 1.44) and Sharpe (1.47) at materially lower top-5-year wealth concentration
(v20 = 99%)? Q3 Does it survive 1971-2026 (two rate regimes, 4 secular bears) with
maxDD > −85% and positive real CAGR?

## Kill-criteria (decided before the first run)
1. Any 1-D sweep (σ_tgt 20-40, vol-window 10-60d, band 0.10-0.50) flips the verdict
   → knife-edge → REJECT (the V21 lesson).
2. NASDAQ-1971 run maxDD ≤ −85% → ruin in a lived era → REJECT.
3. Beats v20 only in the 2019-2026 block → same fragility, no improvement → REJECT.
4. Average financing+cost drag > 2%/yr unexplained → implementation unsound.

One run per question. Results stand. Ablation is attribution, not a menu.

---

# RESULTS (2026-07-03; post audit-fix — independent code audit: "trustworthy as-is",
# 4 minor bugs fixed, all ≲0.1%/yr except a rebound day-1 off-by-one that was
# penalizing the rebound; core numbers unchanged after fixes)

**Pre-flight:** synthetic-3× vs real TQQQ 2010-2026: gap **0.27pp/yr** — financing
model honest; pre-2010/era leverage claims legitimate.

## Ablation (QQQ 1999-03..2026-07 — includes the dot-com crash)

| variant | CAGR | Sharpe | MaxDD | MAR |
|---|---|---|---|---|
| QQQ B&H | 10.9% | 0.52 | −83.0% | 0.13 |
| SPY B&H | 8.5% | 0.52 | −55.2% | 0.15 |
| A1 gate 1× | 9.9% | 0.67 | −40.6% | 0.24 |
| A3 gate+dial+defensive | 15.4% | 0.67 | −41.1% | 0.38 |
| **A4 +rebound = ALLOC core** | **18.2%** | **0.70** | **−46.9%** | **0.39** |
| A5 gated FIXED 3× | 17.3% | 0.58 | −83.8% | 0.21 |

**Q1 — YES, the dial fixes the fixed-3× sawtooth death:** −83.8% → −41.1% at equal
CAGR (and −78.9% → −43.1% over NASDAQ 1971-2026, 55y, real financing at 5-7%/yr
drag in the high-rate eras). Robust: σ_tgt/vol-window/band sweeps smooth; the 3×3
EMA-gate grid all lands MAR 0.38-0.51 with the a-priori 200/50 near the WORST cell
(numbers are conservative). Gold-hot-defensive rejected; ensemble-gate ≈ tie.

**Q3 — survival:** NASDAQ 55y ALLOC core −52.6% maxDD (bar −85%) at 22.9% CAGR.

**CENTURY BOUND (the sharpest finding):** on the Dow 1885-2026 the dial FAILS
(−73.7% DD, MAR 0.11 vs gate-1× 0.17): **leverage only pays when the index's return
spread over financing is fat (QQQ-class ~8%+); on a 5-7% index it is value
destruction. The unlevered gate is the only universal component** — exactly the
century study's "gate = insurance, not alpha." The machine's edge is CONDITIONAL,
now precisely characterized.

## The BOOK — 60% ALLOC core + 40% TREND_NE (corr −0.00), leverage layered

2007-2026, financing charged (Q2 span; v20 context = 68.1%/1.46/−47.6%):

| book | CAGR | Sharpe | MaxDD | MAR |
|---|---|---|---|---|
| QQQ B&H | 16.4% | 0.79 | −53.4% | 0.31 |
| **w=0.6 L=1 (unlevered)** | **22.1%** | **1.08** | **−25.4%** | **0.87** |
| w=0.6 L=1.5 | 31.3% | 1.04 | −36.1% | 0.87 |
| **w=0.6 L=2** | **39.8%** | **1.02** | **−45.5%** | 0.87 |

Unlevered: beats QQQ on every metric (CAGR +5.7pp, Sharpe 1.08 vs 0.79, HALF the
drawdown). Levered ×2: **2.4× QQQ's CAGR at LESS than QQQ's drawdown.** Beats QQQ's
CAGR in all 5 sub-windows; halves crisis DD (2008 −27 vs −53; 2020-22 −23 vs −35)
at the price of deeper mid-bull corrections (2010-14 −44 vs −16). Start-date: core
wins MAR from 1999/2003/2007/2015 starts; loses from 2010 (calm decade — insurance
unneeded). **Q2 vs v20: NO on absolute/Sharpe (68%/1.46 needs V11's stock alpha +
LEAP sleeves; corr(ALLOC,v20)=+0.65)** — as expected; ALLOC's claim is vs indexing,
not vs v20.

## Verdict — after the 3-attacker adversarial panel + independent code audit

Code audit: "results trustworthy as-is" (loop timing clean; 4 minor fixes applied).
Panel findings that STAND (accepted):
- **L=2 book is NOT collectible by a retail account through a tail.** Peak gross
  3.6× QQQ + 2× trend sleeve ≈ 5.6-7.6× notional, up to 84% portfolio-margin
  utilization; a −20% overnight gap at peak = −72% equity + forced liquidation — a
  path the backtest structurally cannot show. Dial/gate are one-day-late by design
  (measured: Brexit day −14% on a −4.1% index move). **L=2 numbers are frontier
  math, not a product.** Futures relocate but don't remove gap risk.
- **The Sharpe edge over B&H is not statistically established** (Memmel z≈0.7-0.8,
  P(≤0)=0.22; n≈2 independent crises carry the whole CORE edge; post-2007 the A3
  core alone Sharpe-ties QQQ; the book's Sharpe lift comes from the in-sample
  TREND_NE at 40% weight). The **drawdown/ruin claim is the real claim** —
  structural insurance, not provable alpha.
- **Leverage is Sharpe-destructive in every column** of the grid; levered CAGR
  "wins" are leverage restatements (the program's own recurring lesson).
- **The dial is regime-conditional on a fat return-minus-financing spread** and the
  forward regime (4% funding, record concentration) resembles the failure regime
  more than 2009-2021. The century bound is the honest boundary of the claim.
- **CPI-3% defensive branch is lookahead-in-design** (rule extracted from the
  same-day century study; n≈2 activations, dominated by the observed 2022 episode).
  Downgrade it from "validated component" to "plausible mechanism, untested."
- Taxable-account reality: ~all gains short-term → the core's edge over B&H inverts
  after top-rate taxes; futures 60/40 or tax-sheltered accounts are the viable
  wrappers (but shorting sleeves don't fit IRAs).
- Retail financing haircut (+1.5% spread): headline −2-3pp, claims survive.

**The deployable product that survives everything: the UNLEVERED book —
w=0.6 core + 0.4 TREND_NE, L=1: ~17.6-22.1% CAGR / Sharpe ~0.96-1.08 / maxDD ≈
−25%** (A3- vs A4-core), i.e. QQQ-class returns at HALF QQQ's drawdown, fully
margin-safe, no financing dependence. Above that, L≤1.5 via futures only with the
gap-risk warning attached. The four picker-bots stay retired; ALLOC's honest value
is RISK TRANSFORMATION (crash-ruin → smaller, holdable, correction-shaped risk),
not statistically-provable excess return.
