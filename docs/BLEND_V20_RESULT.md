# Step 1 result — blend v20 + MACRO + index-beta vs v20 alone

Pre-registered in `docs/HANDOFF.md §Next steps #1`. One run, results stand. Executed
2026-07-02 on the v20 machine (`ple2026/trading-agent` local, with its gitignored
`backtest-output/` present). Reproducible via:

```bash
python -m research.scripts.gen_macro_curve --start 2004-01-01 --end 2026-07-01
python -m research.scripts.blend_v20
```

## Inputs (all real data)

| Sleeve | Source | Span | CAGR | Sharpe | MaxDD | Vol |
|---|---|---|---|---|---|---|
| **v20** | `trading-agent/backtest-output/v20/equity-layers.csv` (`v20_equity`) | 2007-01-04 → 2026-07-01 | 68.7% | 1.47 | −47.6% | 41.4% |
| **MACRO** | `gen_macro_curve.py` — walk-forward OOS, FRED-driven | same | 3.5% | 0.87 | −6.9% | 4.0% |
| **SPY** | Tiingo b&h | same | 10.9% | 0.63 | −55.2% | 19.6% |

MACRO reproduces the handoff verdict to the decimal (2007-24: Sharpe **0.88**, MaxDD
**−6.9%**, CAGR **3.5%**). Note: the OOS must begin in 2007 to cover v20's span — so
the FRED/ETF panel is loaded from **2004** and the first 3 years are burned as
walk-forward training. Loading only from 2007 pushes the OOS to 2010 and drops the
GFC (Sharpe falls to 0.71) — an easy off-by-a-training-window trap.

## Correlations (daily returns, 2007-2026)

```
corr(v20, MACRO) = -0.004     <- genuine diversifier: v20's timing overlay strips beta
corr(v20, SPY)   = +0.630     <- v20 is a QQQ-timing book; SPY is redundant beta
corr(MACRO, SPY) = -0.138     <- matches the handoff's -0.15
```

## The honest test: MACRO vs just holding CASH for the defensive half

"Halve v20's drawdown by holding 50% v20" is trivial — 50% cash does it too. The only
question that matters is whether MACRO beats **cash** as the defensive sleeve. At
matched risk (~20.7% vol) it does, but **only marginally**:

| 50% v20 + … | CAGR | Sharpe¹ | MaxDD |
|---|---|---|---|
| … **cash** (~2%) | 34.0% | 1.52 | −25.4% |
| … **MACRO** | 35.0% | 1.55 | −24.3% |

MACRO's true marginal edge over cash: **≈ +1pp CAGR, +0.03 Sharpe, ~1pp less DD.**
Most of the "halved drawdown" below is just *holding less of the aggressive book*, not
diversification. Two consequences:

- **MACRO cannot deliver v20's return at lower drawdown.** It is low-vol / low-return,
  so meaningful diversification requires holding a *lot* of it, which drags return
  toward its own 3.5%. Keeping v20's return level while diversifying needs leverage —
  and that stacks leverage on an already-3× book. Not realistic.
- ¹ The frontier Sharpe lift (1.47 → 1.55) is **partly a raw-Sharpe artifact**: these
  numbers use mean/vol, not excess-of-cash, so blending in *any* low-vol positive
  sleeve (cash included) inflates the ratio. Against the fair cash benchmark, MACRO's
  real edge is the +0.03, not +0.08.

**Sober verdict:** MACRO is a *slightly-better-than-cash* home for the defensive half
of a v20 book — genuinely uncorrelated (corr ≈ 0.00) and positive-Sharpe, so it beats
cash, but modestly. It lets you **de-risk without going fully to cash**; it is **not**
a way to beat v20's returns. The tables below stand as measured — read them through
this cash-benchmark lens. The takeaway for the fleet: the payoff scales with the
*quality* of the defensive sleeve, which motivates Step 2 (a higher-Sharpe long/short
multi-asset trend program to replace this thin MACRO).

## Verdict — MACRO diversifies v20; SPY does not

**1. The pre-registered allocator blend (risk-parity + ¼-Kelly + [10,40]% caps, monthly, point-in-time)**

| | CAGR | Sharpe | MaxDD | Vol |
|---|---|---|---|---|
| v20 alone | 68.7% | 1.47 | −47.6% | 41.4% |
| allocator blend | 23.2% | **1.37** | **−24.3%** | 16.2% |

Avg weights v20 0.28 / MACRO 0.40 / SPY 0.32; corr-freeze fired 88/230 months. The
allocator **halves drawdown but does NOT beat v20 on Sharpe** — risk-parity pins v20
near its 10% floor and force-feeds low-Sharpe SPY, which drags the ratio. Risk-parity
optimises equal-risk-contribution (capital preservation), not Sharpe. Correct tool,
wrong objective for "beat v20."

**2. Fixed v20 + MACRO frontier (drop SPY — the optimiser zeroes it anyway)**

| w_v20 / w_MACRO | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| 1.00 / 0.00 (v20) | 68.8% | 1.47 | −47.6% |
| 0.70 / 0.30 | 48.4% | 1.51 | −34.1% |
| 0.60 / 0.40 | 41.7% | 1.53 | −29.3% |
| **0.50 / 0.50** | **35.0%** | **1.55** | **−24.3%** |

Sharpe rises **monotonically** as MACRO's share grows (1.47 → 1.55) while MaxDD more
than halves — **no leverage required, no knife-edge**. Adding SPY to any of these
*lowers* Sharpe (it is +0.63 correlated redundant beta). The in-sample max-Sharpe
tangency puts **0% on SPY**, 86% on MACRO, 14% on v20 → Sharpe **1.72** (the ceiling).

**3. Sub-period robustness — the edge is structural, not one 2008 crisis**

| window | v20 Sharpe | v20 DD | 50/50 Sharpe | 50/50 DD | corr(v20,MACRO) |
|---|---|---|---|---|---|
| 2007-2009 | 0.96 | −31.0% | **1.24** | **−14.3%** | −0.14 |
| 2010-2014 | 0.89 | −38.2% | **0.98** | **−19.7%** | −0.16 |
| 2015-2019 | 1.57 | −30.5% | **1.59** | **−16.2%** | +0.19 |
| 2020-2022 | 1.65 | −47.6% | **1.76** | **−24.3%** | −0.19 |
| 2023-2026 | 2.22 | −33.1% | 2.21 | **−18.1%** | +0.32 |

Every window: the blend **roughly halves drawdown** and holds-or-beats Sharpe. Corr
goes mildly positive only in the calmest bulls (2015-19, 2023-26) — MACRO leans
risk-on then — but the drawdown protection survives even there because MACRO's low
vol dominates the blend's tail.

## Bottom line

**How do we do better than v20?** Honestly: MACRO alone doesn't move the needle much.
Holding **~50-60% v20 + ~40-50% MACRO (skip SPY)** cuts drawdown roughly in half
(−48% → −24 to −29%) at a still-strong 35-42% CAGR — but ~all of that de-risking is
available from cash too; MACRO beats cash as the defensive half by only **≈ +1pp CAGR
/ +0.03 Sharpe** (see the cash-benchmark section). SPY is worse than either (redundant
+0.63-corr beta; every optimiser zeroes it). So the concrete answer is modest: MACRO
is a *slightly-better-than-cash* diversifier — worth holding over cash, not a way to
beat v20's returns. The real lever is a **higher-quality defensive sleeve** (Step 2).

The improvement is **diversification (drawdown), not extra selection return** — which
is exactly the fleet's meta-finding: MACRO's value is its ~0 correlation, not its 3.5%
CAGR. It converges with v20's own "QQQ-plus with −50%+ DD" self-assessment: the −0.15
sleeve is its natural complement, and the measurement confirms it.

### Honest caveats

- v20's **return level** is itself discounted (rides the 2010-24 Nasdaq bull; ~⅔ of
  terminal wealth from ~5 trades; TQQQ never lived a 3× grinding bear). The blend
  reduces v20's drawdown pain and adds an OOS-validated uncorrelated sleeve — it does
  **not** repair v20's forward-return fragility.
- The blend is measured on 2007-2026, v20's own construction era. What is walk-forward
  OOS is **MACRO's positive Sharpe** and the **structural ~0 correlation**; v20's
  curve carries its own in-sample discount.
- Leverage on a book already holding 3× ETFs is not realistically financeable at
  scale; the vol-matched CAGRs are risk-adjusted illustrations. The **unlevered 50/50**
  (Sharpe 1.55, DD −24%, CAGR 35%) is the fully-deployable recommendation.

### Files

- `research/scripts/gen_macro_curve.py` — regenerate MACRO's walk-forward OOS curve.
- `research/scripts/blend_v20.py` — correlations, allocator blend, frontier, tangency,
  vol-match, sub-period robustness.
- `data/macro_oos_equity.csv`, `data/blend_report.csv` — the curves (gitignored).
