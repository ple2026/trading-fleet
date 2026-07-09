# Step 2 result — a trend hedge sleeve for v20 (honest version)

Pre-registered in `docs/HANDOFF.md §Next steps #2`. Executed 2026-07-02. Follows
Step 1 (MACRO barely beat cash as v20's defensive half); goal was a *higher-quality*
defensive sleeve. This doc leads with the vs-CASH benchmark and the unlevered,
implementable number — an earlier draft oversold it with a free-leverage headline.

```bash
python -m research.scripts.trend_sleeve --no-equity --out data/trend_noeq_oos_equity.csv
python -m research.scripts.trend_trades       # actual positions + P&L attribution
python -m research.scripts.compare_sleeves     # cash vs MACRO vs TREND vs TREND_NE
```

## Bottom line (read this first)

**Most of the risk reduction from any "defensive half" is just holding less v20.**
Swapping that half from CASH into the trend sleeve (TREND_NE) adds a real but
*incremental* edge — meaningfully bigger than MACRO's, not a transformation.

At ~half of v20's volatility (~20.7%):

| ~half-risk strategy | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| v20 alone (full) | 68.8% | 1.47 | −47.6% |
| 50% v20 + 50% **CASH** ("just hold half") | 34.0% | 1.52 | −25.4% |
| 47% v20 + 53% **TREND_NE** (the blend) | **37.4%** | **1.64** | **−19.0%** |

**Drawdown decomposition** (v20 −47.6% → blend −19.0%, a −28.6pp cut):
- **−22.2pp is de-risking** (holding half — CASH gets you this).
- **−6.4pp is the actual hedge** (TREND_NE beyond cash).
→ ~78% of the drawdown improvement is "hold less v20"; ~22% is the sleeve.

**TREND_NE's true marginal value over cash: +3.3pp CAGR, +0.12 Sharpe, +6.4pp DD.**
That is ~3× MACRO's marginal-over-cash (+1pp CAGR / +0.03 Sharpe) — so "find a better
MACRO" succeeded in relative terms — but it is an increment on top of de-risking, not a
different animal. Part of the raw "1.47 → 1.64" is the mean/vol Sharpe artifact (any
low-vol positive sleeve inflates it); the clean vs-cash figure is **+0.12 Sharpe**.

## Implementable, no leverage, no timing

You do **not** switch between v20 and the hedge — you hold both, always, at fixed
weights, and rebalance mechanically. Rebalancing frequency barely matters:

| 47/53 blend | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| daily rebal, 5 bps cost | 37.2% | 1.64 | −19.0% |
| **monthly rebal, 5 bps cost** | **37.8%** | **1.65** | **−19.1%** |
| quarterly rebal, 5 bps cost | 38.1% | 1.62 | −19.8% |

So capital assignment is trivial: monthly rebalance to 47/53, negligible cost, no
forecast. This row — **~38% CAGR / Sharpe 1.65 / −19% DD** — is the real deliverable.

## The sleeve — vol-targeted long/short managed-futures replication (on ETFs)

Textbook Moskowitz-Ooi-Pedersen TSMOM, applied **blind** with literature-standard
params (no fit to this data → OOS by construction): signal = mean(sign of 3/6/12-month
return); size = signal / vol; scaled to 15% ex-ante (covariance) vol; **long AND
short**; monthly rebalance; 10 bps cost; strictly point-in-time. **TREND_NE** = 8
non-equity ETFs (TLT, IEF, GLD, SLV, DBC, USO, DBA, UUP); avg gross 2.5×.

| standalone, 2007-2026 | CAGR | Sharpe | MaxDD | corr to v20 |
|---|---|---|---|---|
| MACRO (Step 1) | 3.5% | 0.88 | −6.9% | −0.00 |
| **TREND_NE** | 8.8% | 0.57 | −44.9% | **−0.11** |

Clears the bar (OOS Sharpe > 0.5, |corr| < 0.3). Note it is a *thin* standalone book —
its worth is the negative correlation, not its solo return.

## Why it hedges — the actual trades

TREND_NE's P&L is concentrated in safe-haven/commodity trends: **GLD +57.9%** (biggest,
~30% of gross), DBA +34, IEF +30, UUP +25, DBC +22. **324 trades, only 108 winners /
216 losers** — wrong 2/3 of the time, losers cut tiny, winners run (the trend-following
signature). Biggest trades: long GLD 2009-12 & 2023-now, SHORT DBC/USO in the 2014-16
commodity crash, long commodities in the 2021-22 inflation, long bonds into the GFC/COVID.

The mechanism, concretely: **on v20's 40 worst days (avg −10.1%), TREND_NE averaged
+1.17% and was up on 27 of 40**; on v20's 40 best days it averaged only −0.75%. It
cushions the downside and barely gives up the upside — because when 3× Nasdaq craters,
money flows into gold/bonds/dollar, which the sleeve already holds.

## "Keep improving it" — the textbook improvements are LATERAL

Three pre-registered a-priori variants:

| sleeve | solo Sharpe | corr to v20 | blend Sharpe | blend MaxDD |
|---|---|---|---|---|
| **NE1** (orig 8 ETFs) | 0.57 | −0.11 | 1.64 | **−19.0%** |
| NE2 (16 ETFs + signal ensemble) | 0.80 | −0.04 | 1.68 | −26.1% |
| SAFE (bonds+gold+$ only) | 0.79 | −0.13 | **1.69** | −25.1% |

- The **signal ensemble** (multi-horizon momentum + MA-cross + channel) on the same 8
  ETFs was a **no-op** (Sharpe 0.53). All of NE2's standalone gain came from **breadth**.
- But breadth **diluted the hedge** (corr −0.11 → −0.04), so NE2's blend *drawdown* got
  *worse*. SAFE has the deepest avg corr and best blend Sharpe, yet NE1 still gives the
  best blend *drawdown*, because blend DD is set by v20's worst window (2020-22, an
  *inflation* bear) which NE1's broad-commodity leg offset best.
- **Lesson: for a hedge, optimise the correlation/tail behaviour, not standalone
  Sharpe.** NE1's original universe (deflation hedge = bonds/gold + inflation hedge =
  commodities, no FX) is already near the sweet spot. Further universe-spinning would be
  overfit-by-iteration — stopped after three a-priori variants.

## If you insist on matching v20's return: leverage (with caveats)

The unlevered blend gives up raw CAGR (38% vs 69%). Lever it ×2 to v20's vol:

| levered ×2 to v20's 41% vol | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| **0% financing (fantasy)** | 81.2% | 1.64 | −36.6% |
| 3% financing | ~76% | 1.57 | −37.7% |
| 5% financing | ~72% | 1.52 | −38.5% |

Even at 5% it edges v20 on all three — but this stacks ×2 on top of v20's internal 3×
and the sleeve's 2.5× gross, held through a −37% drawdown without a margin call. Treat
the leverage-invariant **Sharpe (1.64 vs 1.47)** as the honest core; the CAGR is a
leverage restatement, not independent evidence. **Do not lead with the 81%.**

## Verdict

TREND_NE is a **modestly better-than-cash hedge** for v20 — real, ~3× MACRO's marginal
edge (+0.12 Sharpe / +3pp CAGR / +6pp DD over cash), regime-robust (negative corr in
4/5 sub-periods), and fully implementable unlevered with monthly rebalancing and no
timing. It is **not** a way to keep v20's returns at lower drawdown without leverage,
and most of a de-risked book's drawdown reduction is simply holding less v20. Promote
it over MACRO as the diversifying sleeve; size it as a genuine-but-incremental
improvement, not a transformation.

### Honest caveats
- In-sample era (2007-26 = v20's construction window); the **negative correlation** is
  the OOS-robust load-bearing claim, not the magnitudes.
- Thin standalone (Sharpe 0.57); vulnerable to trend droughts (2010-14 lagged cash).
- ~30% of P&L is gold; the biggest trade (long GLD 2023→now) is open/unrealized.
- Leverage/financing/margin-call risk unmodeled beyond the flat-rate sensitivity above.
- Does not repair v20's own return-level fragility.

### Files
- `research/scripts/trend_sleeve.py` (`--no-equity`, `--symbols`), `trend_sleeve_v2.py`
  (`--narrow`), `trend_trades.py`, `compare_sleeves.py`.
- Curves (gitignored): `data/trend_noeq_oos_equity.csv`, `trend_ne2_oos_equity.csv`,
  `trend_ne1b_oos_equity.csv`, `trend_safehaven_oos_equity.csv`.
