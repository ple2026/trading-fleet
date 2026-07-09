# HANDOFF — cross-machine continuity (written 2026-07-02)

Purpose: resume this work on the machine that has the **v20 repo local**
(`ple2026/trading-agent`, with its gitignored `backtest-output/` present). This doc
carries the session's measured verdicts, the v20 diagnosis, and the pre-registered
next steps, since chat context and per-machine memory do not transfer.

## Setup on the new machine

```bash
git clone https://github.com/ple2026/trading-fleet.git && cd trading-fleet
git checkout claude/improvement-engine        # PR #1 branch (or merge it first)
python3.13 -m venv .venv                      # needs Python >= 3.11
.venv/bin/pip install -e ".[dev]"
cp .env.example .env                          # then set the keys below — NOT in git
.venv/bin/python -m pytest research/tests -q  # 94 tests, ~8 min, fully offline
```

`.env` needs: `TIINGO_API_KEY` (adjusted prices; fundamentals are DOW-30 on the
free tier), `FRED_API_KEY` (macro), optionally `ALPACA_KEY_ID`/`ALPACA_SECRET_KEY`
(paper, secret still missing as of handoff). `data/` caches are gitignored and
rebuild on first real-data run (~10 min of Tiingo fetching).

## Measured verdicts (all real data; OOS = walk-forward out-of-sample)

| Strategy / experiment | Result | Verdict |
|---|---|---|
| BREAKOUT (O'Neil), 147-name survivorship-free OOS 2020-24 | −7.9% CAGR vs **EW same-universe +18.8%**, SPY +14.4% | **Dead.** Negative selection (~−27 pts/yr vs doing nothing). N-gate "fix" failed OOS and was reverted (`079ed09`) |
| ARB (pairs, 5 ETF pairs) OOS 2020-24 | corr to SPY −0.03, MaxDD −0.4%, but **3 trades/5yrs** | Right shape, inert. Needs ~100-pair breadth or factor-residual redesign to matter |
| Static regime rotation (20yr, pre-registered, `regime_portfolio.py`) | cash rule: DD −55%→−28% but Sharpe 0.62 vs SPY 0.63; TLT+GLD rule worse (2022 bonds fell WITH stocks) | Halves drawdown, does NOT beat SPY risk-adjusted. Slow signal; fixed defensive fails 2022 |
| **MACRO (FRED-driven dynamic rotation), 18y OOS 2007-24** | **Sharpe 0.88 vs SPY 0.59, corr to SPY −0.15, MaxDD −6.9%** (GFC −5.3 / COVID −3.4 / 2022 −4.8), CAGR 3.5% unleveraged | **The survivor.** The anti-correlated sleeve. Note: PF/hit stats undercount open-window positions; trust the curve metrics |
| CATALYST | Tier-A tuner accepted a param change; shadow test correctly held champion | Untested at depth; engine gates work |

Meta-finding (converges with v20's own `V11-vs-TQQQ` insight): **selection alpha
≈ 0; returns come from time-in-market × beta × timing × risk-sizing.** Both repos
proved this independently.

## v20 diagnosis (from `ple2026/trading-agent` docs, read 2026-07-02)

v20 = timed 3× Nasdaq (TQQQ/SQQQ + rebound bet + vol-targeting) on V11's regime
calendar. 19y book ≈ $2.6B, Sharpe 1.37, MaxDD −48%. Durability discounts (mostly
conceded in its own `V20-SYSTEM-EXPLAINED.md` Risks):
1. Return level rides the 2010-24 Nasdaq secular bull; never lived a 3× grinding
   bear (pre-2010 legs ran 1× QQQ — TQQQ launched 2010-02).
2. ~⅔ of terminal wealth from ~5 trades (rebound n=4, capitulation stop n=1).
3. Lineage of retracted look-ahead versions (V17.x) = genre base rate.
4. No multi-month real-money shadow.
Forward expectation: "QQQ-plus with −50%+ possible DD," sized as the aggressive
sleeve, never the core. **MACRO (corr −0.15) is its natural complement.**

## Next steps, in order (pre-register each; one run; kill or keep)

1. ~~**Blend experiment (needs the v20 machine).**~~ **DONE 2026-07-02 — see
   `docs/BLEND_V20_RESULT.md`.** Verdict: **v20 + MACRO (skip SPY) is a regime-robust
   Pareto win** — corr(v20,MACRO) ≈ **0.00** (SPY is +0.63 redundant beta and every
   optimiser zeroes it). A ~50/50 v20+MACRO blend **halves MaxDD (−48% → −24%) in
   every sub-period** while nudging Sharpe up (1.47 → 1.55), no leverage. The
   allocator's risk-parity choice halves DD too but does NOT beat v20 on Sharpe
   (1.37) — it optimises equal-risk-contribution, not Sharpe; for "beat v20" tilt to
   v20-heavy with MACRO as the diversifier. Improvement is diversification, not
   return; v20's return-level fragility is unchanged.
2. ~~**Multi-asset trend program**~~ **DONE 2026-07-02 — see
   `docs/STEP2_TREND_SLEEVE_RESULT.md`. Modest win (honest version).** Vol-targeted
   long/short TSMOM on ETFs, textbook params (OOS). **TREND_NE** (8 non-equity ETFs) =
   corr to v20 **−0.11**, robust across 4/5 sub-periods. But benchmarked against CASH
   (the honest test): a de-risked book gets ~78% of its drawdown cut just from holding
   *less v20*; TREND_NE's marginal edge OVER cash is **+3.3pp CAGR / +0.12 Sharpe /
   +6.4pp DD** — real, ~3× MACRO's, but an increment, not a transformation. Deliverable
   = unlevered 47/53 v20+TREND_NE, monthly rebalance, no timing: **~38% CAGR / Sharpe
   1.65 / −19% DD** (vs v20 69%/1.47/−48%). Leverage to match v20's return works but is
   the soft part (financing + stacked-leverage risk; DON'T lead with the 81% free-
   leverage number). Lesson: for a hedge, optimise **correlation/tail**, not standalone
   Sharpe — the breadth/ensemble "improvements" raised solo Sharpe but *diluted* the
   hedge (worse blend DD). Promote over MACRO; size as incremental.
3. **Cross-sectional 12-1 momentum** (monthly, top decile) on the
   survivorship-free universe (`universe.py`) — the honest factor test.
4. Bench: VRP via trading-agent's options infra (ORATS/LEAPS); EDGAR
   insider-cluster buying (plumbing exists in `edgar_fundamentals.py`/`smart_money.py`).

## Method constitution (what kept us honest)

Pre-registered rules; one run, results stand; walk-forward OOS or it didn't
happen; dual benchmarks (SPY b&h + equal-weight same-universe); point-in-time
data only (filed dates, publication lags); survivorship-free universes; costs
charged; reject failures including our own ideas (see the N-gate revert).
Acceptance bar for a new sleeve: **OOS Sharpe > 0.5 AND |corr| < 0.3** to
existing sleeves — it does not need to beat SPY solo; the allocator's blend is
the product.
