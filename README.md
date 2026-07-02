# Trading Fleet

Four self-improving trading bots — each a modern take on a legendary trader —
competing for capital on one shared platform with a disciplined improvement loop.

> **NOT FINANCIAL ADVICE. Research/education only.** Runs on **paper** until every
> bot clears its promotion gates. There is no "best" strategy; there is only
> measured, out-of-sample edge.

| Bot | Lineage | Style | Horizon | Fits $10k? |
|-----|---------|-------|---------|-----------|
| **BREAKOUT** | Livermore × O'Neil | Pivotal-point breakouts in leading stocks (CAN SLIM + pyramiding) | weeks–months | ✅ cleanly |
| **ARB** | Ed Thorp | Cointegration / pairs stat-arb, market-neutral | days | ⚠️ research pod |
| **CATALYST** | Steve Cohen | PDT-safe swing trading around catalysts (mosaic score) | 3–15 days | ✅ adapted |
| **MACRO** | Stan Druckenmiller | Concentrated top-down macro via ETFs, conviction-scaled | weeks–months | ✅ via ETFs |

Ed Thorp also serves as the fleet **treasurer**: quarter-Kelly money management
sizes every position and allocates capital across bots by measured edge.

## Architecture

```
L3 META      allocator · correlation monitor · fleet reports
L2 BOTS      breakout · arb · catalyst · macro   (one Bot interface)
L1 PLATFORM  data · journal · backtest (walk-forward) · regime · kelly
```

- **Research / backtest / improvement** → Python (`research/`): pandas, statsmodels
  (cointegration), scikit-learn (regime GMM, mosaic re-fit).
- **Execution / dashboard** → TypeScript (`execution/`): Alpaca paper trading,
  ledger-enforced per-bot sub-books.
- **Journal** → Postgres (Neon), with JSONL fallback offline. Every decision —
  *including rejected ones* — is logged for the improvement engine.

See [`docs/FLEET_PLAN.md`](docs/FLEET_PLAN.md) for the full plan: bot specs, the
three-tier self-improvement loop, the anti-overfitting constitution, promotion
gates, and the build sequence.

## Quickstart (offline, zero credentials)

```bash
pip install -e ".[dev]"                    # or: pip install pandas numpy statsmodels scikit-learn scipy
python -m research.scripts.backtest        # all four bots on synthetic data
python -m research.scripts.backtest --bot breakout --start 2021-01-01 --end 2024-12-31
python -m research.scripts.improve         # the improvement run: journal→Tier C→Tier B→allocation
python -m research.scripts.improve --tune  # + Tier-A walk-forward parameter re-fit (slow)
```

With no `ALPACA_*` keys the data plane falls back to a deterministic synthetic
generator, so the whole fleet runs and backtests out of the box. Synthetic results
are a **plumbing check, not an edge estimate** — wire real data (below) before
drawing any conclusion.

## Going live-ish (real data → paper trading)

1. `cp .env.example .env` and fill in `ALPACA_*` (paper), `DATABASE_URL` (Neon),
   and the research vendor keys (`FMP_API_KEY`, `FINNHUB_API_KEY`, `FRED_API_KEY`).
2. `psql "$DATABASE_URL" -f db/schema.sql` to create the journal.
3. Backtest each bot on real bars; only bots clearing the promotion gates in
   `docs/FLEET_PLAN.md §10` advance to paper.
4. 30-day paper burn-in per bot before any real capital.

## The self-improvement loop (short version)

Not RL on P&L. An **evidence flywheel**:

```
journal every decision → analyze outcomes → propose bounded changes
     → shadow-test on paper → promote winners → repeat
```

- **Tier A** (weekly, automated): walk-forward parameter tuning within hard bounds;
  accepted only on out-of-sample gates; ships as a PR you merge.
- **Tier B** (weekly, human-gated): an LLM reads the trade journal + counterfactuals
  and proposes ONE bounded rule change as a draft PR with backtest evidence; you
  approve/reject; approved rules run as shadow challengers before promotion.
- **Tier C** (real-time): each bot only trades in regimes where the journal shows it
  works.

## Layout

```
research/
  fleet/
    types.py        indicators.py   bot.py          # shared contracts
    data.py         regime.py       kelly.py        # platform
    journal.py      backtest.py                     # platform (journal-wired)
    walkforward.py                                  # walk-forward + Tier-A tuner
    allocator.py                                    # L3 meta-allocator + corr monitor
    improve.py                                      # Tier-B proposals + Tier-C gates
    theses.py                                       # MACRO thesis 60/120d scoring
    promotion.py                                    # champion/challenger shadow test
    bots/
      breakout.py   arb.py          catalyst.py     macro.py
  scripts/
    backtest.py     improve.py                      # backtest / the improvement run
  tests/                                            # offline, no credentials
db/schema.sql                                       # journal (Postgres)
execution/                                          # TS Alpaca executor (phase 2)
docs/FLEET_PLAN.md
```

The improvement engine is wired end-to-end offline: `run_backtest(..., journal=…)`
records every decision (taken **and** rejected) plus closed-position outcomes;
`walkforward.tune` re-fits parameters against the §7b OOS gates; `allocator.allocate`
sizes bots by risk-parity + quarter-Kelly Sharpe tilt with the §8 correlation
freeze; `improve` turns the journal into Tier-C regime recommendations and
bounded Tier-B rule-change proposals; `theses` scores MACRO's falsifiable theses at
60/120 days; and `promotion` shadow-validates an accepted challenger on a held-out
window before it can replace the champion. ARB is modelled as a true two-leg,
beta-neutral pair. All of it runs on synthetic data with no credentials — a
plumbing check, not an edge estimate.
