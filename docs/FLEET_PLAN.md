# Fleet Plan — Four Self-Improving Trading Bots

**Bots:** BREAKOUT (Livermore × O'Neil) · ARB (Thorp) · CATALYST (Cohen) · MACRO (Druckenmiller)
**Capital:** $10k per bot (paper first), one shared platform, one improvement engine.

---

## 0. Executive summary

We are building a miniature multi-strategy platform: four bots with deliberately
*uncorrelated* alpha philosophies, running on shared infrastructure, each keeping a
structured trade journal that feeds a disciplined improvement loop.

The core thesis on "self-improvement": **naive RL on P&L does not work in markets**
(too little data, non-stationary regimes, catastrophic overfitting). What works is an
**evidence flywheel**:

```
journal every decision → analyze outcomes → propose bounded changes
      → shadow-test on paper → promote winners → repeat
```

Three improvement tiers, in increasing order of human involvement:

| Tier | What changes | Who decides | Cadence |
|------|-------------|-------------|---------|
| A | Numeric parameters (thresholds, stops, lookbacks) | Automated walk-forward, hard acceptance gates | Weekly |
| B | Rules and features (new filters, new signals) | LLM proposes → human reviews → shadow test | Weekly |
| C | Deployment (which regimes each bot may trade in) | Regime engine, learned from journal | Real-time |

The human never leaves the loop for Tier B. That is the "teach it to improve" part:
you review proposals like a PM reviews an analyst, and your accept/reject decisions
are themselves logged and become training signal for better proposals.

---

## 1. Where we are today (repo audit)

The existing codebase is ~60% of Bot 1 plus early platform primitives:

| Existing file | Role in fleet architecture |
|---|---|
| `lib/agent.ts` (`runTick`) | Becomes `bots/breakout/` — the Livermore × O'Neil bot's engine |
| `lib/scanner.ts` (`scoreLeaderRescan`) | BREAKOUT signal engine — Profile A is already "pivotal point" detection (pocket pivots, volume dry-up, tight consolidation near 52wk highs) |
| `lib/strategies/ensemble.ts` (`isRiskOn`) | Regime service v0 (2-gate SPY filter) — MACRO bot upgrades this into a proper macro state vector, shared fleet-wide |
| `lib/risk.ts` | Risk kernel v0 — fixed-fractional sizing + daily kill switch; gains a Kelly layer (Thorp) |
| `lib/llm-overlay.ts` (`llmVeto`) | Pattern for all LLM touchpoints: LLM constrained to bounded decisions, never originating trades |
| `lib/indicators.ts` | Shared — SMA/EMA/RSI/ATR/ADX/Donchian/Clenow used by all bots |
| `lib/manage-positions.ts`, `lib/weekly.ts` | Position management — climax/trend-break exits are literally Livermore's "abnormal reaction" exits |
| `scripts/backtest-nasdaq.ts` | Backtest engine v1 — survivorship-adjusted, T+1 execution. Needs: shorts, pairs, ETFs, event windows, walk-forward harness |
| `scripts/regime-bear-tests.ts` | Already stress-tests 2008/2020/2018 — becomes a mandatory promotion gate |
| `scripts/factor-analysis.ts` | Seed of ARB bot's factor-residual model |
| `lib/leaps-agent.ts` | Options execution path — later reused by MACRO for convex expressions |
| Vercel cron + dashboard | Fleet scheduler + fleet dashboard skeleton |

**Biggest gaps:** persistent journal (nothing is logged to a DB today), fundamentals
data (no C-A-N of CAN SLIM), catalyst calendar, macro data (FRED), pyramiding
(explicitly unsupported today), short/pair support in backtester, and the entire
improvement engine.

---

## 2. Target architecture

```
┌──────────────────────────────────────────────────────────────┐
│ L3 META      allocator · correlation monitor · fleet reports │
├──────────────────────────────────────────────────────────────┤
│ L2 BOTS      breakout/ · arb/ · catalyst/ · macro/           │
│              (common Bot interface, virtual sub-books)       │
├──────────────────────────────────────────────────────────────┤
│ L1 PLATFORM  data plane · journal · backtest v2 · regime     │
│              service · risk kernel (Kelly) · execution router│
└──────────────────────────────────────────────────────────────┘
```

### The Bot interface

Every bot implements the same contract so the meta-layer treats them identically:

```ts
// lib/fleet/bot.ts
export interface Bot {
  id: 'breakout' | 'arb' | 'catalyst' | 'macro';
  favorableRegimes: RegimeTag[];          // Tier C gate
  scan(ctx: MarketContext): Promise<FleetSignal[]>;
  size(signal: FleetSignal, book: SubBook): SizedOrder | null;
  manage(book: SubBook, ctx: MarketContext): Promise<ManagementAction[]>;
  paramSpace(): ParamSpec[];              // Tier A: what walk-forward may tune, with bounds
}
```

`FleetSignal` extends the existing `Signal` type with `botId`, a `features`
snapshot (every input the bot saw, for the journal), and a `thesis` string.

### Virtual sub-books (one Alpaca account, four books)

Like a real multi-strat: one prime broker, many pods. Orders carry
`client_order_id = "<botId>:<signalId>"`; a ledger (`lib/fleet/ledger.ts`) tracks
per-bot cash, positions, and P&L in Postgres so bots cannot spend each other's
capital. The existing kill switches move up to fleet level; each bot also gets its
own drawdown circuit breaker (pause at −15% from high-water mark, human review to
restart).

### Scheduling (Vercel cron, staggered)

| Route | Cadence | Bot |
|---|---|---|
| `/api/cron/tick` | 30 min (exists) | breakout |
| `/api/cron/arb` | hourly | arb |
| `/api/cron/catalyst` | daily 8:00 ET + 17:00 ET post-print pass | catalyst |
| `/api/cron/macro` | daily 8:30 ET | macro |
| `/api/cron/learn` | nightly + weekly jobs | improvement engine |

---

## 3. Bot 1 — BREAKOUT (Livermore × O'Neil)

The two philosophies merge cleanly — O'Neil explicitly built on Livermore. Livermore
contributes *behavioral* rules (probing, pyramiding, sitting, abnormal-reaction
exits); O'Neil contributes *selection* rules (CAN SLIM fundamentals, RS leadership,
market gate). Horizon: weeks–months, 4–8 concentrated longs. Fits $10k cleanly.

**Already built:** Profile A/B scanner, 2-gate market filter (O'Neil's "M"),
Clenow ranking, climax + trend-break exits, winner protection, bracket orders.

**To build:**

1. **Fundamentals overlay (C, A, N of CAN SLIM)** — quarterly EPS growth ≥ 25%,
   accelerating; annual growth ≥ 25%; recent catalyst/new-high flag. Source: FMP
   starter tier. New `lib/fundamentals.ts` with point-in-time caching (never use a
   later-revised number in backtests).
2. **True RS rank** — percentile-rank every universe symbol's 12-month
   weighted return (IBD-style, recent quarter double-weighted); require RS ≥ 85.
   Replaces raw momentum comparison with cross-sectional rank.
3. **Livermore probing + pyramiding** — replace all-at-once entry: enter 1/2 at
   pivot, add 1/4 at +2.5%, add 1/4 at +5%, *only while above pivot*. Never add to a
   loser (inverts today's "no pyramiding" rule into "pyramid winners only").
   Stop for the whole line stays at pivot −7-8% (O'Neil) — tightened from today's 20%.
4. **Follow-through-day detection** — O'Neil's market-bottom timing signal (day 4-10
   of rally attempt, +1.5%+ on higher volume) added as a third regime gate so the bot
   re-enters early after corrections instead of waiting for the 200-EMA.
5. **Industry-group strength** — require the stock's group (exists:
   `lib/industries.ts`) in the top 40% by group RS. Livermore: trade leaders in
   leading groups.

**Tier A tunables:** RS floor, EPS-growth floor, pyramid spacing, stop distance,
volume-surge multiple, group-strength floor.
**Learning targets:** breakout failure post-mortems ("volume was only 32% above
average on breakout day — raise the S threshold?").

---

## 4. Bot 2 — ARB (Thorp) + fleet money management

Thorp contributes two things, and the second matters more than his bot:

### 4a. The strategy pod (stat-arb, honest version)

At $10k this is a *research pod*, not an edge machine — 4–6 pair positions max,
low expected absolute P&L. Build it anyway: it produces the fleet's only
market-neutral return stream (valuable for the correlation monitor) and the
infrastructure (shorts, pairs, factor residuals) unlocks future strategies.

- **Universe:** liquid sector ETF pairs (XLE/OIH, SMH/QQQ…) + top-200 S&P names
  within sector clusters.
- **Signal:** rolling 252d Engle-Granger cointegration; tradeable only if spread
  half-life ≤ 20 days; enter at z = ±2, exit at 0, hard stop at ±3.5, time stop at
  2× half-life. Beta-neutral leg sizing.
- **Modern upgrade (phase 2):** factor-residual reversal — regress daily returns on
  market/size/value/momentum/quality (extends `scripts/factor-analysis.ts`), trade
  mean reversion of the *residual*, which is far less crowded than price pairs.
- **Tier A tunables:** entry/exit/stop z-scores, half-life ceiling, lookback.
- **Learning:** weekly universe re-fit; every pair carries a rolling out-of-sample
  Sharpe; pairs below 0.5 get benched automatically (journal keeps counterfactuals:
  "what would benched pairs have done?").

### 4b. Thorp as fleet treasurer — Kelly money management (`lib/fleet/kelly.ts`)

Kelly sizing applied at three levels, always as **quarter-Kelly**, and always as a
*cap-reducer* below existing hard limits (Kelly may shrink a position, never exceed
the fixed-fractional/25%-notional caps):

1. **Per trade (within bot):** f* ≈ μ/σ² of the bot's edge estimate for that setup
   class, from its journal, shrunk toward zero (small-sample prior — an untested
   setup gets near-zero Kelly weight until n ≥ 30 trades).
2. **Per bot (meta-allocator input):** each bot's capital share scales with its
   rolling out-of-sample edge estimate.
3. **Fleet:** total gross exposure capped so that estimated fleet volatility stays
   within target (10–12% annualized to start).

This is the piece that compounds: bots earn capital by demonstrated edge, measured
out-of-sample, sized by the same math Thorp used at PNP.

---

## 5. Bot 3 — CATALYST (Cohen)

Cohen's real style is intraday and information-driven; at $10k (PDT rule live) we run
the honest adaptation: **swing catalyst trading, 3–15 day horizon**, entries ≥ 1 day
apart, never intraday round-trips.

- **Calendar:** pull 7-day forward earnings/guidance/FDA calendar (Finnhub free tier
  to start, Benzinga later).
- **Mosaic score** (composite, weights re-fit weekly):
  - Analyst estimate-revision trend last 30d (FMP)
  - Options implied move vs. that stock's trailing 8-quarter realized earnings moves
    (Alpaca options data) — rich/cheap event vol
  - Short-interest delta over 30d (FINRA bi-monthly)
  - Insider Form-4 net buying prior 90d (SEC/FMP)
  - (later) social/web sentiment for consumer names
- **Trade:** enter long/short 3–7 days pre-catalyst when |mosaic| clears threshold;
  exit 1–3 days post. Position risk 1% of sub-book per event, max 4 concurrent.
- **Post-print LLM read:** extend the `llm-overlay.ts` pattern — LLM reads the press
  release/transcript and must choose from `{exit-now, hold-to-plan, trim-half}`,
  bounded exactly like the veto (it can never originate or size). Decision + reasoning
  journaled.
- **Learning (cleanest of all four):** every catalyst has ground truth within days.
  Log predicted direction/magnitude vs. actual per mosaic component; weekly logistic
  re-fit of component weights; components whose coefficient goes ~0 get dropped.
  This bot generates labeled data fastest, so expect the most Tier A/B churn here.

---

## 6. Bot 4 — MACRO (Druckenmiller)

Concentrated directional macro expressed through ETFs. Horizon weeks–6 months.
1–3 positions, conviction-scaled: 20% of sub-book at low conviction → 60% at high.
"It takes courage to be a pig" — implemented as sizing ∝ conviction score, with the
fleet risk kernel as the adult in the room.

- **Macro state vector** (all free: FRED + ETF prices):
  yield-curve slope (10y−2y), real-rate trend, Fed stance (FFR trend + dot-plot-free
  proxy: 3m changes), credit spreads (HY OAS), USD trend, oil trend, ISM PMI, VIX
  regime, global breadth (EEM/EWJ vs SPY).
- **Regime classifier:** start rule-based (explicit, auditable playbook table:
  "late-cycle + easing + falling PMI → long duration + gold"), upgrade to
  GMM/HMM once the journal has depth. This *replaces and generalizes* `isRiskOn` —
  the whole fleet consumes its output as Tier C gates.
- **Conviction score:** count of agreeing indicators + distance of state vector from
  its neutral centroid. Maps to size buckets.
- **Expression set:** SPY/QQQ/IWM, TLT/IEF, GLD/SLV, UUP, DBC/USO, EEM/EWJ, and
  (phase 2) defined-risk options via the existing LEAPS machinery for asymmetric
  expressions.
- **LLM reading loop:** monthly, LLM summarizes FOMC statement changes vs. prior
  (public text) into structured deltas ("hawkish→neutral on inflation language");
  affects *priors only*, human-gated, journaled. This automates the reading
  Druckenmiller actually does.
- **Thesis journal:** every position carries a falsifiable thesis ("long TLT: Fed
  pivot within 2 FOMC meetings") with explicit invalidation criteria; scored at
  60/120 days. Quarterly LLM post-mortem across theses: "6 of 8 'Fed pivot' theses
  lost — the real-rate trigger was too loose; propose tightening."

---

## 7. The self-improvement engine (the core)

### 7a. Journal-first (the substrate everything learns from)

Neon Postgres (already the README's "next step #1"). Log **every decision, including
the ones not taken** — vetoed signals, benched pairs, regime-blocked entries — with
enough features to replay them. Counterfactuals are half the training data.

Tables (full schema in Appendix A): `signals` (features JSON, thesis, taken/rejected
+ why), `orders`/`fills` (slippage vs. model), `positions` (MFE/MAE, exit reason),
`theses` (macro), `regime_snapshots`, `proposals` (Tier B audit trail),
`allocations` (meta-layer history).

### 7b. Tier A — automated parameter tuning (weekly)

- Nightly data refresh; weekly walk-forward re-fit per bot over its `paramSpace()`
  — bounded search **around current values only** (±20% per dimension), anchored
  walk-forward: fit 3y → validate 6m → roll.
- **Acceptance gates (all required):** OOS Sharpe improves ≥ 0.2 · OOS max-DD not
  worse by > 10% · parameter drift ≤ 10% per week · n ≥ 30 trades in validation.
- Accepted changes are **not deployed directly** — they open a PR
  (`scripts/learn/propose-params.ts` writes the config diff + attaches walk-forward
  report). Merge = deploy. Git history *is* the parameter audit trail.

### 7c. Tier B — LLM-proposed rule changes (weekly, human-gated)

The "teach it to improve" loop, concretely:

1. Weekly job assembles the bot's last 20–50 closed trades **plus counterfactuals**
   (vetoed signals and what happened after) with features, thesis, outcome, MFE/MAE.
2. LLM prompt: *"Identify the two costliest recurring failure modes. Propose ONE
   bounded rule change that would have filtered the losers without losing more than
   20% of the winners. State the exact rule, the trades it flips, and a falsifiable
   success criterion."*
3. Proposal lands in the `proposals` table **and as a draft PR** containing: the rule
   as code, backtest-with-rule vs. without, list of historical trades flipped.
4. **You review** (~30 min/week): approve → the rule runs as a **challenger** — a
   shadow bot on paper with small virtual capital for 2–4 weeks; reject → logged
   with your reason (rejections teach the proposer: they're injected into future
   proposal prompts as "previously rejected because…").
5. Challenger beats champion out-of-sample → promote. Champion/challenger switch is
   itself a journaled allocation event.

### 7d. Tier C — regime-conditional deployment (real-time)

Each bot declares `favorableRegimes` (priors: BREAKOUT wants confirmed uptrends;
ARB wants mid/high-vol range; CATALYST is regime-agnostic but halves size when VIX >
30; MACRO is all-weather). The regime service gates scanning per bot. Over time the
journal replaces priors with *measured* regime-conditional Sharpe per bot — deploy
where each bot demonstrably works. This is what saves the fleet in a 2022.

### 7e. Anti-overfitting constitution (non-negotiable)

1. **Holdout vault:** 2008, 2015–16, 2020-Q1, 2022 never enter any fit; quarterly
   final validation only (extends `regime-bear-tests.ts`).
2. Walk-forward everything; in-sample results are never reported anywhere.
3. Degrees-of-freedom budget: each bot may change ≤ 1 rule + bounded params per week.
4. n ≥ 30 before any inference; Kelly shrinkage handles the rest.
5. Backtest Sharpe > 3 on a daily-bar strategy = assume bug, not edge.
6. Slippage floor: 5 bps equities, 15 bps options — in every backtest and every
   live-vs-model reconciliation (journal catches divergence).
7. 30-day paper burn-in for every new bot and every promoted challenger.

### 7f. Improvement metrics (per bot, on the dashboard)

Rolling 90d OOS Sharpe · hit rate · avg win/avg loss · MFE capture (what fraction of
max favorable excursion we keep — the single best exit-quality metric) · slippage vs.
model · regime-conditional returns · proposal acceptance rate & post-promotion delta
(is Tier B actually adding value? measure it like everything else).

---

## 8. Meta-allocator (L3)

- Start: $10k × 4, equal.
- Monthly rebalance: **risk-parity base** (equal vol contribution) **+ bandit tilt**
  toward rolling 90d OOS Sharpe, quarter-Kelly-scaled; hard caps 10%–40% per bot.
- **Correlation monitor:** pairwise 60d return correlation of the four sub-books;
  any pair > 0.7 → alert; fleet average > 0.5 → freeze new entries pending review
  (guards against "four bots that are all secretly long QQQ").
- Nightly LLM-written fleet narrative (extends the dashboard): what each bot did,
  why, current theses, pending proposals — so the weekly human review takes 30 min,
  not 3 hours.

---

## 9. Data & infrastructure additions

| Need | Source | Cost |
|---|---|---|
| Journal DB | Neon Postgres (Vercel-native) | free tier |
| Fundamentals (C-A-N), estimates, insiders | FMP starter | ~$25/mo |
| Earnings/catalyst calendar | Finnhub free → Benzinga later | $0 → later |
| Macro series | FRED API | free |
| Options chains/IV | Alpaca options data | included |
| Short interest | FINRA published data | free |
| Backtest v2 | extend `backtest-nasdaq.ts`: shorts, pairs, ETFs, event windows, walk-forward harness | build |

Total new spend to start: **~$25/month.**

---

## 10. Build sequence & promotion gates

Each phase has a definition-of-done; a bot only advances Paper → Live by passing all
gates: walk-forward OOS Sharpe ≥ 0.8 (≥ 0.5 for ARB, it's market-neutral) · bear-year
holdout doesn't blow the 15% circuit breaker · 30 paper days, zero critical bugs ·
live slippage within 2× model · human sign-off.

| Phase | Weeks | Deliverable |
|---|---|---|
| 0 — Platform | 1–2 | Journal schema + ledger + Bot interface; refactor existing agent into `bots/breakout/` unchanged (regression: backtest reproduces 46/46 trades); Neon wired; every signal/order journaled |
| 1 — BREAKOUT upgrade | 3–4 | Fundamentals overlay, RS rank, pyramiding, follow-through day; walk-forward; paper with $10k sub-book |
| 2 — Kelly + ARB | 5–6 | `kelly.ts` (fleet-wide), backtest v2 shorts+pairs, cointegration engine, ARB on paper; meta-allocator v0 (2 bots) |
| 3 — MACRO | 7–8 | FRED pipeline, macro state vector, playbook table, regime service v2 replaces `isRiskOn` fleet-wide; MACRO on paper |
| 4 — CATALYST | 9–10 | Calendar + mosaic score + post-print LLM read; event-window backtests; paper |
| 5 — Improvement engine | 11–12 | Tier A walk-forward job + PR generator; Tier B proposal pipeline; Tier C gates live; dashboard: 4 sub-books + improvement metrics |
| 6 — Burn-in & go-live | 13+ | 30-day full-fleet paper; promote qualifying bots to live small; monthly meta-rebalance on |

Honest expectation-setting: BREAKOUT and MACRO are most likely to pass gates first.
ARB at $10k proves infrastructure, not P&L. CATALYST needs a full earnings season
(~13 weeks) of paper data before its mosaic weights mean anything.

---

## 11. Operating cadence (steady state)

- **Nightly (automated):** journal ingest, metrics refresh, LLM fleet narrative.
- **Weekly (30 min human):** read narrative → review Tier A param PRs + Tier B
  proposals → approve/reject with one-line reasons (logged).
- **Monthly (automated + 15 min):** meta-reallocation, correlation report.
- **Quarterly (1–2 h):** holdout-vault validation, thesis post-mortems (MACRO),
  strategy review — kill, keep, or expand each bot; consider next bot (a fifth,
  Simons-style pure-quant pod, only once the journal is mature).

---

## 12. Risk register

| Risk | Mitigation |
|---|---|
| Overfitting via the improvement loop itself | §7e constitution: holdout vault, DoF budget, OOS-only acceptance, challenger burn-in |
| All four bots secretly long the same beta | Correlation monitor + freeze rule (§8) |
| PDT at $10k live (CATALYST) | Swing horizon ≥ 1 day by design; entries spaced |
| Paper fills flatter than live | Slippage floors in backtests; live-vs-model reconciliation from `fills` |
| LLM originates a bad trade | Structural: LLM is veto/choose-from-menu/propose-only in every touchpoint, everywhere |
| Data vendor lookahead (revised fundamentals) | Point-in-time caching in `lib/fundamentals.ts`; store as-first-reported |
| Regime engine wrong at the worst time | Per-bot circuit breakers + fleet kill switch are regime-independent hard stops |
| Motivation decay ("it's been 8 weeks of paper") | Gates are contractual; the dashboard shows gate progress so paper time feels like progress |

---

## Appendix A — journal schema (Postgres)

```sql
create table signals (
  id uuid primary key default gen_random_uuid(),
  bot_id text not null,
  at timestamptz not null,
  symbol text not null,
  side text not null,
  features jsonb not null,          -- full input snapshot at decision time
  thesis text not null,
  confidence real,
  regime jsonb not null,            -- regime vector at signal time
  action text not null,             -- taken | rejected_llm | rejected_risk | rejected_regime | rejected_rank
  reject_reason text
);

create table orders (
  id uuid primary key,
  signal_id uuid references signals(id),
  bot_id text not null,
  client_order_id text unique not null,   -- "<botId>:<signalId>"
  submitted_at timestamptz, type text, qty numeric, limit_price numeric
);

create table fills (
  order_id uuid references orders(id),
  filled_at timestamptz, qty numeric, price numeric,
  slippage_bps real                  -- vs decision price: live-vs-model truth
);

create table positions (
  id uuid primary key,
  bot_id text not null, symbol text not null,
  opened_at timestamptz, closed_at timestamptz,
  entry_px numeric, exit_px numeric, qty numeric,
  max_favorable_bps real, max_adverse_bps real,   -- MFE / MAE
  exit_reason text,                  -- stop | target | time | climax | trend_break | regime | llm_post_print
  pnl_usd numeric, r_multiple real
);

create table theses (               -- MACRO falsifiable theses
  id uuid primary key, bot_id text, opened_at timestamptz,
  statement text, invalidation text,
  score_60d real, score_120d real, post_mortem text
);

create table regime_snapshots (
  at timestamptz primary key, vector jsonb, label text
);

create table proposals (            -- Tier A/B audit trail
  id uuid primary key, bot_id text, tier text, created_at timestamptz,
  description text, diff text, backtest_report jsonb,
  status text,                       -- draft | approved | rejected | shadow | promoted | retired
  human_reason text,                 -- rejections teach the proposer
  shadow_start timestamptz, shadow_result jsonb
);

create table allocations (
  at timestamptz, bot_id text, capital_usd numeric, reason text
);
```

## Appendix B — new env vars

```
DATABASE_URL=            # Neon
FMP_API_KEY=             # fundamentals, estimates, insiders
FINNHUB_API_KEY=         # catalyst calendar
FRED_API_KEY=            # macro series
FLEET_TOTAL_USD=40000
FLEET_VOL_TARGET_PCT=12
BOT_DD_CIRCUIT_PCT=15    # per-bot pause threshold
KELLY_FRACTION=0.25
```
