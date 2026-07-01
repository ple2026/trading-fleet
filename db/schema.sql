-- Fleet journal — the substrate the improvement engine learns from.
-- Every decision is logged, including the ones NOT taken (vetoed signals,
-- benched pairs, regime-blocked entries). Counterfactuals are half the data.

create extension if not exists "pgcrypto";

create table if not exists signals (
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
create index if not exists signals_bot_at on signals (bot_id, at desc);

create table if not exists orders (
  id uuid primary key default gen_random_uuid(),
  signal_id uuid references signals(id),
  bot_id text not null,
  client_order_id text unique not null,   -- "<botId>:<signalId>"
  submitted_at timestamptz,
  type text,
  qty numeric,
  limit_price numeric
);

create table if not exists fills (
  id uuid primary key default gen_random_uuid(),
  order_id uuid references orders(id),
  filled_at timestamptz,
  qty numeric,
  price numeric,
  slippage_bps real                  -- vs decision price: live-vs-model truth
);

create table if not exists positions (
  id uuid primary key default gen_random_uuid(),
  bot_id text not null,
  symbol text not null,
  opened_at timestamptz,
  closed_at timestamptz,
  entry_px numeric,
  exit_px numeric,
  qty numeric,
  max_favorable_bps real,            -- MFE
  max_adverse_bps real,              -- MAE
  exit_reason text,                  -- stop | target | time | climax | trend_break | regime | llm_post_print
  pnl_usd numeric,
  r_multiple real
);

create table if not exists theses (   -- MACRO falsifiable theses
  id uuid primary key default gen_random_uuid(),
  bot_id text not null,
  opened_at timestamptz,
  statement text,
  invalidation text,
  score_60d real,
  score_120d real,
  post_mortem text
);

create table if not exists regime_snapshots (
  at timestamptz primary key,
  vector jsonb,
  label text
);

create table if not exists proposals (   -- Tier A/B audit trail
  id uuid primary key default gen_random_uuid(),
  bot_id text not null,
  tier text not null,                -- A | B
  created_at timestamptz not null,
  description text,
  diff text,
  backtest_report jsonb,
  status text not null,              -- draft | approved | rejected | shadow | promoted | retired
  human_reason text,                 -- rejections teach the proposer
  shadow_start timestamptz,
  shadow_result jsonb
);

create table if not exists allocations (
  id uuid primary key default gen_random_uuid(),
  at timestamptz not null,
  bot_id text not null,
  capital_usd numeric,
  reason text
);
