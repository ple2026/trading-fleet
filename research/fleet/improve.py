"""The improvement engine's Tier B and Tier C (§7c / §7d).

**Tier C — regime-conditional deployment.** Each bot ships with *prior* favorable
regimes. Once the journal has depth, priors get replaced by *measured*
regime-conditional edge: group every closed trade by the regime it was entered in,
score it, and recommend deploying each bot only where it demonstrably works. This
is what saves the fleet in a 2022.

**Tier B — proposed rule changes.** In production an LLM reads the trade journal
plus counterfactuals and proposes ONE bounded rule change with backtest evidence.
Offline (and as the deterministic fallback / test oracle), this module runs the
same *shape* of analysis without an LLM: it finds the single entry feature whose
threshold best separates winners from losers, and proposes adding that filter —
"filter the losers without losing more than 20% of the winners", exactly the §7c
prompt. The proposal is a reviewable draft, never an auto-deploy: it lands in the
`proposals` table for a human to approve, after which it runs as a shadow
challenger before promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

# A trade must clear this many samples before we infer anything from it — the §7e
# "n >= 30 before inference" rule, relaxed slightly for per-regime buckets.
MIN_REGIME_TRADES = 20
MIN_PROPOSAL_TRADES = 20
MAX_WINNER_LOSS = 0.20        # a filter may sacrifice at most 20% of winners
# A filter must drop this much MORE of the losers than of the winners (as a rate)
# before we believe it separates signal from noise. Without this, random features
# produce spurious asymmetric splits and the proposer overfits — the §7e trap.
SEPARATION_MARGIN = 0.35


# ============================================================ Tier C: regimes

@dataclass
class RegimePerformance:
    bot_id: str
    regime: str
    n: int
    hit_rate: float          # fraction of trades with pnl > 0
    avg_r: float             # mean R-multiple
    total_pnl: float
    edge_t: float            # avg_r / stderr(r): a crude t-stat on the edge


def regime_conditional_performance(
    positions: list[dict[str, Any]],
) -> list[RegimePerformance]:
    """Per (bot, entry-regime) performance rollup from journaled closed positions."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for p in positions:
        bot = p.get("bot_id")
        regime = p.get("entry_regime") or "unknown"
        if bot is None:
            continue
        buckets.setdefault((bot, regime), []).append(p)

    out: list[RegimePerformance] = []
    for (bot, regime), rows in sorted(buckets.items()):
        n = len(rows)
        pnls = [float(r.get("pnl_usd", 0.0)) for r in rows]
        rs = [float(r.get("r_multiple", 0.0)) for r in rows]
        wins = sum(1 for x in pnls if x > 0)
        avg_r = float(np.mean(rs)) if rs else 0.0
        sd = float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0
        edge_t = (avg_r / (sd / np.sqrt(n))) if sd > 0 else 0.0
        out.append(RegimePerformance(
            bot_id=bot, regime=regime, n=n,
            hit_rate=round(wins / n, 3) if n else 0.0,
            avg_r=round(avg_r, 3), total_pnl=round(sum(pnls), 2),
            edge_t=round(edge_t, 2),
        ))
    return out


def recommend_regimes(
    positions: list[dict[str, Any]],
    min_trades: int = MIN_REGIME_TRADES,
    favorable_priors: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-bot Tier-C recommendation: which regimes the journal says to trade.

    A regime is *recommended* when the bot has at least `min_trades` there and a
    positive average R. When `favorable_priors` is supplied (each bot's current
    declared regimes), we also surface `add` (measured-good but not currently
    traded) and `drop` (currently traded but measured-bad with enough evidence).
    """
    perf = regime_conditional_performance(positions)
    by_bot: dict[str, list[RegimePerformance]] = {}
    for rp in perf:
        by_bot.setdefault(rp.bot_id, []).append(rp)

    result: dict[str, dict[str, Any]] = {}
    for bot, rows in by_bot.items():
        recommended = {
            rp.regime for rp in rows if rp.n >= min_trades and rp.avg_r > 0
        }
        measured_bad = {
            rp.regime for rp in rows if rp.n >= min_trades and rp.avg_r <= 0
        }
        entry = {
            "recommended": sorted(recommended),
            "per_regime": [rp for rp in sorted(rows, key=lambda r: -r.n)],
        }
        if favorable_priors is not None:
            prior = favorable_priors.get(bot, set())
            entry["add"] = sorted(recommended - prior)
            entry["drop"] = sorted(measured_bad & prior)
        result[bot] = entry
    return result


# ============================================================ Tier B: proposals

@dataclass
class Proposal:
    """A Tier-B rule-change proposal — mirrors the `proposals` journal table."""

    bot_id: str
    tier: str
    created_at: datetime | None
    description: str
    diff: str
    backtest_report: dict[str, Any]
    success_criterion: str
    status: str = "draft"
    human_reason: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "bot_id": self.bot_id, "tier": self.tier, "created_at": self.created_at,
            "description": self.description, "diff": self.diff,
            "backtest_report": self.backtest_report, "status": self.status,
            "human_reason": self.human_reason,
        }


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _numeric_feature_keys(trades: list[dict], min_coverage: float = 0.5) -> list[str]:
    """Feature keys that are numeric and present in at least `min_coverage` of trades."""
    counts: dict[str, int] = {}
    for t in trades:
        for k, v in (t.get("entry_features") or {}).items():
            if _numeric(v):
                counts[k] = counts.get(k, 0) + 1
    n = len(trades)
    return [k for k, c in counts.items() if n and c / n >= min_coverage]


def _vals(trades: list[dict], key: str) -> list[float]:
    return [float(t["entry_features"][key]) for t in trades
            if _numeric((t.get("entry_features") or {}).get(key))]


def _best_filter(
    winners: list[dict], losers: list[dict], keys: list[str],
    max_winner_loss: float = MAX_WINNER_LOSS,
) -> dict[str, Any] | None:
    """Find the one-sided threshold on one feature that drops the most losers while
    sacrificing at most `max_winner_loss` of the winners.

    For each feature we try both directions ("keep if >= t" drops the low tail;
    "keep if <= t" drops the high tail). The threshold is pinned to the winners'
    tail quantile so at most `max_winner_loss` winners are ever dropped; we then
    count how many losers fall on the dropped side. Best = most losers removed.
    """
    n_win = len(winners)
    best: dict[str, Any] | None = None
    for key in keys:
        win_vals = _vals(winners, key)
        los_vals = _vals(losers, key)
        if len(win_vals) < 2 or not los_vals:
            continue
        allowed_win_drop = int(np.floor(max_winner_loss * n_win))
        win_sorted = sorted(win_vals)

        # Direction ">=": drop trades below t. t = winners' lower tail quantile.
        t_ge = win_sorted[min(allowed_win_drop, len(win_sorted) - 1)]
        los_drop_ge = sum(1 for v in los_vals if v < t_ge)
        win_drop_ge = sum(1 for v in win_vals if v < t_ge)

        # Direction "<=": drop trades above t. t = winners' upper tail quantile.
        t_le = win_sorted[max(len(win_sorted) - 1 - allowed_win_drop, 0)]
        los_drop_le = sum(1 for v in los_vals if v > t_le)
        win_drop_le = sum(1 for v in win_vals if v > t_le)

        n_los = len(los_vals)
        for op, thr, los_drop, win_drop in (
            (">=", t_ge, los_drop_ge, win_drop_ge),
            ("<=", t_le, los_drop_le, win_drop_le),
        ):
            # Reject unless the filter removes materially MORE of the losers than of
            # the winners — the guard against fitting a noise feature.
            separation = los_drop / n_los - win_drop / n_win
            if los_drop < 3 or separation < SEPARATION_MARGIN:
                continue
            pnl_saved = sum(
                -float(lp["pnl_usd"]) for lp in losers
                if _numeric((lp.get("entry_features") or {}).get(key))
                and ((op == ">=" and float(lp["entry_features"][key]) < thr)
                     or (op == "<=" and float(lp["entry_features"][key]) > thr))
            )
            cand = {
                "feature": key, "op": op, "threshold": float(thr),
                "losers_flipped": los_drop, "winners_lost": win_drop,
                "n_winners": n_win, "n_losers": len(losers),
                "separation": round(separation, 3), "pnl_saved": round(pnl_saved, 2),
            }
            # Prefer the cleanest separation; break ties by dollars of loss avoided.
            if best is None or cand["separation"] > best["separation"] or (
                cand["separation"] == best["separation"]
                and cand["pnl_saved"] > best["pnl_saved"]
            ):
                best = cand
    return best


def propose_rule_change(
    bot_id: str,
    closed_positions: list[dict[str, Any]],
    *,
    prior_rejections: tuple[str, ...] = (),
    min_trades: int = MIN_PROPOSAL_TRADES,
    max_winner_loss: float = MAX_WINNER_LOSS,
    created_at: datetime | None = None,
) -> Proposal | None:
    """Deterministic Tier-B proposer (LLM-shaped, offline).

    Returns a bounded, falsifiable proposal to add ONE entry filter, or None when
    the journal is too thin or no feature separates winners from losers well
    enough. `prior_rejections` lists feature keys a human already turned down, so
    the proposer never re-suggests them — the rejection-feedback loop from §7c.
    """
    rows = [
        p for p in closed_positions
        if p.get("bot_id") == bot_id and p.get("entry_features")
    ]
    if len(rows) < min_trades:
        return None
    winners = [p for p in rows if float(p.get("pnl_usd", 0.0)) > 0]
    losers = [p for p in rows if float(p.get("pnl_usd", 0.0)) <= 0]
    if not winners or not losers:
        return None

    keys = [
        k for k in _numeric_feature_keys(rows)
        if k not in prior_rejections
    ]
    best = _best_filter(winners, losers, keys, max_winner_loss)
    if best is None:
        return None

    feat, op, thr = best["feature"], best["op"], best["threshold"]
    win_kept = best["n_winners"] - best["winners_lost"]
    description = (
        f"[{bot_id}] Add entry filter: require `{feat} {op} {thr:.4g}`. "
        f"On {len(rows)} closed trades this would have skipped "
        f"{best['losers_flipped']}/{best['n_losers']} losers "
        f"(${best['pnl_saved']:.0f} of losses avoided) while keeping "
        f"{win_kept}/{best['n_winners']} winners."
    )
    diff = (
        f"# {bot_id}.scan() — new gate before emitting a FleetSignal\n"
        f"if not (features['{feat}'] {op} {thr:.4g}):\n"
        f"    continue  # Tier-B filter: {feat} {op} {thr:.4g}"
    )
    success_criterion = (
        f"Over the shadow period, {bot_id}'s hit-rate and average R on trades "
        f"passing `{feat} {op} {thr:.4g}` exceed the champion's, with no more than "
        f"{int(max_winner_loss * 100)}% of would-be winners filtered."
    )
    return Proposal(
        bot_id=bot_id, tier="B", created_at=created_at,
        description=description, diff=diff,
        backtest_report=best, success_criterion=success_criterion,
        status="draft",
    )
