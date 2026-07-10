"""Tier B / Tier C improvement-engine tests (§7c / §7d).

Tier C must read regime-conditional edge off journaled closed trades and recommend
where each bot works. Tier B must find a planted separating feature, respect prior
human rejections, and stay silent when the evidence is thin — never fit noise.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from research.fleet.backtest import run_backtest
from research.fleet.bots.macro import Macro
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.improve import (
    Proposal,
    propose_rule_change,
    recommend_regimes,
    regime_conditional_performance,
)
from research.fleet.journal import Journal


# ------------------------------------------------------------------ Tier C

def _pos(bot, regime, pnl, r):
    return {"bot_id": bot, "entry_regime": regime, "pnl_usd": pnl, "r_multiple": r}


def test_regime_conditional_performance_groups_and_scores():
    positions = (
        [_pos("macro", "trending-up", 100.0, 1.5) for _ in range(15)]
        + [_pos("macro", "trending-up", -40.0, -0.6) for _ in range(5)]
        + [_pos("macro", "risk-off", -50.0, -1.0) for _ in range(20)]
    )
    perf = {(p.bot_id, p.regime): p for p in regime_conditional_performance(positions)}
    up = perf[("macro", "trending-up")]
    off = perf[("macro", "risk-off")]
    assert up.n == 20 and up.hit_rate == 0.75 and up.avg_r > 0
    assert off.n == 20 and off.avg_r < 0
    assert up.total_pnl > 0 and off.total_pnl < 0


def test_recommend_regimes_add_and_drop_vs_priors():
    positions = (
        [_pos("macro", "trending-up", 100.0, 1.2) for _ in range(25)]
        + [_pos("macro", "risk-off", -30.0, -0.7) for _ in range(25)]
    )
    rec = recommend_regimes(
        positions, min_trades=20, favorable_priors={"macro": {"risk-off"}}
    )["macro"]
    assert rec["recommended"] == ["trending-up"]
    # Journal says: start trading trending-up, stop trading risk-off.
    assert rec["add"] == ["trending-up"]
    assert rec["drop"] == ["risk-off"]


def test_tier_c_runs_on_real_journal():
    U = ["SPY", "QQQ", "IWM", "TLT", "GLD", "DBC", "UUP", "EEM"]
    panel = load_panel(U, datetime(2020, 1, 1), datetime(2023, 12, 31),
                       SyntheticProvider(seed=9))
    j = Journal(keep_in_memory=True)
    run_backtest(Macro(), panel, datetime(2020, 1, 1), datetime(2023, 12, 31), journal=j)
    perf = regime_conditional_performance(j.records.get("positions", []))
    assert perf and all(p.bot_id == "macro" and p.n > 0 for p in perf)


# ------------------------------------------------------------------ Tier B

def _trades_with_separating_feature():
    """20 winners with high vol_ratio, 20 losers with low vol_ratio, plus a pure
    noise feature that must NOT be chosen."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(20):
        rows.append({
            "bot_id": "breakout", "pnl_usd": 100.0, "r_multiple": 2.0,
            "entry_features": {"vol_ratio": float(rng.uniform(1.8, 2.2)),
                               "noise": float(rng.uniform(0, 1))},
        })
    for _ in range(20):
        rows.append({
            "bot_id": "breakout", "pnl_usd": -50.0, "r_multiple": -1.0,
            "entry_features": {"vol_ratio": float(rng.uniform(0.9, 1.1)),
                               "noise": float(rng.uniform(0, 1))},
        })
    return rows


def test_proposer_finds_the_separating_feature():
    prop = propose_rule_change("breakout", _trades_with_separating_feature())
    assert isinstance(prop, Proposal)
    assert prop.backtest_report["feature"] == "vol_ratio"
    assert prop.backtest_report["op"] == ">="
    # All 20 losers sit below the winners' low tail -> most/all should be filtered.
    assert prop.backtest_report["losers_flipped"] >= 15
    # At most 20% of winners sacrificed (the §7c constraint).
    assert prop.backtest_report["winners_lost"] <= 4
    assert prop.tier == "B" and prop.status == "draft"
    assert prop.diff and prop.success_criterion


def test_proposer_respects_prior_rejections():
    rows = _trades_with_separating_feature()
    # Having rejected the only separating feature, the proposer must not resurface
    # it — and noise won't separate, so it should stay silent.
    prop = propose_rule_change("breakout", rows, prior_rejections=("vol_ratio",))
    assert prop is None


def test_proposer_silent_when_thin():
    rows = _trades_with_separating_feature()[:10]  # below min_trades
    assert propose_rule_change("breakout", rows) is None


def test_proposer_silent_when_no_separation():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(40):
        rows.append({
            "bot_id": "breakout",
            "pnl_usd": 100.0 if i % 2 == 0 else -50.0,
            "r_multiple": 1.0 if i % 2 == 0 else -1.0,
            "entry_features": {"vol_ratio": float(rng.uniform(1.0, 2.0))},  # random
        })
    assert propose_rule_change("breakout", rows) is None


def test_record_proposal_captured_in_journal():
    j = Journal(keep_in_memory=True)
    prop = propose_rule_change("breakout", _trades_with_separating_feature(),
                               created_at=datetime(2024, 1, 1))
    j.record_proposal(prop.as_row())
    stored = j.records.get("proposals", [])
    assert len(stored) == 1 and stored[0]["bot_id"] == "breakout"
    assert stored[0]["status"] == "draft" and stored[0]["tier"] == "B"
