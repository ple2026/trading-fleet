"""Walk-forward harness + Tier-A tuner tests.

These pin the anti-overfitting contract: OOS windows never overlap, the first
training span is never traded, and the tuner accepts a change ONLY when every §7b
gate holds. On near-random synthetic data the honest outcome is almost always
"no change proposed" — a tuner that "improved" here would be fitting noise.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from research.fleet.bots.breakout import Breakout
from research.fleet.bots.macro import Macro
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.walkforward import (
    MAX_DD_WORSENING,
    MIN_SHARPE_GAIN,
    MIN_VALIDATION_TRADES,
    TuningProposal,
    fleet_oos_sharpe,
    make_windows,
    oos_sharpe,
    tune,
    walk_forward,
)

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "META", "XLE", "OIH",
            "SMH", "GLD", "GDX", "TLT", "UUP", "DBC", "EEM"]
START = datetime(2019, 1, 1)
END = datetime(2023, 12, 31)


@pytest.fixture(scope="module")
def panel():
    return load_panel(UNIVERSE, START, END, SyntheticProvider(seed=11))


def test_windows_are_contiguous_and_non_overlapping(panel):
    dates = sorted({d for df in panel.values() for d in df.index})
    windows = make_windows(dates, train_years=3.0, test_months=6)
    assert windows, "no walk-forward windows produced"
    # First test window starts only after the training span (no in-sample trading).
    assert windows[0].test_start > windows[0].train_start
    # Test windows tile forward without overlap.
    for a, b in zip(windows, windows[1:]):
        assert b.test_start > a.test_end
    # Anchored: every train window shares the same anchor.
    assert all(w.train_start == windows[0].train_start for w in windows)


def test_rolling_windows_move_their_anchor(panel):
    dates = sorted({d for df in panel.values() for d in df.index})
    rolling = make_windows(dates, train_years=2.0, test_months=6, anchored=False)
    assert rolling
    # A rolling train window advances its start over time (unlike anchored).
    assert rolling[-1].train_start > rolling[0].train_start


def test_walk_forward_runs_and_is_oos_only(panel):
    res = walk_forward(Macro, panel, START, END, train_years=3.0, test_months=6)
    assert res.windows
    if not res.oos_equity.empty:
        # OOS curve begins at/after the first test window — never in-sample.
        assert res.oos_equity.index.min() >= res.windows[0].test_start
        m = res.metrics()
        # No fake alpha on random data.
        assert m["sharpe"] < 4.0


def test_walk_forward_is_reproducible(panel):
    a = walk_forward(Macro, panel, START, END).metrics()
    b = walk_forward(Macro, panel, START, END).metrics()
    assert a == b


def test_oos_sharpe_matches_walk_forward(panel):
    """The helper is just the walk-forward OOS Sharpe — no in-sample leakage."""
    direct = walk_forward(Macro, panel, START, END).metrics().get("sharpe", 0.0)
    assert oos_sharpe(Macro, panel, START, END) == direct


def test_fleet_oos_sharpe_covers_every_bot(panel):
    bots = {"breakout": Breakout, "macro": Macro}  # skip slow catalyst here
    sharpe = fleet_oos_sharpe(bots, panel, START, END)
    assert set(sharpe) == set(bots)
    assert all(isinstance(v, float) for v in sharpe.values())


def test_accepted_proposal_renders_config_diff_and_row():
    prop = TuningProposal(
        bot_id="breakout", accepted=True,
        reason="OOS Sharpe 0.50 -> 0.80 (+0.30) with 42 OOS trades",
        current_params={"rs_floor": 85.0, "stop_pct": 8.0},
        current_metrics={"sharpe": 0.5},
        proposed_params={"rs_floor": 88.0, "stop_pct": 8.0},  # only rs_floor moves
        proposed_metrics={"sharpe": 0.8}, candidates_evaluated=12,
    )
    diff = prop.config_diff()
    assert "-rs_floor: 85" in diff and "+rs_floor: 88" in diff
    assert "stop_pct" not in diff  # unchanged params are omitted
    row = prop.as_row()
    assert row["tier"] == "A" and row["status"] == "draft" and row["human_reason"] is None


def test_rejected_proposal_row_records_failed_gate():
    prop = TuningProposal(
        bot_id="macro", accepted=False,
        reason="only 4 OOS trades (< 30) — insufficient to infer",
        current_params={"stop_pct": 12.0}, current_metrics={"sharpe": 0.1},
    )
    assert prop.config_diff() == ""            # nothing proposed
    row = prop.as_row()
    assert row["status"] == "rejected"
    assert "insufficient" in row["human_reason"]  # the gate becomes proposer feedback


@pytest.mark.parametrize("bot_class", [Breakout, Macro])
def test_tuner_returns_gated_proposal(panel, bot_class):
    prop = tune(bot_class, panel, START, END, n_candidates=6, seed=3)
    assert prop.bot_id == bot_class().id
    assert prop.candidates_evaluated == 6
    # If accepted, EVERY gate must actually hold on the proposed metrics.
    if prop.accepted:
        assert prop.proposed_params is not None and prop.proposed_metrics is not None
        gain = prop.proposed_metrics["sharpe"] - prop.current_metrics["sharpe"]
        assert gain >= MIN_SHARPE_GAIN
        assert prop.proposed_metrics["trades"] >= MIN_VALIDATION_TRADES
        base_dd = abs(prop.current_metrics.get("max_dd", 0.0))
        if base_dd > 0:
            assert abs(prop.proposed_metrics["max_dd"]) <= base_dd * (1 + MAX_DD_WORSENING)
        # Proposed params stay within each param's declared bounds.
        specs = {s.name: s for s in bot_class().param_space()}
        for name, val in prop.proposed_params.items():
            assert specs[name].lo <= val <= specs[name].hi
    else:
        assert prop.proposed_params is None
        assert prop.reason  # always explains why nothing was proposed
