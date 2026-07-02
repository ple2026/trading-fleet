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
    make_windows,
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
