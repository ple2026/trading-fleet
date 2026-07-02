"""Smoke test for the end-to-end improvement run (scripts/improve.py).

The individual tiers are unit-tested elsewhere; this guards the wiring — that the
whole flywheel (journal -> Tier C -> Tier B -> allocation) runs without error and
emits each section. Trimmed to two bots / a small universe to stay fast.
"""

from __future__ import annotations

from datetime import datetime

from research.fleet.bots.breakout import Breakout
from research.fleet.bots.macro import Macro
from research.scripts import improve


def test_improvement_run_wires_end_to_end(monkeypatch, capsys):
    monkeypatch.setattr(improve, "BOTS", {"breakout": Breakout, "macro": Macro})
    monkeypatch.setattr(
        improve, "UNIVERSE",
        ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TLT", "GLD", "DBC", "UUP", "EEM"],
    )
    improve.run(
        start=datetime(2020, 1, 1), end=datetime(2023, 12, 31),
        total_usd=20_000.0, do_tune=False, tune_candidates=4,
    )
    out = capsys.readouterr().out
    for section in ("Journal", "Tier C", "Tier B", "MACRO theses", "Meta-allocation"):
        assert section in out
    # Allocation must name both bots and be a plausible fleet report.
    assert "breakout" in out and "macro" in out
    assert "OOS Sharpe" in out
