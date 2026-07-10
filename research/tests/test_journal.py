"""Journal + reproducibility tests.

The improvement engine only works if two things hold: the backtest records every
decision (taken AND rejected — the counterfactuals), and the synthetic data plane
is byte-for-byte reproducible so walk-forward / Tier-A comparisons mean something.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from research.fleet.backtest import run_backtest
from research.fleet.bots.breakout import Breakout
from research.fleet.bots.macro import Macro
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.journal import Journal

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "META", "XLE", "OIH",
            "SMH", "GLD", "GDX", "TLT", "UUP", "DBC", "EEM"]
START = datetime(2021, 1, 1)
END = datetime(2023, 12, 31)


@pytest.fixture(scope="module")
def panel():
    return load_panel(UNIVERSE, START, END, SyntheticProvider(seed=7))


def test_synthetic_provider_is_reproducible():
    """Same seed -> identical bars. Guards the hash-salt reproducibility bug."""
    a = SyntheticProvider(seed=42).daily_bars("AAPL", START, END)
    b = SyntheticProvider(seed=42).daily_bars("AAPL", START, END)
    assert a.equals(b)


def test_symbol_offset_is_stable_across_processes():
    """A content hash — not hash() — so the offset is fixed forever, every run."""
    # If this constant ever changes, per-symbol seeding drifted and every backtest
    # silently sits on different data. blake2b('AAPL', digest_size=4) is fixed.
    assert SyntheticProvider._symbol_offset("AAPL") == \
        int.from_bytes(__import__("hashlib").blake2b(b"AAPL", digest_size=4).digest(), "big") % 10_000
    # Distinct symbols get distinct offsets (no accidental collisions in universe).
    offsets = {s: SyntheticProvider._symbol_offset(s) for s in UNIVERSE}
    assert len(set(offsets.values())) == len(UNIVERSE)


def test_journal_records_taken_and_counterfactuals(panel):
    """Every taken trade is journaled, and rejected signals are captured too."""
    j = Journal(keep_in_memory=True)
    result = run_backtest(Macro(), panel, START, END, journal=j)

    signals = j.records.get("signals", [])
    closes = j.records.get("positions", [])
    assert signals, "no signals journaled"

    taken = [s for s in signals if s["action"] == "taken"]
    rejected = [s for s in signals if s["action"].startswith("rejected_")]
    assert rejected, "no counterfactuals captured — rejected signals missing"

    # Taken signals must equal positions opened = closed + still-open at end.
    still_open = len(taken) - len(closes)
    assert still_open >= 0
    assert len(closes) == len(result.trades)

    # Every taken signal carries its regime (Tier C needs it) and a disposition.
    for s in taken:
        assert s["regime"] is not None and "label" in s["regime"]
    # Closed positions carry MFE/MAE, exit reason, and entry regime.
    for c in closes:
        assert "max_favorable_bps" in c and "max_adverse_bps" in c
        assert c["exit_reason"] and c["entry_regime"]


def test_regime_blocked_signals_are_counterfactuals(panel):
    """A regime-gated bot scans blocked days only to journal them, never trades."""
    j = Journal(keep_in_memory=True)
    run_backtest(Breakout(), panel, START, END, journal=j)
    signals = j.records.get("signals", [])
    # On this synthetic SPY the tape is rarely TRENDING_UP, so Breakout is mostly
    # regime-blocked — those signals must be logged as rejected_regime, not traded.
    blocked = [s for s in signals if s["action"] == "rejected_regime"]
    assert blocked, "regime-blocked counterfactuals not captured"
    for s in blocked:
        assert "not in favorable set" in (s["reject_reason"] or "")


def test_in_memory_journal_writes_no_files(tmp_path):
    """A transient in-memory journal must not touch disk (no cross-run dupes)."""
    j = Journal(jsonl_dir=tmp_path / "journal", keep_in_memory=True)
    assert j.persist_jsonl is False
    assert not (tmp_path / "journal").exists()
