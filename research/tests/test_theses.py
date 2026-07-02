"""MACRO thesis scoring tests (§6).

Pin the falsification loop: theses are lifted from the journal, scored on the
signed forward return (long scores up-moves, short scores down-moves), and rolled
up by driver so a systematically-losing driver gets flagged for tightening.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from research.fleet.backtest import run_backtest
from research.fleet.bots.macro import Macro
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.journal import Journal
from research.fleet.theses import (
    post_mortem,
    score_theses,
    theses_from_journal,
)


def _ramp(start: str, n: int, slope: float, p0: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = p0 * (1 + slope) ** np.arange(n)
    return pd.DataFrame({"c": close}, index=idx)


def test_theses_from_journal_extracts_taken_macro_signals():
    signals = [
        {"action": "taken", "bot_id": "macro", "symbol": "TLT", "side": "buy",
         "at": datetime(2022, 1, 3), "thesis": "long duration",
         "invalidation": "duration_trend flips", "features": {"driver_key": "duration_trend", "driver_sign": 1}},
        {"action": "rejected_cap", "bot_id": "macro", "symbol": "GLD", "side": "buy",
         "at": datetime(2022, 1, 3), "thesis": "x", "features": {}},         # not taken
        {"action": "taken", "bot_id": "breakout", "symbol": "AAPL", "side": "buy",
         "at": datetime(2022, 1, 3), "thesis": "breakout", "features": {}},  # not macro
    ]
    rows = theses_from_journal(signals)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "TLT" and rows[0]["driver_key"] == "duration_trend"


def test_score_theses_direction_aware():
    panel = {"UP": _ramp("2021-01-01", 200, 0.002), "DN": _ramp("2021-01-01", 200, -0.002)}
    rows = [
        {"bot_id": "macro", "symbol": "UP", "side": "buy", "opened_at": datetime(2021, 1, 4),
         "statement": "long UP", "invalidation": None, "driver_key": "k", "driver_sign": 1},
        {"bot_id": "macro", "symbol": "DN", "side": "sell", "opened_at": datetime(2021, 1, 4),
         "statement": "short DN", "invalidation": None, "driver_key": "k", "driver_sign": -1},
    ]
    scored = {t.symbol: t for t in score_theses(rows, panel)}
    # Long a riser and short a faller both score POSITIVE (thesis played out).
    assert scored["UP"].score_60d > 0 and scored["UP"].score_120d > 0
    assert scored["DN"].score_60d > 0 and scored["DN"].score_120d > 0


def test_score_is_none_when_horizon_exceeds_history():
    panel = {"UP": _ramp("2021-01-01", 40, 0.002)}  # < 60 bars after entry
    rows = [{"bot_id": "macro", "symbol": "UP", "side": "buy",
             "opened_at": datetime(2021, 1, 4), "statement": "x", "invalidation": None,
             "driver_key": "k", "driver_sign": 1}]
    assert score_theses(rows, panel)[0].score_60d is None


def test_post_mortem_flags_a_losing_driver():
    # A long thesis on a FALLING name loses -> the driver should be flagged.
    panel = {"DN": _ramp("2021-01-01", 200, -0.003)}
    rows = [
        {"bot_id": "macro", "symbol": "DN", "side": "buy",
         "opened_at": datetime(2021, 1, 4 + i), "statement": f"long DN {i}",
         "invalidation": None, "driver_key": "bad_driver", "driver_sign": 1}
        for i in range(6)
    ]
    pm = {p.driver_key: p for p in post_mortem(score_theses(rows, panel))}
    assert "bad_driver" in pm
    assert pm["bad_driver"].mean_score_60d < 0
    assert "tightening" in pm["bad_driver"].verdict


def test_thesis_scoring_on_real_macro_journal():
    U = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "UUP", "DBC", "EEM"]
    panel = load_panel(U, datetime(2020, 1, 1), datetime(2023, 12, 31),
                       SyntheticProvider(seed=9))
    j = Journal(keep_in_memory=True)
    run_backtest(Macro(), panel, datetime(2020, 1, 1), datetime(2023, 12, 31), journal=j)
    rows = theses_from_journal(j.records.get("signals", []))
    assert rows, "MACRO journaled no theses"
    scored = score_theses(rows, panel)
    # Most theses should be scorable (entered well before the window end).
    assert any(t.score_60d is not None for t in scored)
    j.record_thesis(scored[0].as_row(post_mortem="unit test"))
    assert j.records.get("theses")
