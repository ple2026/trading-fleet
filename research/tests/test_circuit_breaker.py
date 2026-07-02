"""Per-bot drawdown circuit-breaker tests.

The breaker must (a) be a strict no-op when disabled — backtests unchanged — and
(b) when enabled, pause NEW entries once equity is far enough below its high-water
mark, capping the drawdown and logging the paused signals as counterfactuals.
"""

from __future__ import annotations

from datetime import datetime

from research.fleet.backtest import run_backtest
from research.fleet.bots.catalyst import Catalyst
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.journal import Journal

U = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "META", "XLE", "GLD", "TLT", "EEM"]
START, END = datetime(2020, 1, 1), datetime(2023, 12, 31)


def _panel():
    return load_panel(U, START, END, SyntheticProvider(seed=13))


def test_disabled_breaker_is_a_noop():
    panel = _panel()
    base = run_backtest(Catalyst(), panel, START, END).metrics()
    off = run_backtest(Catalyst(), panel, START, END, circuit_breaker_pct=None).metrics()
    assert base == off                       # identical: default changes nothing


def test_breaker_caps_drawdown_and_logs_counterfactuals():
    panel = _panel()
    j_no = Journal(keep_in_memory=True)
    no = run_backtest(Catalyst(), panel, START, END, journal=j_no).metrics()

    j_cb = Journal(keep_in_memory=True)
    cb = run_backtest(Catalyst(), panel, START, END, journal=j_cb,
                      circuit_breaker_pct=0.10).metrics()

    # A tight 10% breaker cannot make the drawdown WORSE, and here it caps it.
    assert abs(cb["max_dd"]) <= abs(no["max_dd"]) + 1e-9
    assert abs(cb["max_dd"]) < abs(no["max_dd"])          # it actually bit
    # Paused entries are journaled as counterfactuals, not silently dropped.
    circuit = [s for s in j_cb.records["signals"] if s["action"] == "rejected_circuit"]
    assert circuit, "no rejected_circuit signals logged while halted"
    # Fewer trades taken than with no breaker (some entries were paused).
    assert cb["trades"] <= no["trades"]
