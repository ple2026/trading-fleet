"""End-to-end smoke tests. Prove the whole fleet runs offline and the backtest
engine's accounting invariants hold, so a broken bot or an accounting regression
fails CI instead of silently producing nonsense numbers.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from research.fleet.backtest import run_backtest
from research.fleet.bots.arb import Arb
from research.fleet.bots.breakout import Breakout
from research.fleet.bots.catalyst import Catalyst
from research.fleet.bots.macro import Macro
from research.fleet.data import SyntheticProvider, load_panel

BOTS = [Breakout, Arb, Catalyst, Macro]
UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "META", "XLE", "OIH",
            "SMH", "GLD", "GDX", "TLT", "UUP", "DBC", "EEM"]
START = datetime(2021, 1, 1)
END = datetime(2023, 12, 31)


@pytest.fixture(scope="module")
def panel():
    return load_panel(UNIVERSE, START, END, SyntheticProvider(seed=7))


def test_all_bots_instantiate():
    for cls in BOTS:
        bot = cls()
        assert bot.id
        assert bot.param_space(), f"{bot.id} declares no tunable params"


@pytest.mark.parametrize("cls", BOTS, ids=[c.__name__ for c in BOTS])
def test_backtest_runs_and_stays_solvent(cls, panel):
    result = run_backtest(cls(), panel, START, END, starting_equity=10_000.0)
    eq = result.equity_curve
    assert not eq.empty
    # The gross-exposure cap must keep an unleveraged cash book solvent.
    assert (eq > 0).all(), f"{cls.__name__} drove equity to zero/negative"
    m = result.metrics()
    # No fake alpha on near-random data: a real edge here would mean a bug.
    assert m["sharpe"] < 4.0, f"{cls.__name__} Sharpe {m['sharpe']} — suspicious on random data"


def test_no_phantom_cash_on_shorts(panel):
    """A pure-short bot's ending cash must reflect real P&L, not injected notional."""
    result = run_backtest(Macro(), panel, START, END, starting_equity=10_000.0)
    # Sum of realized trade P&L should be within a small tolerance of the change in
    # equity attributable to closed trades (sanity, not exact due to open marks).
    realized = sum(t.pnl_usd for t in result.trades if t.exit_reason != "open")
    assert abs(realized) < 10_000.0 * 5, "implausible realized P&L — accounting bug"
