"""BREAKOUT exit-side tests — the O'Neil "M" market-exit rule.

When the general market rolls over, BREAKOUT must sell every leader at once (sell
into weakness), not wait for each name's slow trend-break. This was the exit-side
control the survivorship-free drawdown analysis showed it needed.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from research.fleet.backtest import run_backtest
from research.fleet.bot import MarketContext
from research.fleet.bots.breakout import Breakout
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.types import RegimeState, RegimeTag


def _ctx(regime_label: RegimeTag, open_symbols: set[str]) -> MarketContext:
    idx = pd.bdate_range("2021-01-01", periods=60)
    prices = {
        s: pd.DataFrame({"o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0, "v": 1e6}, index=idx)
        for s in open_symbols
    }
    return MarketContext(
        now=datetime(2021, 3, 26), prices=prices,
        regime=RegimeState(at=datetime(2021, 3, 26), label=regime_label),
        equity_usd=10_000.0, open_symbols=open_symbols,
    )


def test_exits_every_leader_when_market_risk_off():
    bot = Breakout()
    actions = bot.manage(_ctx(RegimeTag.RISK_OFF, {"AAA", "BBB", "CCC"}))
    assert {a.symbol for a in actions} == {"AAA", "BBB", "CCC"}
    assert all(a.kind == "exit" for a in actions)
    assert all("market turned down" in a.reason for a in actions)


def test_does_not_blanket_exit_in_healthy_regime():
    bot = Breakout()
    # In a benign RANGE tape a flat, unremarkable holding is left to run — the
    # blanket market-exit must NOT fire (per-name tells govern instead).
    actions = bot.manage(_ctx(RegimeTag.RANGE, {"AAA"}))
    assert actions == []


def test_n_gate_only_tightens_entries():
    """A stricter near-high floor (O'Neil 'N') can only remove breakouts that are
    too far below their 52-week high — never add trades."""
    U = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "META", "XLE", "OIH",
         "SMH", "GLD", "GDX", "TLT", "UUP", "DBC", "EEM"]
    start, end = datetime(2020, 1, 1), datetime(2023, 12, 31)
    panel = load_panel(U, start, end, SyntheticProvider(seed=7))
    loose = run_backtest(Breakout({"near_high_floor": 70.0}), panel, start, end).metrics()
    strict = run_backtest(Breakout({"near_high_floor": 99.0}), panel, start, end).metrics()
    assert strict.get("trades", 0) <= loose.get("trades", 0)
