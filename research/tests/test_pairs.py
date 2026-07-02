"""Two-leg (market-neutral pair) accounting tests for the backtest engine.

ARB carries a hedge leg on every signal; the engine must trade BOTH legs so the
bot is actually market-neutral (not naked long/short leg A). These pin the leg
cash math, that ARB opens beta-neutral hedges, and that a hedged round-trip's cash
change equals its combined P&L (no phantom notional).
"""

from __future__ import annotations

from datetime import datetime

from research.fleet.backtest import (
    Position,
    _close_leg,
    _close_position,
    _leg_mark,
    _open_leg_cash,
    run_backtest,
)
from research.fleet.bots.arb import Arb
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.journal import Journal

U = ["SPY", "QQQ", "IWM", "XLE", "OIH", "SMH", "XLF", "KRE",
     "GLD", "GDX", "XLK", "TLT", "IEF", "UUP", "DBC", "EEM"]
START = datetime(2019, 1, 1)
END = datetime(2024, 12, 31)


def test_leg_cash_helpers_round_trip():
    # Long leg: pay to open, receive to close; net cash change == P&L.
    cash0 = 10_000.0
    after_open = _open_leg_cash(cash0, "buy", 10, 100.0)   # pay 1000
    assert after_open == 9_000.0
    pnl, after_close = _close_leg(after_open, "buy", 100.0, 10, 110.0)  # sell at 110
    assert pnl == 100.0 and after_close == cash0 + pnl
    # Short leg: receive to open, pay to close; profit when price falls.
    after_short = _open_leg_cash(cash0, "sell", 10, 100.0)  # receive 1000
    assert after_short == 11_000.0
    spnl, after_cover = _close_leg(after_short, "sell", 100.0, 10, 90.0)  # cover at 90
    assert spnl == 100.0 and after_cover == cash0 + spnl


def test_leg_mark_sign():
    assert _leg_mark("buy", 10, 50.0) == 500.0     # long adds
    assert _leg_mark("sell", 10, 50.0) == -500.0   # short subtracts


def test_close_position_sums_both_legs():
    pos = Position(
        symbol="XLE", qty=10, entry_px=100.0, entry_at=START, stop=90.0,
        target=130.0, side="buy",
        hedge_symbol="OIH", hedge_qty=8, hedge_entry_px=50.0, hedge_side="sell",
    )
    # Long XLE 100->105 (+50), short OIH 50->48 (+16) => combined +66.
    pnl, _ = _close_position(1_000.0, pos, main_fill=105.0, hedge_fill=48.0)
    assert pnl == 50.0 + 16.0


def test_arb_opens_beta_neutral_hedges_and_is_solvent():
    panel = load_panel(U, START, END, SyntheticProvider(seed=5))
    j = Journal(keep_in_memory=True)
    result = run_backtest(Arb(), panel, START, END, journal=j)
    assert (result.equity_curve > 0).all(), "ARB drove the book insolvent"

    closes = j.records.get("positions", [])
    assert closes, "ARB never closed a pair over six years — check the engine"
    hedged = [c for c in closes if c.get("hedge_symbol")]
    assert hedged, "ARB positions were opened NAKED — the hedge leg is missing"
    for c in hedged:
        # Hedge trades the opposite side of leg A, with real shares.
        assert c["hedge_qty"] > 0
        assert c["hedge_side"] != c["side"]
        assert c["hedge_side"] in ("buy", "sell")


def test_hedged_positions_are_beta_neutral():
    """The hedge notional must equal beta * main notional (the cointegration hedge
    ratio), up to one share of integer rounding. This is what 'market-neutral'
    means for a regression pair — NOT dollar-balanced, which only holds at beta=1."""
    panel = load_panel(U, START, END, SyntheticProvider(seed=5))
    j = Journal(keep_in_memory=True)
    run_backtest(Arb(), panel, START, END, journal=j)
    hedged = [c for c in j.records.get("positions", []) if c.get("hedge_symbol")]
    assert hedged
    for c in hedged:
        beta = abs(float(c["entry_features"]["beta"]))
        main_notional = c["entry_px"] * c["qty"]
        hedge_notional = c["hedge_entry_px"] * c["hedge_qty"]
        target = beta * main_notional
        # hedge_qty = int(target / hedge_px), so hedge_notional lands within one
        # share of the beta-neutral target.
        assert abs(hedge_notional - target) <= c["hedge_entry_px"] + 1e-6
