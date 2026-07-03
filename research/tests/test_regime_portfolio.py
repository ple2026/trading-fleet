"""Regime-rotation experiment tests (no network — synthetic panel).

Pin the experiment's honesty guarantees: the rule mapping, T+1 no-lookahead
(weights change strictly AFTER the regime flips), de-risking actually engaging in
a crash, and switch costs being charged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.fleet.regime import classify_price_regime
from research.fleet.types import RegimeTag
from research.scripts.regime_portfolio import WARMUP, run_rotation, target_weights


def _df(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range("2015-01-01", periods=len(closes))
    return pd.DataFrame({
        "o": closes, "h": closes * 1.005, "l": closes * 0.995,
        "c": closes, "v": 1e6,
    }, index=idx)


def _crash_panel(n_up: int = 320, n_down: int = 120) -> dict[str, pd.DataFrame]:
    """SPY grinds up then crashes ~1%/day; TLT/GLD stay flat."""
    up = 100.0 * (1.003) ** np.arange(n_up)
    down = up[-1] * (0.99) ** np.arange(1, n_down + 1)
    spy = np.concatenate([up, down])
    flat = np.full(len(spy), 100.0)
    return {"SPY": _df(spy), "TLT": _df(flat), "GLD": _df(flat)}


def test_target_weights_mapping():
    assert target_weights(RegimeTag.TRENDING_UP, "hold") == {"SPY": 1.0}
    assert target_weights(RegimeTag.RISK_OFF, "hold") == {"SPY": 1.0}   # hold never sells
    assert target_weights(RegimeTag.RISK_OFF, "cash") == {}
    assert target_weights(RegimeTag.RISK_OFF, "rotate") == {"TLT": 0.5, "GLD": 0.5}
    # Any healthy regime keeps the market position under every active rule.
    for label in (RegimeTag.TRENDING_UP, RegimeTag.RANGE, RegimeTag.HIGH_VOL):
        assert target_weights(label, "cash") == {"SPY": 1.0}
        assert target_weights(label, "rotate") == {"SPY": 1.0}


def test_no_lookahead_weights_lag_regime_by_one_day():
    panel = _crash_panel()
    spy = panel["SPY"]
    _, weights, _ = run_rotation(panel, "cash")
    # First day the classifier calls RISK_OFF (point-in-time, same call the sim uses).
    first_off = None
    for t in spy.index[WARMUP:]:
        if classify_price_regime(spy, t).label == RegimeTag.RISK_OFF:
            first_off = t
            break
    assert first_off is not None, "synthetic crash never triggered RISK_OFF"
    # On the flip day itself the portfolio must STILL be invested (T+1 discipline)…
    assert weights.loc[first_off, "SPY"] == 1.0
    # …and de-risked strictly after.
    later = weights.loc[weights.index > first_off, "SPY"]
    assert (later == 0.0).any()


def test_derisking_beats_holding_through_a_crash():
    panel = _crash_panel()
    hold, _, _ = run_rotation(panel, "hold")
    cash, _, sw = run_rotation(panel, "cash")
    assert sw >= 1                                # it actually switched
    assert cash.iloc[-1] > hold.iloc[-1]          # sidestepped part of the crash


def test_switch_costs_are_charged():
    panel = _crash_panel()
    free, _, _ = run_rotation(panel, "cash", cost_bps=0.0)
    costly, _, _ = run_rotation(panel, "cash", cost_bps=500.0)   # absurd, to be visible
    assert costly.iloc[-1] < free.iloc[-1]
