"""Meta-allocator + correlation monitor tests (§8).

Pin the governance invariants: weights are a valid simplex inside the [10%, 40%]
caps, risk-parity down-weights the noisier bot, the bandit tilt only leans toward
positive OOS Sharpe, and the correlation monitor freezes a crowded fleet.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from research.fleet.allocator import (
    FLEET_FREEZE,
    MAX_WEIGHT,
    MIN_WEIGHT,
    PAIR_ALERT,
    _apply_caps,
    allocate,
    bandit_tilt,
    correlation_monitor,
    risk_parity_weights,
    run_fleet_backtest,
)
from research.fleet.bots.arb import Arb
from research.fleet.bots.breakout import Breakout
from research.fleet.bots.catalyst import Catalyst
from research.fleet.bots.macro import Macro
from research.fleet.data import SyntheticProvider, load_panel

UNIVERSE = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "META", "XLE", "OIH",
            "SMH", "XLF", "KRE", "GLD", "GDX", "XLK", "TLT", "IEF",
            "UUP", "DBC", "USO", "EEM", "EWJ"]
START = datetime(2020, 1, 1)
END = datetime(2023, 12, 31)
BOTS = {"breakout": Breakout, "arb": Arb, "catalyst": Catalyst, "macro": Macro}


@pytest.fixture(scope="module")
def fleet():
    panel = load_panel(UNIVERSE, START, END, SyntheticProvider(seed=5))
    return run_fleet_backtest(BOTS, panel, START, END)


def _rng_returns(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=252)
    return pd.DataFrame(
        {"breakout": rng.normal(0, 0.01, 252),
         "arb": rng.normal(0, 0.003, 252),      # deliberately the quietest bot
         "catalyst": rng.normal(0, 0.02, 252),
         "macro": rng.normal(0, 0.012, 252)},
        index=idx,
    )


def test_fleet_backtest_returns_all_streams(fleet):
    assert set(fleet.returns.columns) == set(BOTS)
    assert not fleet.equity.empty
    # Each bot's equity curve shares the panel's date index.
    assert (fleet.equity.index == fleet.returns.index.union(fleet.equity.index)).all()


def test_risk_parity_downweights_the_noisy_bot():
    w = risk_parity_weights(_rng_returns(1))
    assert abs(w.sum() - 1.0) < 1e-9
    # Lowest-vol bot (arb) must get the LARGEST inverse-vol weight.
    assert w.idxmax() == "arb"
    assert w.idxmin() == "catalyst"  # noisiest -> smallest


def test_apply_caps_produces_valid_capped_simplex():
    # A wildly skewed vector must be pulled inside [MIN, MAX] and re-normalized.
    skewed = pd.Series({"a": 0.97, "b": 0.01, "c": 0.01, "d": 0.01})
    w = _apply_caps(skewed)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= MIN_WEIGHT - 1e-9).all() and (w <= MAX_WEIGHT + 1e-9).all()


def test_apply_caps_rejects_infeasible():
    with pytest.raises(ValueError):
        _apply_caps(pd.Series({"a": 0.5, "b": 0.5}), lo=0.6, hi=0.9)


def test_bandit_tilt_favors_higher_sharpe():
    base = pd.Series({"breakout": 0.25, "arb": 0.25, "catalyst": 0.25, "macro": 0.25})
    sharpe = pd.Series({"breakout": 2.0, "arb": 0.0, "catalyst": 0.0, "macro": 0.0})
    w = bandit_tilt(base, sharpe)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= MIN_WEIGHT - 1e-9).all() and (w <= MAX_WEIGHT + 1e-9).all()
    # The high-Sharpe bot must end up weighted above the flat ones.
    assert w["breakout"] > w["arb"]
    # Negative Sharpe never tilts a bot below the others via the multiplier.
    neg = bandit_tilt(base, pd.Series({"breakout": -3.0, "arb": 0, "catalyst": 0, "macro": 0}))
    assert abs(neg["breakout"] - neg["arb"]) < 1e-9


def test_correlation_monitor_flags_and_freezes():
    rng = np.random.default_rng(7)
    shared = rng.normal(0, 0.01, 252)          # a common beta factor
    idx = pd.bdate_range("2022-01-01", periods=252)
    crowded = pd.DataFrame(
        {name: shared + rng.normal(0, 0.001, 252) for name in BOTS}, index=idx
    )
    rep = correlation_monitor(crowded)
    assert rep.hot_pairs, "near-identical streams should trip the pair alert"
    assert all(c > PAIR_ALERT for *_, c in rep.hot_pairs)
    assert rep.fleet_avg > FLEET_FREEZE and rep.freeze

    independent = _rng_returns(2)
    rep2 = correlation_monitor(independent)
    assert not rep2.freeze and rep2.fleet_avg < FLEET_FREEZE


def test_allocate_end_to_end(fleet):
    alloc = allocate(fleet.returns, total_usd=40_000.0, at=END)
    assert abs(sum(alloc.weights.values()) - 1.0) < 1e-6
    assert all(MIN_WEIGHT - 1e-6 <= w <= MAX_WEIGHT + 1e-6 for w in alloc.weights.values())
    # Capital is weights * total.
    assert abs(sum(alloc.capital_usd.values()) - 40_000.0) < 1.0
    assert alloc.reason
    assert set(alloc.weights) == set(BOTS)
