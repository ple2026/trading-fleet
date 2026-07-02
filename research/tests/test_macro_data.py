"""FRED macro-plane tests (no network — the FRED client is stubbed).

Pin the transforms, point-in-time discipline (no lookahead), and that run_backtest
actually feeds MACRO's real `ctx.macro` path when a macro panel is supplied.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from research.fleet.backtest import run_backtest
from research.fleet.bots.macro import Macro
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.journal import Journal
from research.fleet.macro_data import MACRO_SERIES, MacroPanel, build_macro_frame


class _StubFred:
    """Returns a deterministic ramp for every series id, so transforms are checkable."""

    def series(self, series_id, start, end):
        idx = pd.bdate_range(start, end)
        return pd.Series(np.arange(len(idx), dtype=float), index=idx, name=series_id)


def test_build_macro_frame_has_all_features_and_transforms():
    frame = build_macro_frame(_StubFred(), datetime(2019, 1, 1), datetime(2020, 1, 1))
    assert set(frame.columns) == set(MACRO_SERIES)
    tail = frame.dropna().iloc[-1]
    # On a unit-step ramp: a 3-month CHANGE is +63 business steps; the negated one
    # (fed_stance) is -63; a level feature just tracks the ramp (large, positive).
    assert round(tail["real_rate_trend"]) == 63       # change_3m
    assert round(tail["fed_stance"]) == -63           # neg_change_3m
    assert tail["yield_curve_slope"] > 0              # level
    assert tail["usd_trend"] > 0                      # return_3m of an increasing ramp


def test_macro_panel_as_of_is_point_in_time():
    idx = pd.bdate_range("2020-01-01", periods=10)
    frame = pd.DataFrame({"vix": range(10), "usd_trend": range(10)}, index=idx).astype(float)
    panel = MacroPanel(frame)
    # as_of returns the latest row on/ before the date — never a future row.
    at = panel.as_of(idx[4].to_pydatetime())
    assert at["vix"] == 4.0
    # A date before any observation yields an empty dict (nothing known yet).
    assert panel.as_of(datetime(2019, 1, 1)) == {}
    # A date after the last row uses the last row (no peeking past the frame).
    assert panel.as_of(datetime(2021, 1, 1))["vix"] == 9.0


def test_run_backtest_feeds_macro_path():
    U = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "UUP", "DBC", "EEM"]
    start, end = datetime(2020, 1, 1), datetime(2023, 12, 31)
    panel = load_panel(U, start, end, SyntheticProvider(seed=9))

    class _StubMacro:
        def as_of(self, date):
            return {"vix": 30.0, "fed_stance": 0.02, "yield_curve_slope": 0.5,
                    "usd_trend": 0.01, "oil_trend": 0.03, "real_rate_trend": -0.01,
                    "credit_spread": 0.2}

    j = Journal(keep_in_memory=True)
    run_backtest(Macro(), panel, start, end, journal=j, macro=_StubMacro())
    taken = [s for s in j.records.get("signals", []) if s["action"] == "taken"]
    assert taken, "MACRO took no trades"
    # With ctx.macro present, the state vector is built on the real (macro) path.
    assert all(s["features"]["mode"] == "macro" for s in taken)


def test_run_backtest_without_macro_uses_proxy():
    U = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "UUP", "DBC", "EEM"]
    start, end = datetime(2020, 1, 1), datetime(2023, 12, 31)
    panel = load_panel(U, start, end, SyntheticProvider(seed=9))
    j = Journal(keep_in_memory=True)
    run_backtest(Macro(), panel, start, end, journal=j)   # no macro
    taken = [s for s in j.records.get("signals", []) if s["action"] == "taken"]
    assert taken and all(s["features"]["mode"] == "proxy" for s in taken)
