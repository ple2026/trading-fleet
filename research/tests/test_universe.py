"""Survivorship-free universe selection tests (no network — records built inline).

Pin the core anti-hindsight rule: a name that traded during the window is kept even
if it later delisted; a name that delisted BEFORE the window is dropped; non-US /
non-common-stock rows are excluded.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from research.fleet.universe import (
    TickerRecord,
    filter_panel_by_liquidity,
    sample_universe,
    survivorship_free_universe,
)


def _rec(ticker, start, end, exch="NYSE", atype="Stock", cur="USD"):
    return TickerRecord(ticker, exch, atype, cur,
                        date.fromisoformat(start) if start else None,
                        date.fromisoformat(end) if end else None)


RECORDS = [
    _rec("SURV", "2000-01-01", None),                    # still listed -> keep
    _rec("BOUGHT", "2005-01-01", "2021-06-01"),          # delisted mid-window -> keep
    _rec("DEAD", "1998-01-01", "2010-01-01"),            # delisted before window -> drop
    _rec("FUTURE", "2026-01-01", None),                  # ipo'd after window -> drop
    _rec("LSE", "2000-01-01", None, exch="LSE"),         # non-US exchange -> drop
    _rec("FUND", "2000-01-01", None, atype="ETF"),       # not common stock -> drop
    _rec("EUR", "2000-01-01", None, cur="EUR"),          # non-USD -> drop
]
START, END = datetime(2019, 1, 1), datetime(2024, 12, 31)


def test_includes_delisted_that_traded_in_window():
    u = survivorship_free_universe(RECORDS, START, END)
    assert "BOUGHT" in u                       # the key anti-survivorship case
    assert "SURV" in u


def test_excludes_pre_window_and_post_window_and_non_us():
    u = set(survivorship_free_universe(RECORDS, START, END))
    assert "DEAD" not in u                     # delisted before the window
    assert "FUTURE" not in u                   # not yet listed
    assert {"LSE", "FUND", "EUR"} & u == set()  # non-US / non-common / non-USD


def test_survivors_only_flag_drops_delisted():
    biased = survivorship_free_universe(RECORDS, START, END, include_delisted=False)
    assert "BOUGHT" not in biased              # the bias this whole module fights
    assert "SURV" in biased


def _panel(close: float, vol: float, n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"c": close, "v": vol}, index=idx)


def test_liquidity_filter_drops_illiquid_keeps_benchmark():
    panel = {
        "SPY": _panel(400.0, 50_000_000),      # benchmark, always kept
        "LIQUID": _panel(50.0, 2_000_000),     # $100M/day -> kept
        "SHELL": _panel(0.50, 10_000),         # $5k/day -> dropped
    }
    out = filter_panel_by_liquidity(panel, min_dollar_volume=1_000_000)
    assert set(out) == {"SPY", "LIQUID"}
    # Benchmark is kept even if it were somehow below the threshold.
    only_spy = filter_panel_by_liquidity({"SPY": _panel(1.0, 1.0)}, 1e9)
    assert set(only_spy) == {"SPY"}


def test_sample_is_deterministic_and_bounded():
    tickers = [f"T{i}" for i in range(100)]
    a = sample_universe(tickers, 10, seed=1)
    b = sample_universe(tickers, 10, seed=1)
    assert a == b and len(a) == 10             # reproducible across calls
    assert sample_universe(tickers, 10, seed=2) != a  # seed changes the pick
    assert set(a) <= set(tickers)
