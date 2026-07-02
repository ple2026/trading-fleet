"""Tiingo fundamentals plane tests (no network).

Pin the CAN SLIM computation, the publication-lag point-in-time discipline, the
13F overlay merge, and that uncovered tickers are skipped rather than fatal.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from research.fleet.backtest import run_backtest
from research.fleet.bots.catalyst import Catalyst
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.fundamentals import (
    DEFAULT_PUBLICATION_LAG_DAYS,
    FundamentalsPanel,
    build_fundamentals_frame,
)
from research.fleet.journal import Journal


def _rec(date: str, year: int, q: int, eps: float, eps_yoy: float, rev_yoy: float) -> dict:
    return {
        "date": date, "year": year, "quarter": q,
        "statementData": {
            "incomeStatement": [{"dataCode": "eps", "value": eps}],
            "overview": [
                {"dataCode": "epsQoQ", "value": eps_yoy},       # really YoY
                {"dataCode": "revenueQoQ", "value": rev_yoy},
                {"dataCode": "roe", "value": 0.30},
            ],
        },
    }


def _statements() -> list[dict]:
    # Growth accelerating across three quarters: 10% -> 15% -> 30% YoY.
    return [
        _rec("2024-03-31", 2024, 1, 1.0, 0.10, 0.05),
        _rec("2024-06-30", 2024, 2, 1.2, 0.15, 0.07),
        _rec("2024-09-30", 2024, 3, 1.5, 0.30, 0.12),
        {"date": "2024-12-31", "year": 2024, "quarter": 0,      # annual — skipped
         "statementData": {"incomeStatement": [], "overview": []}},
    ]


def test_build_frame_computes_canslim():
    f = build_fundamentals_frame(_statements())
    assert len(f) == 3                                  # annual row dropped
    # eps_growth is the YoY figure as a percent.
    assert round(f["eps_growth"].iloc[-1], 1) == 30.0
    # Acceleration: 10 -> 15 -> 30 all rising, so the later quarters accelerate.
    assert f["eps_growth_accelerating"].iloc[-1]
    assert not f["eps_growth_accelerating"].iloc[0]     # no prior quarter to beat
    # Known-date = period end + publication lag (no lookahead).
    period_end = f["period_end"].iloc[-1]
    assert f.index[-1] == period_end + timedelta(days=DEFAULT_PUBLICATION_LAG_DAYS)


def test_panel_as_of_is_point_in_time():
    f = build_fundamentals_frame(_statements())
    panel = FundamentalsPanel({"XYZ": f})
    # Right after the Q3 period end (Sep 30) but before its ~45d filing: Q3 is NOT
    # yet visible — the latest known quarter is Q2.
    early = panel.as_of(datetime(2024, 10, 15)).get("XYZ", {})
    assert round(early["eps_growth"], 1) == 15.0
    # After the lag, Q3 (30%) becomes visible.
    late = panel.as_of(datetime(2024, 11, 30)).get("XYZ", {})
    assert round(late["eps_growth"], 1) == 30.0
    # Before any filing is public: ticker absent.
    assert panel.as_of(datetime(2024, 1, 1)) == {}


def test_overlay_merges_into_fundamentals():
    f = build_fundamentals_frame(_statements())
    panel = FundamentalsPanel({"XYZ": f}, overlay={"XYZ": {"institutional_conviction": 0.8}})
    at = panel.as_of(datetime(2024, 12, 1))["XYZ"]
    assert at["institutional_conviction"] == 0.8 and "eps_growth" in at
    # Overlay can introduce a ticker with no statements at all.
    panel2 = FundamentalsPanel({}, overlay={"NEW": {"institutional_conviction": -1.0}})
    assert panel2.as_of(datetime(2024, 12, 1)) == {"NEW": {"institutional_conviction": -1.0}}


def test_from_tiingo_skips_uncovered_tickers():
    class _Stub:
        def statements(self, ticker):
            if ticker == "AAPL":
                return _statements()
            raise RuntimeError("not covered on this plan (non-DOW-30)")

    panel = FundamentalsPanel.from_tiingo(["AAPL", "NVDA"], client=_Stub())
    assert set(panel.frames) == {"AAPL"}                # NVDA skipped, no crash
    assert not panel.as_of(datetime(2024, 11, 30)).get("NVDA")
    assert panel.as_of(datetime(2024, 11, 30)).get("AAPL")


def test_run_backtest_feeds_fundamentals_to_bots():
    """A fundamentals lookup flows into ctx.fundamentals — here the 13F tile lights
    up CATALYST's mosaic (proving the plumbing, not just the unit)."""
    U = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "META", "XLE", "GLD", "TLT", "EEM"]
    start, end = datetime(2020, 1, 1), datetime(2023, 12, 31)
    panel = load_panel(U, start, end, SyntheticProvider(seed=9))

    class _StubFund:
        def as_of(self, date):
            return {s: {"institutional_conviction": 0.9} for s in U}

    j = Journal(keep_in_memory=True)
    run_backtest(Catalyst(), panel, start, end, journal=j, fundamentals=_StubFund())
    taken = [s for s in j.records.get("signals", []) if s["action"] == "taken"]
    assert taken, "CATALYST took no trades"
    assert any("institutional_conviction" in s["features"] for s in taken)
