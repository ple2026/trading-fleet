"""EDGAR XBRL fundamentals tests (no network — the SEC client is stubbed).

Pin the two things a naive XBRL read gets wrong: keeping only true quarterly (not
year-to-date) facts, and dating each quarter to its FIRST filing (not a later
comparative). Plus YoY growth and the FundamentalsPanel integration.
"""

from __future__ import annotations

from datetime import datetime

from research.fleet.edgar_fundamentals import (
    _quarterly,
    _yoy_growth,
    build_edgar_frame,
    edgar_fundamentals_panel,
)


def _pt(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def test_quarterly_keeps_3month_and_earliest_filing():
    points = [
        _pt("2023-01-01", "2023-03-31", 1.0, "2023-05-01"),          # quarterly -> keep
        _pt("2023-01-01", "2023-06-30", 3.0, "2023-08-01"),          # 6-mo YTD -> drop
        # same Q1 repeated as a comparative in a later filing -> must NOT override the
        # earlier filing date (that would delay when it looks "known").
        _pt("2023-01-01", "2023-03-31", 1.0, "2024-05-01"),
        _pt("2023-04-01", "2023-06-30", 1.2, "2023-08-01"),          # quarterly -> keep
    ]
    q = _quarterly(points)
    ends = {e: (f, v) for e, f, v in q}
    from datetime import date
    assert set(ends) == {date(2023, 3, 31), date(2023, 6, 30)}       # YTD dropped
    assert ends[date(2023, 3, 31)][0] == date(2023, 5, 1)            # earliest filing wins


def test_yoy_growth_matches_prior_year_quarter():
    from datetime import date
    ends = [date(2022, 3, 31), date(2022, 6, 30), date(2023, 3, 31), date(2023, 6, 30)]
    vals = [1.0, 1.0, 1.5, 2.0]
    g = _yoy_growth(ends, vals)
    assert g[0] != g[0]                       # NaN: no prior-year quarter
    assert round(g[2]) == 50                  # 1.5 vs 1.0 a year earlier -> +50%
    assert round(g[3]) == 100                 # 2.0 vs 1.0 -> +100%


class _StubClient:
    """Serves a two-year quarterly EPS ramp for AAPL only."""

    def ticker_map(self):
        return {"AAPL": 320193}

    def concept(self, cik, tag):
        if tag != "EarningsPerShareDiluted":
            return None
        quarters = [
            ("2022-01-01", "2022-03-31", 1.0, "2022-05-01"),
            ("2022-04-01", "2022-06-30", 1.1, "2022-08-01"),
            ("2023-01-01", "2023-03-31", 1.5, "2023-05-01"),
            ("2023-04-01", "2023-06-30", 1.65, "2023-08-01"),
        ]
        return [_pt(s, e, v, f) for s, e, v, f in quarters]


def test_build_frame_and_panel_point_in_time():
    frame = build_edgar_frame("AAPL", _StubClient())
    assert not frame.empty and "eps_growth" in frame
    # 2023-Q1 vs 2022-Q1: 1.5 vs 1.0 -> +50%.
    q1_2023 = frame[frame["period_end"] == "2023-03-31"].iloc[0]
    assert round(q1_2023["eps_growth"]) == 50

    panel = edgar_fundamentals_panel(["AAPL", "NVDA"], client=_StubClient())
    assert set(panel.frames) == {"AAPL"}      # NVDA unresolved -> omitted, no crash
    # Known only from its FILING date: invisible on 2023-04-01, visible after 2023-05-01.
    assert not panel.as_of(datetime(2023, 4, 1)).get("AAPL", {}).get("eps_growth")
    assert round(panel.as_of(datetime(2023, 6, 1))["AAPL"]["eps_growth"]) == 50
