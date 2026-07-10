"""OOS runner resilience test (no network — provider stubbed).

The survivorship-free run must survive per-ticker failures (rate limits, delisted
names with no data, reused tickers) and still return a usable partial panel.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from research.scripts.oos import _resilient_panel


def _bars(n: int) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1.0}, index=idx)


class _StubProvider:
    def daily_bars(self, symbol, start, end):
        if symbol == "RATELIMIT":
            raise RuntimeError("429 Too Many Requests")
        if symbol == "SHORT":
            return _bars(50)          # too little history -> skipped
        return _bars(500)             # plenty -> kept


def test_resilient_panel_skips_failures_and_short_history():
    symbols = ["SPY", "GOODCO", "RATELIMIT", "SHORT", "DEADCO"]
    panel, skipped = _resilient_panel(
        symbols, datetime(2018, 1, 1), datetime(2024, 12, 31), provider=_StubProvider()
    )
    # The two loadable names survive; the raiser and the short one are skipped.
    assert set(panel) == {"SPY", "GOODCO", "DEADCO"}
    assert skipped == 2
