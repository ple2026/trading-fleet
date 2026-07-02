"""Price-provider tests that need no network.

TiingoProvider's value is its Parquet cache + range logic (so a repeated backtest
doesn't hammer the API); we test that by stubbing the single network method. The
synthetic fallback selection is checked too.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from research.fleet import data as data_mod
from research.fleet.data import SyntheticProvider, TiingoProvider, default_provider


def _bars(dates: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"o": 1.0, "h": 1.0, "l": 1.0, "c": range(len(idx)), "v": 1.0}, index=idx
    ).astype(float)


def test_tiingo_caches_and_slices(monkeypatch, tmp_path):
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    prov = TiingoProvider(cache_dir=str(tmp_path / "tiingo"))

    calls: list[tuple] = []

    def fake_fetch(symbol, start, end):
        calls.append((symbol, start, end))
        return _bars(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])

    monkeypatch.setattr(prov, "_fetch", fake_fetch)

    # First call fetches once and writes the Parquet cache.
    out = prov.daily_bars("AAPL", datetime(2020, 1, 2), datetime(2020, 1, 7))
    assert len(out) == 4 and len(calls) == 1
    assert (tmp_path / "tiingo" / "AAPL.parquet").exists()

    # A covered sub-range is served from cache — no second fetch.
    sub = prov.daily_bars("AAPL", datetime(2020, 1, 3), datetime(2020, 1, 6))
    assert list(sub.index.day) == [3, 6] and len(calls) == 1


def test_tiingo_requires_key(monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    try:
        TiingoProvider()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_default_provider_falls_back_to_synthetic(monkeypatch):
    for var in ("TIINGO_API_KEY", "ALPACA_KEY_ID", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(default_provider(), SyntheticProvider)


def test_default_provider_prefers_tiingo(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    # Point the cache at a temp dir so construction doesn't touch the repo.
    monkeypatch.setattr(data_mod.TiingoProvider, "__init__",
                        lambda self: setattr(self, "token", "x"))
    assert isinstance(default_provider(), TiingoProvider)
