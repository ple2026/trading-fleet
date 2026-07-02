"""Data plane. Loads point-in-time OHLCV panels for the backtest and bots.

Providers are pluggable. Ships with:
  - AlpacaProvider: real daily bars (needs ALPACA_* env)
  - SyntheticProvider: deterministic GBM-with-regimes generator so the whole
    fleet is runnable offline / in CI with zero credentials.

Fundamentals, calendar, and macro loaders are declared here as thin interfaces;
concrete implementations land with their respective bots (FMP, Finnhub, FRED).
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class PriceProvider(Protocol):
    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return OHLCV DataFrame indexed by date with columns o/h/l/c/v."""
        ...


class SyntheticProvider:
    """Deterministic price generator. Seeded per symbol so runs are reproducible.

    Produces trending, ranging, and volatile stretches so regime logic and the
    bots have something meaningful to chew on without any market-data vendor.
    """

    def __init__(self, seed: int = 42, start_price: float = 100.0) -> None:
        self.seed = seed
        self.start_price = start_price

    @staticmethod
    def _symbol_offset(symbol: str) -> int:
        """Stable per-symbol seed offset.

        `hash(symbol)` is salted per process (PYTHONHASHSEED), so using it here
        would make the "deterministic" generator produce different prices on every
        run — silently breaking backtest reproducibility and any walk-forward or
        Tier-A comparison that assumes a fixed data panel. A content hash is stable
        across processes.
        """
        digest = hashlib.blake2b(symbol.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") % 10_000

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + self._symbol_offset(symbol))
        days = pd.bdate_range(start, end)
        n = len(days)
        # Piecewise drift/vol regimes to exercise the regime classifier. Drift is
        # kept SMALL relative to vol (realistic daily signal-to-noise ~0.02-0.04):
        # on near-random data no strategy should show large edge, so backtests read
        # as a plumbing check rather than fake alpha a high-frequency bot compounds.
        drift = np.zeros(n)
        vol = np.full(n, 0.014)
        seg = max(n // 6, 1)
        regimes = [(0.0004, 0.013), (0.0001, 0.012), (-0.0005, 0.024),
                   (0.0005, 0.015), (-0.0002, 0.019), (0.0003, 0.013)]
        for i, (d, v) in enumerate(regimes):
            drift[i * seg:(i + 1) * seg] = d
            vol[i * seg:(i + 1) * seg] = v
        rets = rng.normal(drift, vol)
        close = self.start_price * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0, vol)) * close
        high = close + intraday
        low = close - intraday
        open_ = np.concatenate([[close[0]], close[:-1]])
        volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
        return pd.DataFrame(
            {"o": open_, "h": high, "l": low, "c": close, "v": volume}, index=days
        )


class AlpacaProvider:
    """Daily bars from Alpaca's market-data API. Requires ALPACA_* env vars."""

    BASE = "https://data.alpaca.markets/v2"

    def __init__(self) -> None:
        self.key = os.environ.get("ALPACA_KEY_ID", "")
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "")
        if not self.key or not self.secret:
            raise RuntimeError("AlpacaProvider needs ALPACA_KEY_ID / ALPACA_SECRET_KEY")

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        import requests

        headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        params = {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "timeframe": "1Day",
            "adjustment": "all",
            "limit": 10_000,
        }
        rows: list[dict] = []
        url = f"{self.BASE}/stocks/{symbol}/bars"
        while True:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()
            rows.extend(body.get("bars") or [])
            token = body.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        if not rows:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v"])
        df = pd.DataFrame(rows)
        df["t"] = pd.to_datetime(df["t"]).dt.tz_localize(None)
        df = df.set_index("t").rename(
            columns={"o": "o", "h": "h", "l": "l", "c": "c", "v": "v"}
        )
        return df[["o", "h", "l", "c", "v"]]


class TiingoProvider:
    """Split/dividend-adjusted daily bars from Tiingo. Requires TIINGO_API_KEY.

    Uses the adjusted OHLCV series (adjOpen/High/Low/Close/Volume), so corporate
    actions never masquerade as returns in a backtest. Per-symbol bars are cached
    to Parquet under `cache_dir` (gitignored) and merged across calls, so a repeated
    backtest hits disk, not the API — important on the free tier's rate limits.
    """

    BASE = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, cache_dir: str = "data/tiingo") -> None:
        self.token = os.environ.get("TIINGO_API_KEY", "")
        if not self.token:
            raise RuntimeError("TiingoProvider needs TIINGO_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        import requests

        url = f"{self.BASE}/{symbol}/prices"
        params = {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
            "token": self.token,
            "format": "json",
            "resampleFreq": "daily",
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        cols = ["o", "h", "l", "c", "v"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df.set_index("date").rename(columns={
            "adjOpen": "o", "adjHigh": "h", "adjLow": "l",
            "adjClose": "c", "adjVolume": "v",
        })
        return df[cols].astype(float).sort_index()

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        cache = self.cache_dir / f"{symbol.upper()}.parquet"
        cached = pd.read_parquet(cache) if cache.exists() else None
        covered = (
            cached is not None and not cached.empty
            and cached.index.min() <= start and cached.index.max() >= end
        )
        if not covered:
            fetched = self._fetch(symbol, start, end)
            if cached is None:
                cached = fetched
            elif not fetched.empty:
                cached = pd.concat([cached, fetched]).sort_index()
                cached = cached[~cached.index.duplicated(keep="last")]
            if cached is not None and not cached.empty:
                cached.to_parquet(cache)
        if cached is None or cached.empty:
            return pd.DataFrame(columns=["o", "h", "l", "c", "v"])
        return cached.loc[(cached.index >= start) & (cached.index <= end)]


def default_provider() -> PriceProvider:
    """Best available price plane: Tiingo > Alpaca > synthetic generator.

    Tiingo is preferred for research because it serves adjusted history; Alpaca
    needs both key and secret; the synthetic generator is the zero-credential
    fallback so the fleet always runs.
    """
    if os.environ.get("TIINGO_API_KEY"):
        try:
            return TiingoProvider()
        except RuntimeError:
            pass
    if os.environ.get("ALPACA_KEY_ID") and os.environ.get("ALPACA_SECRET_KEY"):
        try:
            return AlpacaProvider()
        except RuntimeError:
            pass
    return SyntheticProvider()


def load_panel(
    symbols: list[str],
    start: datetime,
    end: datetime,
    provider: PriceProvider | None = None,
) -> dict[str, pd.DataFrame]:
    provider = provider or default_provider()
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = provider.daily_bars(sym, start, end)
        if not df.empty:
            out[sym] = df
    return out


def slice_pit(panel: dict[str, pd.DataFrame], now: datetime) -> dict[str, pd.DataFrame]:
    """Point-in-time slice: drop every bar dated after `now`. Guards lookahead."""
    return {s: df.loc[df.index <= now] for s, df in panel.items()}
