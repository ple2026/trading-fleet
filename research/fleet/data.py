"""Data plane. Loads point-in-time OHLCV panels for the backtest and bots.

Providers are pluggable. Ships with:
  - AlpacaProvider: real daily bars (needs ALPACA_* env)
  - SyntheticProvider: deterministic GBM-with-regimes generator so the whole
    fleet is runnable offline / in CI with zero credentials.

Fundamentals, calendar, and macro loaders are declared here as thin interfaces;
concrete implementations land with their respective bots (FMP, Finnhub, FRED).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
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

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + hash(symbol) % 10_000)
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


def default_provider() -> PriceProvider:
    """AlpacaProvider when credentials exist, else the synthetic generator."""
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
