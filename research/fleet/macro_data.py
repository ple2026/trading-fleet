"""FRED macro data plane — the real state vector behind MACRO (§6).

MACRO already knows how to consume a macro dict (``ctx.macro``): its playbook nudges
the ETF-derived trends with rates, the dollar, oil, credit, and vol. Offline that
dict is ``None`` and the bot runs on the ETF proxy alone. This module builds the
dict from FRED so the real Druckenmiller path runs.

Point-in-time discipline (§7e): every series is shifted by a per-series publication
lag before use (daily market series ~1 day; monthly FEDFUNDS ~30 days), so a
backtest never sees a number before it was actually released. Values are cached to
Parquet under ``data/`` (gitignored).

Feature scaling is intentionally left in the same rough magnitude MACRO's playbook
coefficients expect (levels for the curve/VIX; ~quarterly changes/returns for the
trends). Those coefficients are placeholders to be re-fit — this module's job is to
deliver the real series faithfully, point-in-time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Trading days in a quarter — the window for the "3-month" change/return features.
_QUARTER = 63


@dataclass(frozen=True)
class _Series:
    fred_id: str
    kind: str        # level | change_3m | neg_change_3m | return_3m
    lag_days: int    # publication lag applied before the value is "known"


# The macro state vector, mapped to FRED series and the transform MACRO expects.
MACRO_SERIES: dict[str, _Series] = {
    "yield_curve_slope": _Series("T10Y2Y", "level", 1),        # 10y-2y, %
    "real_rate_trend": _Series("DFII10", "change_3m", 1),      # 10y TIPS yield, %
    "fed_stance": _Series("FEDFUNDS", "neg_change_3m", 30),    # falling FFR = dovish
    "credit_spread": _Series("BAMLH0A0HYM2", "change_3m", 1),  # HY OAS widening = risk-off
    "usd_trend": _Series("DTWEXBGS", "return_3m", 1),          # broad USD index
    "oil_trend": _Series("DCOILWTICO", "return_3m", 1),        # WTI
    "vix": _Series("VIXCLS", "level", 1),                      # VIX level
}


class FredClient:
    """Fetches FRED observation series, cached per series to Parquet."""

    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None, cache_dir: str = "data/fred") -> None:
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("FredClient needs FRED_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(self, series_id: str, start: datetime, end: datetime) -> pd.Series:
        import requests

        params = {
            "series_id": series_id, "api_key": self.api_key, "file_type": "json",
            "observation_start": start.strftime("%Y-%m-%d"),
            "observation_end": end.strftime("%Y-%m-%d"),
        }
        r = requests.get(self.BASE, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        idx, vals = [], []
        for o in obs:
            if o["value"] == ".":
                continue
            idx.append(pd.Timestamp(o["date"]))
            vals.append(float(o["value"]))
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name=series_id)

    def series(self, series_id: str, start: datetime, end: datetime) -> pd.Series:
        cache = self.cache_dir / f"{series_id}.parquet"
        cached = pd.read_parquet(cache)[series_id] if cache.exists() else None
        covered = (
            cached is not None and len(cached)
            and cached.index.min() <= pd.Timestamp(start)
            and cached.index.max() >= pd.Timestamp(end)
        )
        if not covered:
            fetched = self._fetch(series_id, start, end)
            if cached is None:
                cached = fetched
            elif len(fetched):
                cached = pd.concat([cached, fetched]).sort_index()
                cached = cached[~cached.index.duplicated(keep="last")]
            if cached is not None and len(cached):
                cached.to_frame().to_parquet(cache)
        if cached is None:
            return pd.Series(dtype=float, name=series_id)
        return cached.loc[(cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))]


def _transform(series: pd.Series, kind: str) -> pd.Series:
    if kind == "level":
        return series
    if kind == "change_3m":
        return series.diff(_QUARTER)
    if kind == "neg_change_3m":
        return -series.diff(_QUARTER)
    if kind == "return_3m":
        return series.pct_change(_QUARTER)
    raise ValueError(f"unknown transform {kind}")


def build_macro_frame(
    client: FredClient, start: datetime, end: datetime
) -> pd.DataFrame:
    """Build the daily, point-in-time macro feature frame over [start, end].

    Each raw series is aligned to business days, forward-filled, shifted by its
    publication lag, then transformed. A ~200-day head buffer is fetched so the
    3-month features are defined from the very first date of the window.
    """
    head = start - timedelta(days=260)
    idx = pd.bdate_range(head, end)
    frame = pd.DataFrame(index=idx)
    for feature, spec in MACRO_SERIES.items():
        raw = client.series(spec.fred_id, head, end)
        aligned = raw.reindex(idx).ffill()
        lagged = aligned.shift(spec.lag_days)   # respect publication delay
        frame[feature] = _transform(lagged, spec.kind)
    return frame.loc[frame.index >= pd.Timestamp(start)]


class MacroPanel:
    """Point-in-time macro lookup: ``as_of(date)`` -> MACRO's ``ctx.macro`` dict."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    @classmethod
    def from_fred(
        cls, start: datetime, end: datetime, client: FredClient | None = None
    ) -> MacroPanel:
        return cls(build_macro_frame(client or FredClient(), start, end))

    def as_of(self, date: datetime) -> dict[str, float]:
        rows = self.frame.loc[self.frame.index <= pd.Timestamp(date)]
        if rows.empty:
            return {}
        row = rows.iloc[-1].dropna()
        return {k: float(v) for k, v in row.items()}
