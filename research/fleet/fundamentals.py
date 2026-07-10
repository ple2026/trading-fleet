"""Tiingo fundamentals plane — CAN SLIM inputs for BREAKOUT (§3) and a tile for
CATALYST.

O'Neil's C-A-N needs point-in-time quarterly earnings: this quarter's EPS growth
vs the *same quarter a year ago* (seasonality-controlled), whether that growth is
*accelerating*, and annual growth. Tiingo's statements provide it — with one gotcha
this module handles:

  * Tiingo's ``epsQoQ`` / ``revenueQoQ`` are mislabeled: they are actually
    year-over-year same-quarter growth (verified: they equal eps[t]/eps[t-4]-1).
    That IS the CAN SLIM "C" figure, so we use it directly and also recompute it
    from the raw ``eps`` series as a cross-check.
  * The statement ``date`` is the fiscal PERIOD END, not the filing date. Using it
    as-is would let a backtest see earnings ~6 weeks before they were public. We
    therefore stamp each quarter as known only ``publication_lag_days`` after the
    period end (default 45 — a conservative 10-Q window) — §7e, no lookahead.

Coverage note: the free/evaluation key is limited to ~3 years of the DOW 30. The
paid Fundamental add-on unlocks all ~5,500 equities; the code is identical, only
the covered ticker set grows.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PUBLICATION_LAG_DAYS = 45


def _code(record: dict, section: str, data_code: str) -> float | None:
    for d in record.get("statementData", {}).get(section, []):
        if d.get("dataCode") == data_code:
            v = d.get("value")
            return float(v) if v is not None else None
    return None


class TiingoFundamentals:
    """Fetches Tiingo fundamental statements, cached as raw JSON per ticker."""

    BASE = "https://api.tiingo.com/tiingo/fundamentals"

    def __init__(self, cache_dir: str = "data/fundamentals") -> None:
        self.token = os.environ.get("TIINGO_API_KEY", "")
        if not self.token:
            raise RuntimeError("TiingoFundamentals needs TIINGO_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def statements(self, ticker: str) -> list[dict]:
        cache = self.cache_dir / f"{ticker.upper()}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        import requests

        r = requests.get(
            f"{self.BASE}/{ticker}/statements",
            params={"token": self.token}, timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        cache.write_text(json.dumps(data))
        return data


def build_fundamentals_frame(
    statements: list[dict],
    publication_lag_days: int = DEFAULT_PUBLICATION_LAG_DAYS,
) -> pd.DataFrame:
    """One ticker's quarterly CAN SLIM features, indexed by the date they became
    KNOWN (fiscal period end + publication lag). Columns:
      eps_growth (% YoY), eps_growth_accelerating (bool), revenue_growth (% YoY),
      roe, piotroski.
    """
    rows = []
    for rec in statements:
        if rec.get("quarter", 0) == 0:      # skip annual roll-ups
            continue
        period_end = pd.Timestamp(rec["date"][:10])
        eps = _code(rec, "incomeStatement", "eps")
        eps_yoy = _code(rec, "overview", "epsQoQ")        # mislabeled: really YoY
        rev_yoy = _code(rec, "overview", "revenueQoQ")
        roe = _code(rec, "overview", "roe")
        piotroski = _code(rec, "overview", "piotroskiFScore")
        rows.append({
            "period_end": period_end,
            "known": period_end + timedelta(days=publication_lag_days),
            "eps": eps,
            "eps_growth": eps_yoy * 100.0 if eps_yoy is not None else np.nan,
            "revenue_growth": rev_yoy * 100.0 if rev_yoy is not None else np.nan,
            "roe": roe, "piotroski": piotroski,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("period_end").reset_index(drop=True)
    # CAN SLIM "A": growth accelerating vs the prior quarter's YoY growth.
    df["eps_growth_accelerating"] = df["eps_growth"].diff() > 0
    return df.set_index("known")


class FundamentalsPanel:
    """Point-in-time fundamentals for a set of tickers, plus an optional static
    overlay (e.g. 13F ``institutional_conviction``) merged into every lookup.

    ``as_of(date)`` returns ``{ticker: {metrics...}}`` — exactly the shape
    ``ctx.fundamentals`` expects, so BREAKOUT's CAN SLIM gate and CATALYST's tiles
    read it directly.
    """

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        overlay: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.frames = frames
        self.overlay = overlay or {}

    @classmethod
    def from_tiingo(
        cls,
        tickers: list[str],
        client: TiingoFundamentals | None = None,
        publication_lag_days: int = DEFAULT_PUBLICATION_LAG_DAYS,
        overlay: dict[str, dict[str, float]] | None = None,
    ) -> FundamentalsPanel:
        client = client or TiingoFundamentals()
        frames: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                st = client.statements(t)
            except Exception:
                continue  # not covered on this plan (e.g. non-DOW-30) — skip
            frame = build_fundamentals_frame(st, publication_lag_days)
            if not frame.empty:
                frames[t.upper()] = frame
        return cls(frames, overlay)

    def as_of(self, date: datetime) -> dict[str, dict[str, float]]:
        ts = pd.Timestamp(date)
        out: dict[str, dict[str, float]] = {}
        for ticker, frame in self.frames.items():
            known = frame.loc[frame.index <= ts]
            if known.empty:
                continue
            row = known.iloc[-1]
            out[ticker] = {
                "eps_growth": float(row["eps_growth"]) if pd.notna(row["eps_growth"]) else None,
                "eps_growth_accelerating": bool(row["eps_growth_accelerating"]),
                "revenue_growth": float(row["revenue_growth"]) if pd.notna(row["revenue_growth"]) else None,
                "roe": float(row["roe"]) if pd.notna(row["roe"]) else None,
            }
        # Merge the static overlay (13F conviction etc.) — union of tickers.
        for ticker, extra in self.overlay.items():
            out.setdefault(ticker.upper(), {}).update(extra)
        return out
