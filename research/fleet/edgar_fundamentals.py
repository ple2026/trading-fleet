"""SEC EDGAR XBRL fundamentals — free, point-in-time, delisted-capable CAN SLIM.

The Tiingo free key only covers the DOW 30, and yfinance gives ~5 quarters with no
filing date and nothing for delisted names — neither can gate BREAKOUT on a
survivorship-free universe historically. EDGAR can: the XBRL ``companyconcept`` API
returns every quarterly EPS a company ever filed (AAPL back to 2008), each stamped
with its real ``filed`` date, for every filer including delisted ones.

Two things this module gets right that a naive read would not:
  * **Point-in-time by construction.** The XBRL ``filed`` date IS the public release
    date, so a quarter is known exactly when it was filed — no lookahead, and no
    guessed publication lag (unlike the Tiingo plane, which only has period ends).
  * **Quarterly vs year-to-date.** XBRL reports both a 3-month and a cumulative
    (6/9-month) figure for the same period end. We keep only ~90-day-duration facts,
    so "EPS" means the quarter, not the YTD roll-up.

CAN SLIM growth is year-over-year (same quarter, prior year) — computed by matching
each quarter to the one ~365 days earlier, which tolerates the occasional gap.

Limitation: ``company_tickers.json`` maps only *current* filers ticker->CIK, so a
name that has since delisted may not resolve here. That is fine downstream — a name
without fundamentals is simply not EPS-gated (the gate skips it), it is never
dropped. Supplying a historical CIK map (or paid feed) closes that gap.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .fundamentals import FundamentalsPanel

EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
DEFAULT_USER_AGENT = "trading-fleet-research contact@example.com"


class EdgarClient:
    """Fetches SEC XBRL concepts, cached to disk. Polite User-Agent per SEC rules."""

    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"

    def __init__(self, user_agent: str | None = None, cache_dir: str = "data/edgar") -> None:
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ticker_map: dict[str, int] | None = None

    def _get(self, url: str) -> dict:
        import requests
        r = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
        r.raise_for_status()
        return r.json()

    def ticker_map(self) -> dict[str, int]:
        if self._ticker_map is None:
            cache = self.cache_dir / "company_tickers.json"
            data = json.loads(cache.read_text()) if cache.exists() else self._get(self.TICKERS_URL)
            if not cache.exists():
                cache.write_text(json.dumps(data))
            self._ticker_map = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
        return self._ticker_map

    def concept(self, cik: int, tag: str) -> list[dict] | None:
        """Unit points for one us-gaap concept, or None if the filer never reported it."""
        cache = self.cache_dir / f"CIK{cik:010d}_{tag}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            import requests
            try:
                data = self._get(self.CONCEPT_URL.format(cik=cik, tag=tag))
            except requests.HTTPError:
                cache.write_text(json.dumps({"units": {}}))   # negative-cache 404s
                return None
            cache.write_text(json.dumps(data))
        units = data.get("units") or {}
        if not units:
            return None
        return units[next(iter(units))]


def _quarterly(points: list[dict]) -> list[tuple[date, date, float]]:
    """Keep ~3-month (80-100 day) 10-Q/10-K facts, deduped by period end to the
    EARLIEST filing — the as-first-reported value and the date it was first public
    (§7e). Later filings repeat old quarters as comparatives; using their filing date
    would make a quarter look known months after it actually was. Returns
    [(period_end, filed, value)] sorted by period end."""
    by_end: dict[date, tuple[date, date, float]] = {}
    for x in points:
        if x.get("form") not in ("10-Q", "10-K") or not x.get("start"):
            continue
        start, end = date.fromisoformat(x["start"]), date.fromisoformat(x["end"])
        if not 80 <= (end - start).days <= 100:
            continue
        filed = date.fromisoformat(x["filed"])
        prev = by_end.get(end)
        if prev is None or filed < prev[1]:     # earliest filing wins (first report)
            by_end[end] = (end, filed, float(x["val"]))
    return sorted(by_end.values())


def _yoy_growth(ends: list[date], vals: list[float]) -> list[float]:
    """Year-over-year growth per quarter: match each to the quarter ~365d earlier
    (± 45d). NaN when no prior-year quarter exists. Denominator uses abs() so a sign
    flip in EPS does not invert the growth sign nonsensically."""
    out: list[float] = []
    for i, end in enumerate(ends):
        target = end - timedelta(days=365)
        match = None
        for j in range(i):
            if abs((ends[j] - target).days) <= 45:
                match = j
        if match is None or vals[match] == 0:
            out.append(np.nan)
        else:
            out.append((vals[i] - vals[match]) / abs(vals[match]) * 100.0)
    return out


def build_edgar_frame(ticker: str, client: EdgarClient) -> pd.DataFrame:
    """One ticker's CAN SLIM frame from EDGAR, indexed by FILED date (point-in-time)."""
    cik = client.ticker_map().get(ticker.upper())
    if cik is None:
        return pd.DataFrame()

    def _series(tags: tuple[str, ...]) -> list[tuple[date, date, float]]:
        for tag in tags:
            pts = client.concept(cik, tag)
            if pts:
                q = _quarterly(pts)
                if q:
                    return q
        return []

    eps = _series(EPS_TAGS)
    if not eps:
        return pd.DataFrame()
    ends = [e for e, _, _ in eps]
    filed = [f for _, f, _ in eps]
    vals = [v for _, _, v in eps]
    eps_growth = _yoy_growth(ends, vals)

    rev = _series(REVENUE_TAGS)
    rev_growth_by_end: dict[date, float] = {}
    if rev:
        r_ends = [e for e, _, _ in rev]
        r_vals = [v for _, _, v in rev]
        for e, g in zip(r_ends, _yoy_growth(r_ends, r_vals)):
            rev_growth_by_end[e] = g

    df = pd.DataFrame({
        "period_end": pd.to_datetime(ends),
        "eps": vals,
        "eps_growth": eps_growth,
        "revenue_growth": [rev_growth_by_end.get(e, np.nan) for e in ends],
        "roe": np.nan,
    }, index=pd.to_datetime(filed))
    df.index.name = "known"
    # Acceleration is computed in period-end order (rows are already sorted so);
    # then sort by the KNOWN/filed date so FundamentalsPanel.as_of picks the latest
    # actually-released quarter.
    df["eps_growth_accelerating"] = df["eps_growth"].diff() > 0
    return df.sort_index()


def edgar_fundamentals_panel(
    tickers: list[str],
    client: EdgarClient | None = None,
    overlay: dict[str, dict[str, float]] | None = None,
) -> FundamentalsPanel:
    """Build a point-in-time FundamentalsPanel from EDGAR for the given tickers.
    Uncovered / unresolvable tickers are simply omitted (their gate is skipped)."""
    client = client or EdgarClient()
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            frame = build_edgar_frame(t, client)
        except Exception:
            continue
        if not frame.empty:
            frames[t.upper()] = frame
    return FundamentalsPanel(frames, overlay)
