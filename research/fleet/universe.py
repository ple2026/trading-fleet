"""Survivorship-free universe construction from Tiingo's supported-ticker list.

A backtest whose universe is "the S&P 500 as it stands today" is rigged: it only
ever sees the winners that survived. Tiingo publishes every ticker it has ever
covered — ~16k US common stocks, ~6.5k of them already delisted — each with a
``startDate`` and ``endDate``. Selecting the names that were *live during the
backtest window* (delisted ones included) removes the hindsight bias: the bot sees
the failures and the acquisitions too, not just the survivors.

Honest remaining limits (this is a big improvement, not a full fix):
  * No point-in-time *index membership* — this is "all listed common stock", not
    "the S&P 500 on date X". That needs a paid constituents feed.
  * No liquidity/market-cap filter in the ticker file, so the raw pool includes
    micro-caps and shells. Sample and/or validate against the price panel.
  * A few tickers are reused across different listings (same symbol, disjoint date
    ranges); we keep the union of live ranges per symbol.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

SUPPORTED_TICKERS_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
US_EXCHANGES = frozenset({"NYSE", "NASDAQ", "NYSE ARCA", "AMEX", "BATS"})


@dataclass(frozen=True)
class TickerRecord:
    ticker: str
    exchange: str
    asset_type: str
    currency: str
    start_date: date | None
    end_date: date | None      # None => still listed


def _parse_date(s: str) -> date | None:
    return datetime.fromisoformat(s).date() if s else None


def load_supported_tickers(
    cache_dir: str = "data", refresh: bool = False
) -> list[TickerRecord]:
    """Download (once) and parse Tiingo's supported-ticker list. Cached as CSV under
    `cache_dir` (gitignored); pass ``refresh=True`` to re-download."""
    cache = Path(cache_dir) / "tiingo_supported_tickers.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache.exists():
        import requests
        r = requests.get(SUPPORTED_TICKERS_URL, timeout=120)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        cache.write_bytes(z.read(z.namelist()[0]))
    rows = list(csv.DictReader(cache.open()))
    return [
        TickerRecord(
            ticker=r["ticker"], exchange=r["exchange"], asset_type=r["assetType"],
            currency=r["priceCurrency"],
            start_date=_parse_date(r["startDate"]), end_date=_parse_date(r["endDate"]),
        )
        for r in rows
    ]


def survivorship_free_universe(
    records: list[TickerRecord],
    start: datetime,
    end: datetime,
    *,
    exchanges: frozenset[str] = US_EXCHANGES,
    asset_type: str = "Stock",
    currency: str = "USD",
    include_delisted: bool = True,
) -> list[str]:
    """Tickers that were LIVE at any point in [start, end] — delisted included.

    A record is live in the window when it started on/before ``end`` and had not yet
    delisted before ``start``. Setting ``include_delisted=False`` collapses this back
    to a (biased) survivors-only list, for comparison.

    Note: Tiingo sets ``endDate`` to the last date it has DATA for — today for an
    active name, the delisting date for a dead one — so "still listed at the window
    end" means ``end_date`` reaches ``end`` (there is no empty/None sentinel).
    """
    s, e = start.date(), end.date()
    keep: set[str] = set()
    for r in records:
        if r.exchange not in exchanges or r.asset_type != asset_type or r.currency != currency:
            continue
        if r.start_date is None or r.start_date > e:
            continue
        delisted_before_window = r.end_date is not None and r.end_date < s
        if delisted_before_window:
            continue
        # "Survivor at window end" = data runs through end (active names carry
        # end_date = today; dead ones stop at their delisting date).
        still_live_at_end = r.end_date is None or r.end_date >= e
        if not include_delisted and not still_live_at_end:
            continue  # drop names that delisted within the window (bias, on purpose)
        keep.add(r.ticker)
    return sorted(keep)


def filter_panel_by_liquidity(
    panel: dict,
    min_dollar_volume: float,
    keep: tuple[str, ...] = ("SPY",),
) -> dict:
    """Drop symbols whose median daily dollar volume (close x volume) is below
    `min_dollar_volume` — the crude liquidity screen a real tradable universe has
    but the raw ticker list lacks. Symbols in `keep` (the benchmark) always stay.

    Uses the median over the loaded window, so it is a rough (slightly forward-
    looking) liquidity proxy, not a point-in-time trailing ADV — good enough to
    strip micro-cap shells and pump-and-dumps from a survivorship-free sample.
    """
    out = {}
    for sym, df in panel.items():
        if sym in keep:
            out[sym] = df
            continue
        if df is None or df.empty or "c" not in df or "v" not in df:
            continue
        median_dv = float((df["c"] * df["v"]).median())
        if median_dv >= min_dollar_volume:
            out[sym] = df
    return out


def sample_universe(tickers: list[str], n: int, seed: int = 0) -> list[str]:
    """Deterministic size-``n`` sample (seeded, order-stable) for tractable backtests.
    Uses a hash so the pick is reproducible across processes without RNG state."""
    import hashlib

    def _key(t: str) -> str:
        return hashlib.blake2b(f"{seed}:{t}".encode(), digest_size=8).hexdigest()

    return sorted(sorted(tickers, key=_key)[:n])
