"""MACRO thesis scoring (§6).

Druckenmiller's discipline is that a position is held by a *falsifiable thesis*,
not by hope. MACRO already emits, for every entry, a statement ("long TLT: the Fed
is easing") plus an explicit invalidation. This module closes the loop: it lifts
those theses out of the journal, scores each one against what price actually did
over the next 60 and 120 trading days, and rolls the results up by *driver* so the
quarterly post-mortem can say things like "6 of 8 'Fed pivot' theses lost — the
real-rate trigger was too loose."

Scoring is direction-aware: a long thesis scores positive when the name rose, a
short when it fell. The score is the signed forward return, so it measures whether
the thesis played out over the horizon regardless of when the position was closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

SCORE_HORIZONS = (60, 120)   # trading days


@dataclass
class ScoredThesis:
    bot_id: str
    symbol: str
    side: str
    opened_at: datetime
    statement: str
    invalidation: str | None
    driver_key: str | None
    driver_sign: int | None
    score_60d: float | None     # signed forward return over 60 trading days
    score_120d: float | None

    def as_row(self, post_mortem: str | None = None) -> dict[str, Any]:
        """A `theses`-table row for the journal."""
        return {
            "bot_id": self.bot_id, "opened_at": self.opened_at,
            "statement": self.statement, "invalidation": self.invalidation,
            "score_60d": self.score_60d, "score_120d": self.score_120d,
            "post_mortem": post_mortem,
        }


def theses_from_journal(
    signals: list[dict[str, Any]], bot_id: str = "macro"
) -> list[dict[str, Any]]:
    """Lift taken-signal theses out of the journal for `bot_id`."""
    out: list[dict[str, Any]] = []
    for s in signals:
        if s.get("action") != "taken" or s.get("bot_id") != bot_id:
            continue
        if not s.get("thesis"):
            continue
        feats = s.get("features") or {}
        out.append({
            "bot_id": bot_id, "symbol": s["symbol"], "side": s["side"],
            "opened_at": s["at"], "statement": s["thesis"],
            "invalidation": s.get("invalidation"),
            "driver_key": feats.get("driver_key"),
            "driver_sign": feats.get("driver_sign"),
        })
    return out


def _signed_forward_return(
    df: pd.DataFrame, opened_at: datetime, side: str, horizon: int
) -> float | None:
    """Signed forward return of `df` from the first bar >= opened_at, `horizon`
    trading days out. Long -> +return, short -> -return. None if not enough bars."""
    if df is None or "c" not in getattr(df, "columns", ()):
        return None
    close = df["c"]
    pos = close.index.searchsorted(opened_at)   # first bar at/after the decision
    if pos >= len(close) or pos + horizon >= len(close):
        return None
    p0 = float(close.iloc[pos])
    p1 = float(close.iloc[pos + horizon])
    if p0 <= 0:
        return None
    ret = p1 / p0 - 1.0
    return ret if side == "buy" else -ret


def score_theses(
    thesis_rows: list[dict[str, Any]],
    panel: dict[str, pd.DataFrame],
    horizons: tuple[int, int] = SCORE_HORIZONS,
) -> list[ScoredThesis]:
    """Score each thesis on the signed forward return over the two horizons."""
    h60, h120 = horizons
    scored: list[ScoredThesis] = []
    for t in thesis_rows:
        df = panel.get(t["symbol"])
        scored.append(ScoredThesis(
            bot_id=t["bot_id"], symbol=t["symbol"], side=t["side"],
            opened_at=t["opened_at"], statement=t["statement"],
            invalidation=t.get("invalidation"),
            driver_key=t.get("driver_key"), driver_sign=t.get("driver_sign"),
            score_60d=_signed_forward_return(df, t["opened_at"], t["side"], h60),
            score_120d=_signed_forward_return(df, t["opened_at"], t["side"], h120),
        ))
    return scored


@dataclass
class DriverPostMortem:
    driver_key: str
    n: int
    hit_rate_60d: float          # fraction with positive 60d score
    mean_score_60d: float
    mean_score_120d: float
    verdict: str                 # human-readable, with a tightening hint when weak


def post_mortem(scored: list[ScoredThesis], min_n: int = 5) -> list[DriverPostMortem]:
    """Roll theses up by driver and flag the ones that systematically failed.

    A driver whose theses lose on average over the horizon is the signal that its
    trigger is too loose — exactly the §6 post-mortem ("propose tightening")."""
    buckets: dict[str, list[ScoredThesis]] = {}
    for t in scored:
        if t.driver_key is None:
            continue
        buckets.setdefault(t.driver_key, []).append(t)

    out: list[DriverPostMortem] = []
    for key, rows in sorted(buckets.items()):
        s60 = [t.score_60d for t in rows if t.score_60d is not None]
        s120 = [t.score_120d for t in rows if t.score_120d is not None]
        if not s60:
            continue
        hit = float(np.mean([1.0 if x > 0 else 0.0 for x in s60]))
        mean60 = float(np.mean(s60))
        mean120 = float(np.mean(s120)) if s120 else float("nan")
        if len(s60) >= min_n and mean60 < 0:
            verdict = (
                f"{len(s60)} theses, {hit:.0%} hit — driver '{key}' lost "
                f"{mean60:+.1%} avg at 60d; trigger likely too loose, propose tightening."
            )
        else:
            verdict = (
                f"{len(s60)} theses, {hit:.0%} hit, {mean60:+.1%} avg at 60d"
                + ("" if len(s60) >= min_n else " (n<min, inconclusive)")
            )
        out.append(DriverPostMortem(
            driver_key=key, n=len(rows), hit_rate_60d=round(hit, 3),
            mean_score_60d=round(mean60, 4),
            mean_score_120d=round(mean120, 4) if s120 else float("nan"),
            verdict=verdict,
        ))
    return out
