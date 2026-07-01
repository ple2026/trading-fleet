"""Thorp's money management — the fleet treasurer.

Kelly sizing at three levels, always as *quarter*-Kelly and always as a
CAP-REDUCER: Kelly may shrink a position below the hard fixed-fractional / notional
limits, never expand it beyond them. Untested edges get near-zero weight until the
journal has enough samples (small-sample shrinkage), so the fleet only bets big on
edges it has actually measured.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import FleetSignal, SizedOrder

KELLY_FRACTION = 0.25          # quarter-Kelly
MIN_SAMPLES_FOR_FULL = 30      # below this, shrink hard toward zero
MAX_NOTIONAL_PCT = 0.25        # any single position <= 25% of equity
DEFAULT_RISK_PCT = 0.02        # fixed-fractional floor: 2% of equity at risk


@dataclass
class EdgeEstimate:
    """A bot's measured edge for a setup class, read from the journal."""

    win_rate: float          # p
    avg_win_r: float         # b (payoff in R multiples on a win)
    avg_loss_r: float        # typically ~1.0 R by construction of the stop
    n_samples: int


def kelly_fraction(edge: EdgeEstimate) -> float:
    """Full-Kelly fraction f* = p/loss - (1-p)/win, floored at 0.

    Expressed in R-space (payoff ratios), which is how the journal stores outcomes.
    """
    if edge.avg_win_r <= 0 or edge.avg_loss_r <= 0:
        return 0.0
    p = edge.win_rate
    f = p / edge.avg_loss_r - (1 - p) / edge.avg_win_r
    return max(f, 0.0)


def shrink(f: float, n_samples: int) -> float:
    """Small-sample shrinkage: scale Kelly toward 0 until we have enough data.

    Linear ramp to full confidence at MIN_SAMPLES_FOR_FULL trades. An untested
    setup (n=0) gets 0 weight and falls back to the fixed-fractional floor.
    """
    confidence = min(n_samples / MIN_SAMPLES_FOR_FULL, 1.0)
    return f * confidence


def size_order(
    signal: FleetSignal,
    equity_usd: float,
    edge: EdgeEstimate | None = None,
    kelly_frac: float = KELLY_FRACTION,
) -> SizedOrder | None:
    """Size a position. Risk budget = min(fixed-fractional floor, quarter-Kelly).

    Kelly only ever *reduces* risk below the 2% floor; it never levers past it.
    """
    per_share_risk = abs(signal.price - signal.stop_loss)
    if per_share_risk <= 0:
        return None

    risk_pct = DEFAULT_RISK_PCT
    applied_kelly = 0.0
    if edge is not None:
        applied_kelly = shrink(kelly_fraction(edge), edge.n_samples) * kelly_frac
        # Kelly caps the risk budget; it cannot raise it above the floor.
        risk_pct = min(DEFAULT_RISK_PCT, max(applied_kelly, 0.0)) or DEFAULT_RISK_PCT
        if applied_kelly > 0:
            risk_pct = min(DEFAULT_RISK_PCT, applied_kelly)

    risk_usd = equity_usd * risk_pct
    qty = int(risk_usd / per_share_risk)
    if qty <= 0:
        return None

    notional = qty * signal.price
    max_notional = equity_usd * MAX_NOTIONAL_PCT
    if notional > max_notional:
        qty = int(max_notional / signal.price)
        if qty <= 0:
            return None
        notional = qty * signal.price

    return SizedOrder(
        signal=signal,
        qty=qty,
        risk_usd=qty * per_share_risk,
        notional_usd=notional,
        kelly_fraction=applied_kelly,
    )
