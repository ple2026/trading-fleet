"""Shared types for the fleet. Mirrors the TS `lib/types.ts` on the execution side.

Keep this module dependency-free (stdlib only) so every layer can import it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

BotId = Literal["breakout", "arb", "catalyst", "macro"]
Side = Literal["buy", "sell"]


class RegimeTag(str, Enum):
    """Fleet-wide regime labels produced by the regime service (MACRO bot owns it)."""

    TRENDING_UP = "trending-up"
    TRENDING_DOWN = "trending-down"
    RANGE = "range"
    CHOP = "chop"
    HIGH_VOL = "high-vol"
    RISK_OFF = "risk-off"


@dataclass(frozen=True)
class Bar:
    t: datetime
    o: float
    h: float
    l: float
    c: float
    v: float


@dataclass
class RegimeState:
    """A point-in-time snapshot of the macro state vector plus a coarse label."""

    at: datetime
    label: RegimeTag
    vector: dict[str, float] = field(default_factory=dict)
    conviction: float = 0.0  # 0..1, how far from the neutral centroid


@dataclass
class FleetSignal:
    """A trade candidate. Extends the base Signal with journal-critical metadata.

    `features` is the *complete* set of inputs the bot saw at decision time — it is
    what the improvement engine replays. Never omit an input the rule used.
    """

    bot_id: BotId
    at: datetime
    symbol: str
    side: Side
    price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0..1
    thesis: str
    features: dict[str, Any] = field(default_factory=dict)
    # Falsification criteria (MACRO): the explicit condition that would prove the
    # thesis wrong. Journaled to the `theses` table and scored at 60/120 days.
    invalidation: str | None = None
    # For market-neutral bots (ARB): the paired/hedge leg, if any.
    hedge_symbol: str | None = None
    hedge_ratio: float | None = None


@dataclass
class SizedOrder:
    signal: FleetSignal
    qty: float
    risk_usd: float
    notional_usd: float
    kelly_fraction: float  # the shrunk quarter-Kelly weight actually applied


ManagementActionKind = Literal[
    "hold", "exit", "trim", "add", "move_stop"
]


@dataclass
class ManagementAction:
    kind: ManagementActionKind
    symbol: str
    reason: str
    qty: float | None = None
    new_stop: float | None = None
