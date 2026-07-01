"""The Bot contract. Every strategy implements this so the meta-layer, backtest
engine, and improvement engine can treat all four identically.

A bot is pure strategy: it turns market context into signals and management
actions, and it declares which parameters the walk-forward tuner may touch. It
does NOT size positions (the Kelly kernel does), submit orders (the executor
does), or persist anything (the journal does).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from .types import BotId, FleetSignal, ManagementAction, RegimeState, RegimeTag


@dataclass
class ParamSpec:
    """One tunable parameter and the bounds the Tier-A walk-forward tuner may explore.

    Bounds are hard: the tuner searches only within [lo, hi] and never shifts a
    value more than `max_step_pct` of its range in a single weekly re-fit.
    """

    name: str
    default: float
    lo: float
    hi: float
    max_step_pct: float = 0.10
    integer: bool = False


@dataclass
class MarketContext:
    """Everything a bot needs to make decisions at a point in time.

    `prices` maps symbol -> OHLCV DataFrame (index = datetime, columns o/h/l/c/v),
    already sliced to be point-in-time correct (no bars after `now`). Optional
    panels are populated only when the relevant data plane is configured.
    """

    now: datetime
    prices: dict[str, pd.DataFrame]
    regime: RegimeState
    equity_usd: float
    open_symbols: set[str]
    fundamentals: dict[str, Any] | None = None
    calendar: list[dict[str, Any]] | None = None
    macro: dict[str, float] | None = None


class Bot(ABC):
    id: BotId
    favorable_regimes: tuple[RegimeTag, ...]
    # Max positions this bot may hold at once. A $10k book cannot hold unlimited
    # names; concentrated bots (MACRO) override this down. The backtest engine and
    # live executor both enforce it.
    max_concurrent: int = 8

    def __init__(self, params: dict[str, float] | None = None) -> None:
        self.params = {p.name: p.default for p in self.param_space()}
        if params:
            self.params.update(params)

    @abstractmethod
    def param_space(self) -> list[ParamSpec]:
        """Declare tunable params + bounds for the walk-forward optimizer."""

    @abstractmethod
    def scan(self, ctx: MarketContext) -> list[FleetSignal]:
        """Find new trade candidates given the current context."""

    @abstractmethod
    def manage(self, ctx: MarketContext) -> list[ManagementAction]:
        """Decide what to do with open positions (exits, trims, adds, stops)."""

    def may_trade(self, regime: RegimeState) -> bool:
        """Tier-C gate: is the current regime one this bot is allowed to trade?

        Default respects `favorable_regimes`; bots can override for nuance
        (e.g. CATALYST trades any regime but halves size in high-vol).
        """
        return regime.label in self.favorable_regimes
