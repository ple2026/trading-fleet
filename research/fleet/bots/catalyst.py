"""CATALYST — a modern, PDT-safe Steve Cohen swing-catalyst trader.

Steve Cohen (SAC/Point72) is the archetype of *catalyst* trading: build an
"information mosaic" around a discrete, dated event (earnings, product launch,
FDA decision, analyst day) by fusing many weak, independent signals — estimate
revisions, options-implied vs realized move, short-interest changes, insider
buying, and the pre-event price drift — into a single conviction score, then
take a directional swing into (and briefly through) the event. No single tile
of the mosaic is decisive; the *edge is the aggregation* of many marginal edges.

This bot is the small-account adaptation of that idea:

  * Horizon is 3-15 trading days (a SWING, not intraday). That is deliberate:
    a $10k cash/PDT-constrained account cannot day-trade freely (< 4 round
    trips / 5 business days), so CATALYST holds across the event and exits on a
    time stop rather than churning. Every position is one round trip, which is
    PDT-safe by construction.
  * It is regime-agnostic: a catalyst re-rates a stock on its own news, so we
    trade in any regime (`may_trade` always True) but HALVE confidence in
    HIGH_VOL / RISK_OFF regimes, where event reactions are noisier and gap risk
    is worse.

The mosaic
----------
`_mosaic_score` returns a composite in [-1, 1] built from whatever features are
available at decision time. Each component is itself clamped to [-1, 1], and the
composite is the simple MEAN of the components that are actually present (a
missing data plane is skipped, never imputed to zero-with-weight). Components:

  (a) estimate-revision proxy — sign/magnitude of 20-day price momentum. When
      real analyst-revision fundamentals are absent (the usual offline case),
      trailing momentum is the market's own running vote on estimates.
  (b) implied-vs-realized move — with no options plane, we proxy an
      "interesting event" by realized-vol EXPANSION: the percentile of current
      ATR within its trailing window. High ATR percentile => the tape already
      senses something, which raises |mosaic| without biasing its sign.
  (c) short-interest delta — skipped unless `ctx.fundamentals[sym]` carries it.
  (d) insider buying — skipped unless `ctx.fundamentals[sym]` carries it.
  (e) pre-event drift — the 5-day return leading into the event window; smart
      money often positions ahead of the print, so drift is directional.
  (f) institutional conviction — a signed 13F smart-money score (Druckenmiller /
      Cohen newly bought vs exited), skipped unless `ctx.fundamentals[sym]` carries
      it. Lagged and weak by construction (see `smart_money.py`).

NOTE ON WEIGHTS: the composite is an *unweighted mean* of present components on
purpose. These are PLACEHOLDER weights. The improvement engine's Tier-B logistic
re-fit is expected to re-estimate per-component weights weekly from realized
event outcomes; until then an equal-weight mosaic is the honest prior. Every
component value is written into `FleetSignal.features` precisely so that re-fit
can replay them.

Offline / calendar=None fallback
--------------------------------
In OFFLINE smoke runs `ctx.calendar` is None (no event data plane). Rather than
emit nothing, CATALYST synthesizes PSEUDO-CATALYSTS from price alone: any symbol
showing a strong, low-noise pre-breakout drift is treated as if a dated catalyst
sat `days_before` bars in the future. This keeps the bot producing and
backtestable on synthetic prices; such signals are flagged `event="proxy"` in
their features so they are never confused with real calendar-driven trades.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..bot import Bot, MarketContext, ParamSpec
from ..indicators import atr
from ..types import FleetSignal, ManagementAction, RegimeTag


class Catalyst(Bot):
    """Swing-catalyst bot. See module docstring for the mosaic construction."""

    id = "catalyst"
    # Declared for completeness; `may_trade` is overridden to ignore it.
    favorable_regimes = (
        RegimeTag.TRENDING_UP,
        RegimeTag.TRENDING_DOWN,
        RegimeTag.RANGE,
        RegimeTag.CHOP,
        RegimeTag.HIGH_VOL,
        RegimeTag.RISK_OFF,
    )

    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__(params)
        # MarketContext carries no entry dates, so the bot remembers when it
        # produced each entry signal to age the position for the time stop.
        # Maps symbol -> (entry datetime, stop price) recorded in `scan`.
        self._entered: dict[str, datetime] = {}
        self._stops: dict[str, float] = {}
        self._sides: dict[str, str] = {}

    # ------------------------------------------------------------------ params
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="mosaic_threshold", default=0.5, lo=0.2, hi=0.9),
            ParamSpec(name="days_before", default=5, lo=2, hi=10, integer=True),
            ParamSpec(name="hold_days", default=10, lo=3, hi=15, integer=True),
            ParamSpec(name="stop_pct", default=8.0, lo=4.0, hi=15.0),
        ]

    # ------------------------------------------------------------------- gating
    def may_trade(self, regime) -> bool:  # noqa: ANN001 - RegimeState, per Bot ABC
        """Catalyst trading is regime-agnostic: always allowed.

        The regime still matters — HIGH_VOL / RISK_OFF halve confidence inside
        `scan` — but it never *gates* participation, because a stock-specific
        catalyst can re-rate a name against any macro backdrop.
        """
        return True

    # ------------------------------------------------------------- mosaic parts
    @staticmethod
    def _clip1(x: float) -> float:
        """Clamp a component into [-1, 1], mapping non-finite values to 0."""
        if x is None or not np.isfinite(x):
            return 0.0
        return float(np.clip(x, -1.0, 1.0))

    def _estimate_revision(self, close: pd.Series) -> float | None:
        """(a) 20-day momentum as a stand-in for the analyst-revision trend."""
        if len(close) < 21:
            return None
        past = float(close.iloc[-21])
        if past <= 0:
            return None
        ret_20 = float(close.iloc[-1]) / past - 1.0
        # Scale so a ~15% 20-day move saturates the component.
        return self._clip1(ret_20 / 0.15)

    def _implied_vs_realized(self, df: pd.DataFrame) -> float | None:
        """(b) ATR percentile as a realized-vol-expansion proxy (magnitude only).

        Options data would give implied-vs-realized directly; absent it, an
        elevated ATR percentile flags that the tape senses an event. This is a
        *magnitude* signal (always >= 0): it inflates conviction without voting
        on direction, so it is returned as a non-negative interest score.
        """
        if len(df) < 30:
            return None
        atr14 = atr(df["h"], df["l"], df["c"], 14).dropna()
        if len(atr14) < 20:
            return None
        window = atr14.tail(60)
        cur = float(atr14.iloc[-1])
        pct = float((window < cur).mean())  # 0..1 percentile rank
        # Center so median vol -> 0 interest, top-decile vol -> ~+1.
        return self._clip1((pct - 0.5) * 2.0)

    def _short_interest_delta(self, fund: dict[str, Any] | None) -> float | None:
        """(c) Short-interest change, if the fundamentals plane supplies it.

        A rising short interest into a positive catalyst is squeeze fuel
        (bullish); into a negative one it confirms. We read a signed
        `short_interest_delta` (fraction of float) and scale it. Skipped if
        absent — which is the norm offline.
        """
        if not fund:
            return None
        si = fund.get("short_interest_delta")
        if si is None:
            return None
        return self._clip1(float(si) / 0.05)

    def _insider_buying(self, fund: dict[str, Any] | None) -> float | None:
        """(d) Net insider buying, if supplied by the fundamentals plane.

        Read a signed `insider_net_usd` (positive = net buys). Insider buying is
        one of the strongest single mosaic tiles. Skipped if absent.
        """
        if not fund:
            return None
        ins = fund.get("insider_net_usd")
        if ins is None:
            return None
        # $1M net buying saturates the component.
        return self._clip1(float(ins) / 1_000_000.0)

    def _pre_event_drift(self, close: pd.Series) -> float | None:
        """(e) 5-day return leading into the event window (directional)."""
        if len(close) < 6:
            return None
        past = float(close.iloc[-6])
        if past <= 0:
            return None
        ret_5 = float(close.iloc[-1]) / past - 1.0
        # A ~8% 5-day drift saturates the component.
        return self._clip1(ret_5 / 0.08)

    def _institutional_conviction(self, fund: dict[str, Any] | None) -> float | None:
        """(f) Smart-money 13F conviction, if the fundamentals plane supplies it.

        Reads a signed `institutional_conviction` in [-1, 1] built from Druckenmiller
        / Cohen 13F quarter-over-quarter changes (see `smart_money.py`): a name they
        newly bought scores +1, one they exited -1. It is deliberately LAGGED and
        weak (13F is quarter-end, filed up to 45 days late), so it is just one tile
        of the mosaic — the weekly re-fit decides how much it is worth. Skipped when
        absent (the usual offline case)."""
        if not fund:
            return None
        conv = fund.get("institutional_conviction")
        if conv is None:
            return None
        return self._clip1(float(conv))

    def _mosaic_score(
        self, df: pd.DataFrame, fund: dict[str, Any] | None
    ) -> tuple[float, dict[str, float]]:
        """Fuse present components into a composite in [-1, 1].

        Returns (composite, components) where `components` holds only the tiles
        that were actually available. The composite is the equal-weight MEAN of
        those tiles (placeholder weights — see module docstring).
        """
        close = df["c"]
        raw: dict[str, float | None] = {
            "estimate_revision": self._estimate_revision(close),
            "implied_vs_realized": self._implied_vs_realized(df),
            "short_interest_delta": self._short_interest_delta(fund),
            "insider_buying": self._insider_buying(fund),
            "pre_event_drift": self._pre_event_drift(close),
            "institutional_conviction": self._institutional_conviction(fund),
        }
        components = {k: v for k, v in raw.items() if v is not None}
        if not components:
            return 0.0, {}
        composite = float(np.mean(list(components.values())))
        return self._clip1(composite), components

    # ------------------------------------------------------------- candidates
    def _calendar_candidates(
        self, ctx: MarketContext, days_before: int
    ) -> list[dict[str, Any]]:
        """Events within `days_before` trading-ish days ahead of ctx.now.

        Uses a calendar-day window of `days_before` as a simple, dependency-free
        approximation of the trading-day horizon (good enough for a pre-event
        entry gate). Past events are ignored.
        """
        out: list[dict[str, Any]] = []
        if not ctx.calendar:
            return out
        horizon = ctx.now + timedelta(days=days_before)
        for ev in ctx.calendar:
            sym = ev.get("symbol")
            when = ev.get("date")
            if sym is None or when is None:
                continue
            if ctx.now <= when <= horizon:
                out.append(
                    {"symbol": sym, "date": when, "type": ev.get("type", "event")}
                )
        return out

    def _proxy_candidates(
        self, ctx: MarketContext, days_before: int
    ) -> list[dict[str, Any]]:
        """OFFLINE fallback: manufacture pseudo-catalysts from price alone.

        A catalyst is a discrete EVENT, not a persistent state — so we only treat a
        symbol as having a pseudo-catalyst when today prints a fresh 20-day
        extreme AND arrived on meaningful 5-day drift. That makes proxy candidates
        rare and event-like (instead of firing on every drifting name every day),
        which is what a real earnings/FDA calendar would deliver. Flagged
        event="proxy" so downstream never mistakes it for a real calendar entry.
        """
        out: list[dict[str, Any]] = []
        pseudo_date = ctx.now + timedelta(days=days_before)
        for sym, df in ctx.prices.items():
            if df is None or len(df) < 21 or "c" not in df:
                continue
            close = df["c"]
            past = float(close.iloc[-6])
            if past <= 0:
                continue
            last = float(close.iloc[-1])
            drift = abs(last / past - 1.0)
            window = close.iloc[-21:-1]  # prior 20 closes, excluding today
            fresh_event = last > float(window.max()) or last < float(window.min())
            if fresh_event and drift >= 0.03:
                out.append({"symbol": sym, "date": pseudo_date, "type": "proxy"})
        return out

    # ------------------------------------------------------------------- scan
    def scan(self, ctx: MarketContext) -> list[FleetSignal]:
        threshold = float(self.params["mosaic_threshold"])
        days_before = int(self.params["days_before"])
        stop_pct = float(self.params["stop_pct"])

        risk_regime = ctx.regime.label in (RegimeTag.HIGH_VOL, RegimeTag.RISK_OFF)

        # Real calendar events take priority; fall back to price proxies offline.
        candidates = self._calendar_candidates(ctx, days_before)
        if not candidates:
            candidates = self._proxy_candidates(ctx, days_before)

        signals: list[FleetSignal] = []
        seen: set[str] = set()
        for cand in candidates:
            sym = cand["symbol"]
            if sym in seen or sym in ctx.open_symbols:
                continue
            df = ctx.prices.get(sym)
            if df is None or df.empty or "c" not in df:
                continue

            fund = ctx.fundamentals.get(sym) if ctx.fundamentals else None
            mosaic, components = self._mosaic_score(df, fund)
            if abs(mosaic) < threshold:
                continue
            seen.add(sym)

            price = float(df["c"].iloc[-1])
            if not np.isfinite(price) or price <= 0:
                continue
            side = "buy" if mosaic > 0 else "sell"

            # Stop from stop_pct; take-profit sized ~2:1 over the hold.
            if side == "buy":
                stop = price * (1.0 - stop_pct / 100.0)
                take_profit = price * (1.0 + 2.0 * stop_pct / 100.0)
            else:
                stop = price * (1.0 + stop_pct / 100.0)
                take_profit = price * (1.0 - 2.0 * stop_pct / 100.0)

            confidence = min(abs(mosaic), 1.0)
            if risk_regime:
                confidence *= 0.5

            is_proxy = cand.get("type") == "proxy"
            event_label = (
                "proxy"
                if is_proxy
                else {
                    "symbol": sym,
                    "date": cand["date"],
                    "type": cand.get("type", "event"),
                }
            )
            features: dict[str, Any] = {
                **components,
                "mosaic": mosaic,
                "event": event_label,
                "regime": ctx.regime.label.value,
                "risk_regime_halved": risk_regime,
            }

            cat_name = "price-proxy pre-breakout" if is_proxy else cand.get(
                "type", "event"
            )
            direction = "long into" if side == "buy" else "short into"
            thesis = (
                f"{cat_name} catalyst on {sym}: mosaic={mosaic:+.2f} "
                f"({len(components)} tiles) -> {direction} the event"
            )

            signals.append(
                FleetSignal(
                    bot_id=self.id,
                    at=ctx.now,
                    symbol=sym,
                    side=side,
                    price=price,
                    stop_loss=stop,
                    take_profit=take_profit,
                    confidence=confidence,
                    thesis=thesis,
                    features=features,
                )
            )
            # Remember entry for the time-based exit and adverse-gap guard.
            self._entered[sym] = ctx.now
            self._stops[sym] = stop
            self._sides[sym] = side

        return signals

    # ----------------------------------------------------------------- manage
    def manage(self, ctx: MarketContext) -> list[ManagementAction]:
        """Exit on TIME (swing horizon) or a hard adverse gap past the stop.

        CATALYST is a time-stop strategy: the catalyst plays out over the hold
        window, after which edge decays, so any position that has aged
        `hold_days` calendar days is closed. As a secondary guard we also exit
        immediately if price has gapped beyond the stop we recorded at entry.
        """
        hold_days = int(self.params["hold_days"])
        actions: list[ManagementAction] = []

        def _forget(sym: str) -> None:
            self._entered.pop(sym, None)
            self._stops.pop(sym, None)
            self._sides.pop(sym, None)

        for sym in list(ctx.open_symbols):
            entry = self._entered.get(sym)

            # Hard adverse-gap guard: close if the close has gapped past the
            # stop we recorded at entry. Direction depends on the tracked side:
            # a long is breached when price <= stop, a short when price >= stop.
            # The engine already enforces intrabar stops, so this is a
            # belt-and-suspenders guard on gap-through closes.
            stop = self._stops.get(sym)
            side = self._sides.get(sym)
            df = ctx.prices.get(sym)
            if stop is not None and side is not None and df is not None \
                    and not df.empty and "c" in df:
                last = float(df["c"].iloc[-1])
                breached = (side == "buy" and last <= stop) or (
                    side == "sell" and last >= stop
                )
                if breached:
                    actions.append(
                        ManagementAction(kind="exit", symbol=sym, reason="stop_gap")
                    )
                    _forget(sym)
                    continue

            # Time exit: aged to the swing horizon.
            if entry is not None and (ctx.now - entry).days >= hold_days:
                actions.append(
                    ManagementAction(kind="exit", symbol=sym, reason="time_exit")
                )
                _forget(sym)

        return actions
