"""MACRO — a modern Stanley Druckenmiller, expressed through liquid ETFs.

Druckenmiller's edge was never diversification. It was three ideas applied with
uncommon discipline:

  1. **Liquidity leads.** Central-bank policy and the flow of money — not
     earnings — set the tape. Read the direction of liquidity (rates, the dollar,
     the curve) and you know which asset class *wants* to go up.
  2. **Top-down, one big bet at a time.** Find the one or two themes with the
     clearest liquidity tailwind and express them concentrated. This bot holds
     1–3 positions, never a sprawling book — the opposite of the fleet's
     breadth-seeking bots.
  3. **Size to conviction.** "It takes courage to be a pig." When the macro
     picture and price action agree loudly, lean in — bigger confidence, wider
     profit target. When they merely whisper, pass.

The bot is ALL-WEATHER: it trades every regime because it *rotates what it holds*
rather than sitting out. In a risk-off tape it is long duration and gold (or
short equities); in a reflation it is long commodities and small caps. `may_trade`
therefore always returns True; the regime gate belongs to the single-theme bots.

Data planes
-----------
When a real macro panel is wired in (`ctx.macro`, a dict of FRED-style series such
as ``yield_curve_slope``, ``real_rate_trend``, ``fed_stance``, ``credit_spread``,
``usd_trend``, ``oil_trend``, ``pmi``, ``vix``) the playbook reads it directly.
OFFLINE — in backtests and this repo's synthetic harness — ``ctx.macro`` is None,
so the bot DERIVES a proxy macro-state vector purely from the trend of the liquid
ETFs it can already see in ``ctx.prices`` (TLT for duration, UUP for the dollar,
GLD for gold, SPY for equities, DBC for commodities, IWM-vs-SPY for risk
appetite). The playbook, conviction and management logic are identical on both
paths; only the source of the state vector changes.

Future hook
-----------
An LLM FOMC/minutes-reading loop is planned. It will NOT trade. Its only job is to
nudge the *priors* in the state vector (e.g. lean ``fed_stance`` dovish after a
soft statement) before the deterministic playbook below runs. The playbook stays
fully auditable — the LLM adjusts inputs, never picks trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..bot import Bot, MarketContext, ParamSpec
from ..types import (
    FleetSignal,
    ManagementAction,
    RegimeState,
    RegimeTag,
)

# Trend magnitude that counts as "full strength" (1.0). A 10% ETF move over the
# lookback is treated as a maximally strong macro signal.
_TREND_SCALE = 0.10


def _ret(df: pd.DataFrame, lookback: int) -> float:
    """Trailing `lookback`-bar simple return of the close, or 0.0 if too short."""
    if df is None or "c" not in getattr(df, "columns", ()):  # missing series
        return 0.0
    c = df["c"].dropna()
    if len(c) < lookback + 1:
        return 0.0
    prev = float(c.iloc[-lookback - 1])
    if prev == 0.0:
        return 0.0
    return float(c.iloc[-1]) / prev - 1.0


def _strength(x: float, scale: float = _TREND_SCALE) -> float:
    """Map a signed trend to an unsigned magnitude in [0, 1]."""
    return float(np.clip(abs(x) / scale, 0.0, 1.0))


class Macro(Bot):
    """Concentrated top-down macro rotation across liquid ETFs."""

    id = "macro"
    # All-weather: the bot rotates rather than sits out, so every label is fine.
    favorable_regimes = tuple(RegimeTag)
    # Druckenmiller is concentrated — a handful of big expressions, never a basket.
    max_concurrent = 3

    # The expression universe — the only symbols this bot will ever hold, grouped
    # by the macro theme they express. Only symbols actually present in
    # ``ctx.prices`` are considered at decision time.
    EXPRESSIONS = {
        "equities": ["SPY", "QQQ", "IWM"],
        "duration": ["TLT", "IEF"],
        "metals": ["GLD", "SLV"],
        "dollar": ["UUP"],
        "commodities": ["DBC", "USO"],
        "foreign": ["EEM", "EWJ"],
    }

    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__(params)
        # Entry rationale, keyed by symbol — the human-readable thesis we can show
        # in the journal and, crucially, later falsify.
        self._thesis: dict[str, str] = {}
        # Machine-readable driver behind each open position: (state_key, sign).
        # `sign` is +1 if we entered because that trend was positive, -1 if
        # negative. `manage` recomputes the trend and exits when it flips.
        self._drivers: dict[str, tuple[str, int]] = {}

    # ------------------------------------------------------------------ params
    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec("conviction_floor", default=0.4, lo=0.2, hi=0.7),
            ParamSpec("max_positions", default=3, lo=1, hi=5, integer=True),
            ParamSpec("stop_pct", default=12.0, lo=6.0, hi=20.0),
            ParamSpec("trend_lookback", default=90, lo=40, hi=200, integer=True),
        ]

    # All-weather override: macro trades in every regime.
    def may_trade(self, regime: RegimeState) -> bool:  # noqa: ARG002
        return True

    # ------------------------------------------------------------- state model
    def _state_vector(self, ctx: MarketContext, lookback: int) -> dict[str, float]:
        """Build the macro state vector.

        Base layer is always the ETF-derived proxy so the bot runs fully offline.
        If ``ctx.macro`` is present, its FRED-style series *nudge* the proxy trends
        (priors), and the same downstream playbook consumes the result.
        """
        px = ctx.prices

        equity_trend = _ret(px.get("SPY"), lookback)
        duration_trend = _ret(px.get("TLT"), lookback)
        gold_trend = _ret(px.get("GLD"), lookback)
        dollar_trend = _ret(px.get("UUP"), lookback)
        commodity_trend = _ret(px.get("DBC"), lookback)
        # Risk appetite: small-caps leading large-caps => risk-on breadth.
        risk_appetite = _ret(px.get("IWM"), lookback) - equity_trend

        mode = "proxy"
        if ctx.macro:
            mode = "macro"
            m = ctx.macro
            # Falling real rates / dovish Fed / steepening curve => bonds bid.
            duration_trend += (
                0.5 * m.get("fed_stance", 0.0)
                - 0.5 * m.get("real_rate_trend", 0.0)
                + 0.25 * m.get("yield_curve_slope", 0.0)
            )
            dollar_trend += m.get("usd_trend", 0.0)
            commodity_trend += 0.5 * m.get("oil_trend", 0.0)
            # Gold hates positive real rates.
            gold_trend += -0.5 * m.get("real_rate_trend", 0.0)
            # Expansion (PMI>50) helps equities; a hot VIX hurts.
            equity_trend += (m.get("pmi", 50.0) - 50.0) / 100.0 - (
                m.get("vix", 20.0) - 20.0
            ) / 100.0
            # Risk appetite fades on wide spreads / high vol.
            risk_appetite += (
                -(m.get("vix", 20.0) - 20.0) / 100.0
                - 0.5 * m.get("credit_spread", 0.0)
                + (m.get("pmi", 50.0) - 50.0) / 200.0
            )

        return {
            "mode": mode,
            "equity_trend": float(equity_trend),
            "duration_trend": float(duration_trend),
            "gold_trend": float(gold_trend),
            "dollar_trend": float(dollar_trend),
            "commodity_trend": float(commodity_trend),
            "risk_appetite": float(risk_appetite),
        }

    def _present(self, prefs: list[str], px: dict[str, pd.DataFrame]) -> str | None:
        """First symbol in `prefs` that we actually have prices for."""
        for s in prefs:
            df = px.get(s)
            if df is not None and "c" in getattr(df, "columns", ()) and len(df):
                return s
        return None

    # --------------------------------------------------------------- playbook
    def _playbook(
        self, state: dict[str, float], px: dict[str, pd.DataFrame]
    ) -> list[dict]:
        """Explicit, auditable state->expression rules.

        Each firing rule yields a candidate dict:
          {symbol, side, strength(0..1), rule, driver_key, driver_sign}
        `driver_key`/`driver_sign` record which macro trend justified the trade so
        management can later falsify it.
        """
        eq = state["equity_trend"]
        dur = state["duration_trend"]
        gold = state["gold_trend"]
        usd = state["dollar_trend"]
        cmdty = state["commodity_trend"]
        risk = state["risk_appetite"]

        out: list[dict] = []

        def fire(prefs, side, strength, rule, driver_key, driver_sign):
            sym = self._present(prefs, px)
            if sym is None or strength <= 0.0:
                return
            out.append(
                {
                    "symbol": sym,
                    "side": side,
                    "strength": float(strength),
                    "rule": rule,
                    "driver_key": driver_key,
                    "driver_sign": int(driver_sign),
                }
            )

        # 1) Trend + breadth both up => own beta (prefer QQQ, then SPY).
        if eq > 0 and risk > 0:
            fire(
                ["QQQ", "SPY"], "buy",
                _strength(eq) * 0.5 + _strength(risk, 0.05) * 0.5,
                "equity_up_broad_breadth", "equity_trend", +1,
            )

        # 2) Bonds bid (easing / curve / real-rate signal) => long duration.
        if dur > 0:
            fire(["TLT", "IEF"], "buy", _strength(dur),
                 "duration_up_easing", "duration_trend", +1)

        # 3) Gold rallying while the dollar weakens => classic long gold.
        if gold > 0 and usd < 0:
            fire(["GLD", "SLV"], "buy", _strength(gold),
                 "gold_up_weak_dollar", "gold_trend", +1)

        # 4) Dollar in an uptrend => long the dollar.
        if usd > 0:
            fire(["UUP"], "buy", _strength(usd),
                 "dollar_up", "dollar_trend", +1)

        # 5) Commodities trending => long broad commodities.
        if cmdty > 0:
            fire(["DBC", "USO"], "buy", _strength(cmdty),
                 "commodity_up", "commodity_trend", +1)

        # 6) Risk-off equity tape => defensives long (duration + gold) AND the
        #    option to press the downside via a short in equity beta.
        if eq < 0:
            s = _strength(eq)
            fire(["TLT", "IEF"], "buy", s,
                 "risk_off_defensive_duration", "equity_trend", -1)
            fire(["GLD", "SLV"], "buy", s,
                 "risk_off_defensive_gold", "equity_trend", -1)
            fire(["SPY", "QQQ"], "sell", s,
                 "risk_off_short_equity", "equity_trend", -1)

        return out

    # ------------------------------------------------------------------- scan
    def scan(self, ctx: MarketContext) -> list[FleetSignal]:
        floor = float(self.params["conviction_floor"])
        max_pos = int(round(self.params["max_positions"]))
        stop_pct = float(self.params["stop_pct"])
        lookback = int(round(self.params["trend_lookback"]))

        state = self._state_vector(ctx, lookback)
        fired = self._playbook(state, ctx.prices)
        if not fired:
            return []

        # Collapse rules onto (symbol, side). Multiple rules on the same
        # expression are AGREEMENT — the essence of conviction sizing.
        grouped: dict[tuple[str, str], list[dict]] = {}
        for c in fired:
            grouped.setdefault((c["symbol"], c["side"]), []).append(c)

        scored: list[dict] = []
        for (sym, side), rules in grouped.items():
            base = max(r["strength"] for r in rules)
            agreement = len(rules)
            # Agreement bonus: each extra rule pointing the same way adds courage.
            conviction = float(
                np.clip(base + 0.10 * (agreement - 1), 0.0, 1.0)
            )
            # Keep the dominant driver (strongest rule) for falsification.
            dom = max(rules, key=lambda r: r["strength"])
            scored.append(
                {
                    "symbol": sym,
                    "side": side,
                    "conviction": conviction,
                    "agreement": agreement,
                    "rules": [r["rule"] for r in rules],
                    "driver_key": dom["driver_key"],
                    "driver_sign": dom["driver_sign"],
                }
            )

        # If a symbol is proposed both long and short, the loudest voice wins.
        best_per_symbol: dict[str, dict] = {}
        for cand in scored:
            prev = best_per_symbol.get(cand["symbol"])
            if prev is None or cand["conviction"] > prev["conviction"]:
                best_per_symbol[cand["symbol"]] = cand

        # Concentrate: keep the highest-conviction names above the floor.
        ranked = sorted(
            best_per_symbol.values(), key=lambda c: c["conviction"], reverse=True
        )
        ranked = [c for c in ranked if c["conviction"] >= floor][:max_pos]

        signals: list[FleetSignal] = []
        for cand in ranked:
            sym = cand["symbol"]
            side = cand["side"]
            conv = cand["conviction"]
            df = ctx.prices[sym]
            price = float(df["c"].dropna().iloc[-1])

            # Conviction-scaled targets. Wider profit target when we lean in.
            tp_move = conv * 0.30
            if side == "buy":
                stop_loss = price * (1.0 - stop_pct / 100.0)
                take_profit = price * (1.0 + tp_move)
            else:  # short
                stop_loss = price * (1.0 + stop_pct / 100.0)
                take_profit = price * (1.0 - tp_move)

            statement, invalidation = self._make_thesis(sym, side, cand, state)

            signals.append(
                FleetSignal(
                    bot_id=self.id,
                    at=ctx.now,
                    symbol=sym,
                    side=side,
                    price=price,
                    stop_loss=float(stop_loss),
                    take_profit=float(take_profit),
                    confidence=conv,  # sizing follows conviction
                    thesis=statement,
                    invalidation=invalidation,
                    features={
                        "state_vector": {
                            k: v for k, v in state.items() if k != "mode"
                        },
                        "mode": state["mode"],
                        "rules_fired": cand["rules"],
                        "agreement": cand["agreement"],
                        "conviction": conv,
                        "driver_key": cand["driver_key"],
                        "driver_sign": cand["driver_sign"],
                        "lookback": lookback,
                    },
                )
            )
            # Remember why we're in, for both display and falsification.
            self._thesis[sym] = statement
            self._drivers[sym] = (cand["driver_key"], cand["driver_sign"])

        return signals

    def _make_thesis(
        self, sym: str, side: str, cand: dict, state: dict[str, float]
    ) -> tuple[str, str]:
        """Return (statement, invalidation) — the falsifiable thesis and the exact
        condition that would prove it wrong (scored later at 60/120 days)."""
        key = cand["driver_key"]
        sign = cand["driver_sign"]
        val = state.get(key, 0.0)
        direction = "positive" if sign > 0 else "negative"
        verb = "Long" if side == "buy" else "Short"
        statement = (
            f"{verb} {sym}: {key} is {direction} ({val:+.2%}) with "
            f"{cand['agreement']} rule(s) in agreement "
            f"[{', '.join(cand['rules'])}]."
        )
        invalidation = f"{key} crosses back through zero against the position."
        return statement, invalidation

    # ----------------------------------------------------------------- manage
    def manage(self, ctx: MarketContext) -> list[ManagementAction]:
        """Exit when the macro trend that drove the position flips sign.

        Druckenmiller's discipline: the thesis, not the P&L, holds the position.
        When the driving liquidity/trend signal crosses zero against us, we're
        wrong — cut it.
        """
        lookback = int(round(self.params["trend_lookback"]))
        state = self._state_vector(ctx, lookback)

        actions: list[ManagementAction] = []
        for sym in ctx.open_symbols:
            driver = self._drivers.get(sym)
            if driver is None:
                continue  # not one of ours (or restarted process) — leave it
            key, sign = driver
            current = state.get(key, 0.0)
            # `sign` is the direction we needed. If current * sign <= 0 the trend
            # has flipped or gone flat against us.
            if current * sign <= 0.0:
                actions.append(
                    ManagementAction(
                        kind="exit",
                        symbol=sym,
                        reason="thesis_invalidated",
                    )
                )
                self._thesis.pop(sym, None)
                self._drivers.pop(sym, None)

        return actions
