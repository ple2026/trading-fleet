"""BREAKOUT — a modern fusion of Jesse Livermore and William O'Neil (CAN SLIM).

Livermore taught that real money is made buying strength at the *pivotal point*:
the moment a stock clears prior resistance and moves into a "line of least
resistance." O'Neil formalized the same instinct into CAN SLIM — buy the market
*leaders* (high relative strength), demand *heavy volume* on the breakout to
prove institutional demand, insist the company shows real *earnings acceleration*,
and only ever operate in a confirmed up-trend.

This bot fuses the two:

  * Entry (`scan`): a new N-day Donchian high (Livermore's pivot) confirmed by a
    volume surge (O'Neil's supply/demand), inside an EMA up-trend, filtered to
    cross-sectional RS leaders, and — when fundamentals are available — gated on
    quarterly EPS growth. Risk is capped with a fixed percent stop and a 3:1
    reward target.
  * Exit (`manage`): Livermore's "abnormal reaction" — either the trend breaks
    (three straight closes back below the trend EMA) or the move goes parabolic
    into a climax top (far above trend with an overbought RSI). Both are cues to
    ring the register rather than give profit back.

The bot only sizes decisions; the Kelly kernel sizes positions and the executor
places orders. It trades only when the regime service tags the tape TRENDING_UP.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..bot import Bot, MarketContext, ParamSpec
from ..indicators import clenow_momentum, donchian, ema, rsi, sma
from ..types import FleetSignal, ManagementAction, RegimeTag

# Trailing window (~6 months of trading) for O'Neil relative-strength ranking.
_RS_LOOKBACK = 126
# Window for the 52-week-high proximity feature O'Neil watches for base breakouts.
_52W = 252


class Breakout(Bot):
    """Livermore pivot + O'Neil CAN SLIM leadership breakouts.

    Buys market leaders punching through a pivotal high on heavy volume inside a
    confirmed up-trend; exits on an abnormal reaction (trend break or climax top).
    """

    id = "breakout"
    favorable_regimes = (RegimeTag.TRENDING_UP,)

    def param_space(self) -> list[ParamSpec]:
        """Tunables the walk-forward optimizer may explore.

        `rs_floor` sets how selective we are about leadership (O'Neil bought the
        top ~decile), `vol_surge_mult` how much institutional demand a breakout
        must show, `breakout_lookback` defines the pivot horizon, `trend_ema` the
        trend filter, `stop_pct` the hard risk cap, `min_eps_growth` the CAN SLIM
        earnings gate applied only when fundamentals are present, and
        `near_high_floor` the "N" gate — a breakout only counts when price is
        within reach of its 52-week high (a base breakout, not a bounce off a
        crushed name 40% below its high).
        """
        return [
            ParamSpec(name="rs_floor", default=85.0, lo=60.0, hi=95.0),
            ParamSpec(name="vol_surge_mult", default=1.4, lo=1.1, hi=2.5),
            ParamSpec(
                name="breakout_lookback", default=50.0, lo=20.0, hi=100.0, integer=True
            ),
            ParamSpec(name="trend_ema", default=50.0, lo=20.0, hi=200.0, integer=True),
            ParamSpec(name="stop_pct", default=8.0, lo=4.0, hi=12.0),
            ParamSpec(name="min_eps_growth", default=25.0, lo=0.0, hi=50.0),
            ParamSpec(name="near_high_floor", default=90.0, lo=70.0, hi=95.0),
        ]

    # ------------------------------------------------------------------ helpers

    def _rs_ranks(self, ctx: MarketContext) -> dict[str, float]:
        """Cross-sectional O'Neil relative-strength ranks (0..100) for the universe.

        Each symbol's trailing ~126-day return is percentile-ranked against every
        other tradable name so we keep only the leaders. SPY is the benchmark, not
        a candidate, and is excluded from the ranking universe.
        """
        rets: dict[str, float] = {}
        for sym, df in ctx.prices.items():
            if sym == "SPY" or df is None or len(df) < _RS_LOOKBACK + 1:
                continue
            close = df["c"]
            past = close.iloc[-(_RS_LOOKBACK + 1)]
            if past <= 0 or np.isnan(past):
                continue
            rets[sym] = close.iloc[-1] / past - 1.0
        if not rets:
            return {}
        ranked = pd.Series(rets).rank(pct=True) * 100.0
        return ranked.to_dict()

    @staticmethod
    def _eps_growth(fundamentals: dict[str, Any] | None, sym: str) -> float | None:
        """Best-effort quarterly EPS growth (%) lookup — schema-tolerant.

        Fundamentals plane layout can vary; we probe the common CAN SLIM keys and
        return None when the number simply isn't available so the caller can skip
        the gate gracefully (offline / equities-without-fundamentals mode).
        """
        if not fundamentals or sym not in fundamentals:
            return None
        rec = fundamentals[sym]
        if isinstance(rec, (int, float)):
            return float(rec)
        if isinstance(rec, dict):
            for key in (
                "eps_growth_qoq",
                "eps_growth",
                "quarterly_eps_growth",
                "eps_growth_pct",
                "epsGrowth",
            ):
                val = rec.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None
        return None

    # -------------------------------------------------------------------- scan

    def scan(self, ctx: MarketContext) -> list[FleetSignal]:
        """Find fresh pivot breakouts in market leaders.

        A BUY fires only when every gate holds on point-in-time data: (1) today's
        close clears the prior `breakout_lookback`-day Donchian high — Livermore's
        pivotal point; (2) volume >= `vol_surge_mult`x its 50-day average, proving
        O'Neil-style demand; (3) close is above the `trend_ema` EMA (in-trend);
        (4) the name is an RS leader (rank >= `rs_floor`); (5) if fundamentals
        exist for it, quarterly EPS growth clears `min_eps_growth`; and (6) price
        sits within `near_high_floor`% of its 52-week high — O'Neil's "N": a real
        base breakout near new-high ground, not a dead-cat pop in a name still 40%
        underwater. Stops are a fixed percent below entry with a 3:1 target.
        """
        rs_floor = float(self.params["rs_floor"])
        vol_mult = float(self.params["vol_surge_mult"])
        lookback = int(self.params["breakout_lookback"])
        trend_n = int(self.params["trend_ema"])
        stop_pct = float(self.params["stop_pct"])
        min_eps = float(self.params["min_eps_growth"])
        near_high = float(self.params["near_high_floor"])

        rs_ranks = self._rs_ranks(ctx)
        need = max(lookback + 1, trend_n, 50)

        signals: list[FleetSignal] = []
        for sym, df in ctx.prices.items():
            if sym == "SPY" or df is None or len(df) < need:
                continue

            high, low, close, vol = df["h"], df["l"], df["c"], df["v"]

            # (1) Livermore pivot: close clears the prior N-day high (excl. today).
            upper, _ = donchian(high, low, lookback)
            prior_high = upper.shift(1).iloc[-1]
            price = float(close.iloc[-1])
            if np.isnan(prior_high) or price <= prior_high:
                continue

            # (2) O'Neil demand: volume surge over the 50-day average.
            avg_vol = sma(vol, 50).iloc[-1]
            if np.isnan(avg_vol) or avg_vol <= 0:
                continue
            vol_ratio = float(vol.iloc[-1] / avg_vol)
            if vol_ratio < vol_mult:
                continue

            # (3) In-trend: above the trend EMA.
            trend_ema = ema(close, trend_n).iloc[-1]
            if np.isnan(trend_ema) or price <= trend_ema:
                continue

            # (4) RS leadership: cross-sectional percentile rank.
            rs_rank = rs_ranks.get(sym)
            if rs_rank is None or rs_rank < rs_floor:
                continue

            # (5) CAN SLIM earnings gate — only when fundamentals are present.
            eps_growth = self._eps_growth(ctx.fundamentals, sym)
            if eps_growth is not None and eps_growth < min_eps:
                continue

            # (6) O'Neil "N": the breakout must occur near new-high ground, not as a
            # dead-cat pop in a name still deep below its 52-week high. This gate is
            # what separates a base breakout from a bounce.
            win = min(_52W, len(high))
            hh = float(high.iloc[-win:].max())
            pct_of_52w_high = 100.0 * price / hh if hh > 0 else np.nan
            if np.isnan(pct_of_52w_high) or pct_of_52w_high < near_high:
                continue

            # ---- passed every gate: build the signal ----
            clenow = float(clenow_momentum(close, 90).iloc[-1])
            clenow_feat = None if np.isnan(clenow) else clenow

            # Confidence: blend RS leadership with Clenow trend quality (both 0..1).
            rs_component = rs_rank / 100.0
            clenow_norm = 0.0 if clenow_feat is None else float(
                np.clip(clenow / 100.0, 0.0, 1.0)
            )
            confidence = float(np.clip(0.6 * rs_component + 0.4 * clenow_norm, 0.0, 1.0))

            stop_loss = price * (1.0 - stop_pct / 100.0)
            take_profit = price * (1.0 + 3.0 * stop_pct / 100.0)

            features: dict[str, Any] = {
                "rs_rank": round(rs_rank, 2),
                "vol_ratio": round(vol_ratio, 3),
                "clenow": clenow_feat,
                "pct_of_52w_high": (
                    None if np.isnan(pct_of_52w_high) else round(pct_of_52w_high, 2)
                ),
            }
            if eps_growth is not None:
                features["eps_growth"] = eps_growth

            thesis = (
                f"{sym} cleared its {lookback}-day pivot at {price:.2f} on "
                f"{vol_ratio:.1f}x volume as an RS-{rs_rank:.0f} leader in an "
                f"up-trend — Livermore breakout, O'Neil leadership."
            )

            signals.append(
                FleetSignal(
                    bot_id=self.id,
                    at=ctx.now,
                    symbol=sym,
                    side="buy",
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    thesis=thesis,
                    features=features,
                )
            )

        return signals

    # ------------------------------------------------------------------ manage

    def manage(self, ctx: MarketContext) -> list[ManagementAction]:
        """Exit open positions on Livermore's "abnormal reaction" — or when the
        general market rolls over (O'Neil's "M").

        First, the market rule: when the regime turns RISK_OFF / TRENDING_DOWN,
        exit EVERY leader at once. O'Neil found three of four stocks follow the
        market, so when it tops you sell your leaders rather than waiting for each
        to break its own trend — selling INTO weakness. (The per-name tells below
        are too slow in a fast correction; this is the exit-side drawdown control.)

        Otherwise, two per-name tells end a winning trade: a *trend break* — three
        consecutive closes back below the `trend_ema` EMA — or a *climax top* —
        price stretched >50% above trend with RSI(14) over 80. Either fires an EXIT.
        """
        if ctx.regime.label in (RegimeTag.RISK_OFF, RegimeTag.TRENDING_DOWN):
            return [
                ManagementAction(kind="exit", symbol=sym,
                                 reason="market turned down (O'Neil M): sell the leaders")
                for sym in ctx.open_symbols
            ]

        trend_n = int(self.params["trend_ema"])
        actions: list[ManagementAction] = []

        for sym in ctx.open_symbols:
            df = ctx.prices.get(sym)
            if df is None or len(df) < max(trend_n, 14) + 3:
                continue

            close = df["c"]
            trend_ema = ema(close, trend_n)
            price = float(close.iloc[-1])
            ema_now = float(trend_ema.iloc[-1])
            if np.isnan(ema_now):
                continue

            # Trend break: three straight closes below the trend EMA.
            last3_close = close.iloc[-3:]
            last3_ema = trend_ema.iloc[-3:]
            trend_break = bool(
                (~last3_ema.isna()).all() and (last3_close < last3_ema).all()
            )

            # Climax top: parabolic extension above trend with overbought RSI.
            rsi_now = float(rsi(close, 14).iloc[-1])
            climax = bool(
                price > 1.5 * ema_now and not np.isnan(rsi_now) and rsi_now > 80.0
            )

            if not (trend_break or climax):
                continue

            if trend_break and climax:
                reason = (
                    f"Abnormal reaction: 3 closes below the {trend_n}-EMA (trend "
                    f"break) and a climax top (RSI {rsi_now:.0f}, +"
                    f"{100 * (price / ema_now - 1):.0f}% over trend)."
                )
            elif trend_break:
                reason = (
                    f"Trend break: 3 consecutive closes below the {trend_n}-day EMA."
                )
            else:
                reason = (
                    f"Climax top: close +{100 * (price / ema_now - 1):.0f}% above the "
                    f"{trend_n}-EMA with RSI {rsi_now:.0f} > 80."
                )

            actions.append(ManagementAction(kind="exit", symbol=sym, reason=reason))

        return actions
