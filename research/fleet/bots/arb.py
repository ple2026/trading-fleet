"""ARB — a modern Ed Thorp statistical-arbitrage pod (pairs / cointegration).

Thorp's Princeton-Newport playbook in miniature: find two instruments driven by
the same factor, trade the *spread* between them, and stay market-neutral so the
P&L depends on convergence rather than on getting market direction right. For
each candidate ETF pair we OLS-hedge leg A against leg B, test the residual
spread for cointegration (ADF), require a tradeable mean-reversion speed
(Ornstein-Uhlenbeck half-life), and fire when the spread's z-score is stretched.

Honest limitation: at a $10k account this is a research/infrastructure pod, not a
high-Sharpe edge machine. Liquid-ETF pairs are heavily arbitraged, the two-leg
structure doubles commission/borrow drag, and market-neutral by construction
caps the dollar edge. It earns its place by being genuinely uncorrelated with the
directional bots and by exercising the pairs plumbing (hedge legs, cointegration
monitoring) the fleet needs — not by minting alpha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from ..bot import Bot, MarketContext, ParamSpec
from ..types import FleetSignal, ManagementAction, RegimeTag


class Arb(Bot):
    """Cointegration pairs bot. Market-neutral, mean-reversion on the spread."""

    id = "arb"
    favorable_regimes = (RegimeTag.RANGE, RegimeTag.HIGH_VOL)

    # Liquid, economically-linked ETF pairs. Leg A is regressed on leg B.
    CANDIDATE_PAIRS: list[tuple[str, str]] = [
        ("XLE", "OIH"),
        ("SMH", "QQQ"),
        ("XLF", "KRE"),
        ("GLD", "GDX"),
        ("XLK", "QQQ"),
    ]

    def param_space(self) -> list[ParamSpec]:
        return [
            ParamSpec(name="entry_z", default=2.0, lo=1.5, hi=3.0),
            ParamSpec(name="exit_z", default=0.25, lo=0.0, hi=1.0),
            ParamSpec(name="stop_z", default=3.5, lo=3.0, hi=5.0),
            ParamSpec(name="half_life_max", default=20, lo=5, hi=40, integer=True),
            ParamSpec(name="lookback", default=252, lo=120, hi=504, integer=True),
        ]

    # ---- internals ---------------------------------------------------------

    def _log_closes(
        self, ctx: MarketContext, a: str, b: str, lookback: int
    ) -> tuple[pd.Series, pd.Series] | None:
        """Return aligned trailing log-close windows for (leg A, leg B), or None
        if either leg is missing or there is not enough overlapping history.
        """
        if a not in ctx.prices or b not in ctx.prices:
            return None
        ca = ctx.prices[a].get("c")
        cb = ctx.prices[b].get("c")
        if ca is None or cb is None:
            return None
        joined = pd.concat(
            [ca.rename("a"), cb.rename("b")], axis=1, join="inner"
        ).dropna()
        # Prices must be strictly positive to take logs.
        joined = joined[(joined["a"] > 0) & (joined["b"] > 0)]
        if len(joined) < lookback:
            return None
        joined = joined.tail(lookback)
        return np.log(joined["a"]), np.log(joined["b"])

    @staticmethod
    def _hedge_and_spread(
        log_a: pd.Series, log_b: pd.Series
    ) -> tuple[float, pd.Series]:
        """OLS leg A on leg B (with constant); return (beta, spread = a - beta*b)."""
        x = sm.add_constant(log_b.values)
        model = sm.OLS(log_a.values, x).fit()
        beta = float(model.params[1])
        spread = log_a - beta * log_b
        return beta, spread

    @staticmethod
    def _half_life(spread: pd.Series) -> float:
        """OU half-life: regress delta_spread on lagged spread; hl = -ln2 / lambda.

        Returns NaN on a degenerate fit (non-negative lambda => not mean-reverting,
        or insufficient variation).
        """
        lag = spread.shift(1)
        delta = spread - lag
        df = pd.concat([delta.rename("d"), lag.rename("l")], axis=1).dropna()
        if len(df) < 10:
            return float("nan")
        x = sm.add_constant(df["l"].values)
        model = sm.OLS(df["d"].values, x).fit()
        lam = float(model.params[1])
        if lam >= 0:  # not mean-reverting
            return float("nan")
        return float(-np.log(2) / lam)

    @staticmethod
    def _zscore(spread: pd.Series) -> tuple[float, float]:
        """Return (last z-score, spread std) over the full spread window."""
        mu = float(spread.mean())
        sd = float(spread.std(ddof=0))
        if sd <= 0 or not np.isfinite(sd):
            return float("nan"), float("nan")
        z = (float(spread.iloc[-1]) - mu) / sd
        return z, sd

    def _adf_pvalue(self, spread: pd.Series) -> float:
        """ADF p-value on the spread. Returns NaN if adfuller raises (degenerate)."""
        try:
            result = adfuller(spread.values, autolag="AIC")
        except Exception:
            return float("nan")
        return float(result[1])

    def _evaluate(
        self, ctx: MarketContext, a: str, b: str, lookback: int
    ) -> dict | None:
        """Full pairs evaluation for (a, b). Returns a metrics dict or None."""
        logs = self._log_closes(ctx, a, b, lookback)
        if logs is None:
            return None
        log_a, log_b = logs
        try:
            beta, spread = self._hedge_and_spread(log_a, log_b)
        except Exception:
            return None
        z, spread_std = self._zscore(spread)
        if not np.isfinite(z):
            return None
        adf_p = self._adf_pvalue(spread)
        half_life = self._half_life(spread)
        # Leg A last close in price (not log) terms.
        price_a = float(np.exp(log_a.iloc[-1]))
        return {
            "beta": beta,
            "zscore": z,
            "adf_pvalue": adf_p,
            "half_life": half_life,
            "spread_std": spread_std,
            "price_a": price_a,
        }

    # ---- Bot API -----------------------------------------------------------

    def scan(self, ctx: MarketContext) -> list[FleetSignal]:
        entry_z = float(self.params["entry_z"])
        exit_z = float(self.params["exit_z"])
        stop_z = float(self.params["stop_z"])
        half_life_max = float(self.params["half_life_max"])
        lookback = int(self.params["lookback"])

        signals: list[FleetSignal] = []
        for a, b in self.CANDIDATE_PAIRS:
            m = self._evaluate(ctx, a, b, lookback)
            if m is None:
                continue
            adf_p = m["adf_pvalue"]
            half_life = m["half_life"]
            z = m["zscore"]
            beta = m["beta"]
            spread_std = m["spread_std"]

            # Gates: cointegrated, mean-reverting fast enough, and stretched.
            if not np.isfinite(adf_p) or adf_p >= 0.05:
                continue
            if not (np.isfinite(half_life) and 0 < half_life <= half_life_max):
                continue
            if abs(z) < entry_z:
                continue

            # Spread rich (z>0) => short A / long B; spread cheap => long A / short B.
            side = "sell" if z > 0 else "buy"

            # Approximate price stops. The trade is on the SPREAD, so there is no
            # true single-leg price stop. We translate z-distances into leg-A
            # price moves: a move of `dz` z-units changes the spread (in log-price
            # units) by dz * spread_std. Holding leg B fixed, leg A's log-price
            # moves by the same amount, so the multiplicative factor on price_a is
            # exp(+/- dz * spread_std). Target = converge to exit_z, stop = diverge
            # to stop_z. Approximation ignores leg-B co-movement (hedge_ratio beta
            # is carried separately for the sizer/executor).
            price = m["price_a"]
            dz_target = abs(z) - exit_z          # z shrinks toward exit on convergence
            dz_stop = max(stop_z - abs(z), 0.0)  # z grows toward stop on divergence
            move_target = dz_target * spread_std
            move_stop = dz_stop * spread_std
            if side == "sell":
                # Short A: profit as A falls (spread narrows), loss as A rises.
                take_profit = price * float(np.exp(-move_target))
                stop_loss = price * float(np.exp(move_stop))
            else:
                # Long A: profit as A rises (spread narrows), loss as A falls.
                take_profit = price * float(np.exp(move_target))
                stop_loss = price * float(np.exp(-move_stop))

            confidence = float(min(abs(z) / 3.0, 1.0))
            thesis = (
                f"{a}/{b} spread {abs(z):.1f}z {'rich' if z > 0 else 'cheap'} "
                f"(half-life {half_life:.0f}d) — mean-reversion, market-neutral vs {b}."
            )
            signals.append(
                FleetSignal(
                    bot_id=self.id,
                    at=ctx.now,
                    symbol=a,
                    side=side,
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence,
                    thesis=thesis,
                    features={
                        "beta": beta,
                        "zscore": z,
                        "adf_pvalue": adf_p,
                        "half_life": half_life,
                        "spread_std": spread_std,
                    },
                    hedge_symbol=b,
                    hedge_ratio=beta,
                )
            )
        return signals

    def manage(self, ctx: MarketContext) -> list[ManagementAction]:
        exit_z = float(self.params["exit_z"])
        stop_z = float(self.params["stop_z"])
        lookback = int(self.params["lookback"])

        # Map each candidate leg-A symbol to its pair so we can recompute.
        leg_a_pairs = {a: (a, b) for a, b in self.CANDIDATE_PAIRS}

        actions: list[ManagementAction] = []
        for symbol in ctx.open_symbols:
            pair = leg_a_pairs.get(symbol)
            if pair is None:
                continue  # not an ARB leg-A position we recognize
            a, b = pair
            m = self._evaluate(ctx, a, b, lookback)
            if m is None:
                continue
            z = m["zscore"]
            adf_p = m["adf_pvalue"]

            if np.isfinite(z) and abs(z) <= exit_z:
                reason = f"{a}/{b} spread converged (|z|={abs(z):.2f} <= {exit_z}) — take profit"
                actions.append(ManagementAction(kind="exit", symbol=symbol, reason=reason))
            elif np.isfinite(z) and abs(z) >= stop_z:
                reason = f"{a}/{b} spread diverged (|z|={abs(z):.2f} >= {stop_z}) — stop"
                actions.append(ManagementAction(kind="exit", symbol=symbol, reason=reason))
            elif not np.isfinite(adf_p) or adf_p >= 0.10:
                reason = f"{a}/{b} cointegration broke (adf p={adf_p:.2f} >= 0.10)"
                actions.append(ManagementAction(kind="exit", symbol=symbol, reason=reason))
        return actions
