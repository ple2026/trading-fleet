"""L3 meta-allocator and correlation monitor (§8).

The fleet earns capital by measured edge, not by opinion. Monthly:

  1. **Risk-parity base** — weight bots so each contributes equal volatility
     (inverse-vol, the standard first-order ERC approximation). A quiet
     market-neutral pod and a punchy directional bot should not carry the same
     dollar risk at the same weight.
  2. **Bandit tilt** — nudge weight toward each bot's rolling out-of-sample
     Sharpe, quarter-Kelly-scaled, so demonstrated edge compounds while an unproven
     bot stays small.
  3. **Hard caps** — every bot stays within [10%, 40%] of the fleet, so no single
     pod can dominate and none is starved. Enforced by water-filling.

The **correlation monitor** is the guardrail against the classic failure — four
bots that are all secretly long QQQ. Any pair over 0.7 alerts; a fleet-average
pairwise correlation over 0.5 freezes new entries pending human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .backtest import Trade, run_backtest
from .bot import Bot

TRADING_DAYS = 252

# §8 governance constants.
MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.40
KELLY_FRACTION = 0.25          # quarter-Kelly tilt strength
PAIR_ALERT = 0.70              # any pair above this correlation alerts
FLEET_FREEZE = 0.50            # fleet-average pairwise correlation above this freezes
CORR_WINDOW = 60               # trailing days for the correlation monitor
SHARPE_WINDOW = 90             # trailing days for the rolling-Sharpe tilt


@dataclass
class FleetBacktest:
    equity: pd.DataFrame                       # per-bot equity curves (columns=bot ids)
    returns: pd.DataFrame                      # per-bot daily returns
    trades: dict[str, list[Trade]] = field(default_factory=dict)


def run_fleet_backtest(
    bots: dict[str, type[Bot]],
    panel: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    seed_equity: float = 10_000.0,
    benchmark: str = "SPY",
    journals: dict | None = None,
) -> FleetBacktest:
    """Run every bot independently on the same panel; collect their return streams.

    Each bot trades its own $10k sub-book (no shared capital here — the ledger
    enforces that live). The point of running them together is the *joint*
    distribution: the returns DataFrame feeds both the allocator and the
    correlation monitor.
    """
    equity: dict[str, pd.Series] = {}
    trades: dict[str, list[Trade]] = {}
    for name, cls in bots.items():
        journal = journals.get(name) if journals else None
        res = run_backtest(
            cls(), panel, start, end,
            starting_equity=seed_equity, benchmark=benchmark, journal=journal,
        )
        equity[name] = res.equity_curve
        trades[name] = res.trades
    eq_df = pd.DataFrame(equity).sort_index()
    returns = eq_df.pct_change().dropna(how="all")
    return FleetBacktest(equity=eq_df, returns=returns, trades=trades)


# ------------------------------------------------------------------- weighting

def _equal_weights(names: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


def risk_parity_weights(
    returns: pd.DataFrame, lookback: int | None = None
) -> pd.Series:
    """Inverse-volatility weights — the first-order equal-risk-contribution base.

    A bot with no measurable volatility (never traded in the window) gets zero
    inverse-vol; if every bot is flat we fall back to equal weight so the fleet is
    always fully specified.
    """
    if returns.empty or returns.shape[1] == 0:
        return pd.Series(dtype=float)
    r = returns.tail(lookback) if lookback else returns
    vol = r.std()
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = float(inv.sum())
    if total <= 0:
        return _equal_weights(list(returns.columns))
    return inv / total


def _rolling_sharpe(returns: pd.DataFrame, window: int = SHARPE_WINDOW) -> pd.Series:
    """Annualized trailing Sharpe per bot over the last `window` days."""
    r = returns.tail(window)
    mean = r.mean()
    std = r.std()
    sharpe = (mean / std) * np.sqrt(TRADING_DAYS)
    return sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def feasible_caps(
    n: int, lo: float = MIN_WEIGHT, hi: float = MAX_WEIGHT
) -> tuple[float, float]:
    """Relax the §8 caps just enough to stay feasible for `n` bots.

    The [10%, 40%] policy is written for the full four-bot fleet (4*0.40=1.6 >= 1).
    With fewer bots — e.g. a burn-in fleet of two — a 40% ceiling can't sum to 100%,
    so we widen the bound to 1/n. At n=4 this is a no-op and the policy is exact.
    """
    if n <= 0:
        return lo, hi
    return min(lo, 1.0 / n), max(hi, 1.0 / n)


def _apply_caps(
    weights: pd.Series, lo: float = MIN_WEIGHT, hi: float = MAX_WEIGHT
) -> pd.Series:
    """Project weights onto {w: lo <= w_i <= hi, sum w = 1} by water-filling.

    Requires the caps to be feasible for the count of bots (n*lo <= 1 <= n*hi);
    with four bots and [0.10, 0.40] that always holds. Clamped weights are pinned
    and the residual mass is redistributed across the still-free bots until the
    allocation is self-consistent.
    """
    w = weights.astype(float).copy()
    n = len(w)
    if n == 0:
        return w
    if n * lo > 1 + 1e-9 or n * hi < 1 - 1e-9:
        raise ValueError(f"caps [{lo}, {hi}] infeasible for {n} bots")
    # Start from a normalized, non-negative vector.
    w = w.clip(lower=0.0)
    w = _equal_weights(list(w.index)) if w.sum() <= 0 else w / w.sum()

    for _ in range(1000):
        w = w.clip(lower=lo, upper=hi)
        residual = 1.0 - float(w.sum())
        if abs(residual) < 1e-12:
            break
        # Only names that can still move in the needed direction absorb the
        # residual: to ADD weight, names below the cap; to REMOVE, names above the
        # floor. Distributing equally among them always converges to a feasible
        # capped simplex (feasibility guaranteed by the n*lo<=1<=n*hi check above).
        free = (w < hi - 1e-12) if residual > 0 else (w > lo + 1e-12)
        k = int(free.sum())
        if k == 0:
            break
        w[free] = w[free] + residual / k
    return w


def bandit_tilt(
    base_weights: pd.Series,
    oos_sharpe: pd.Series,
    kelly_fraction: float = KELLY_FRACTION,
) -> pd.Series:
    """Tilt the risk-parity base toward positive rolling OOS Sharpe.

    Multiplier for bot i is ``1 + kelly_fraction * max(sharpe_i, 0)`` — a bot with
    no demonstrated edge keeps its base weight; a proven one leans in, but only
    quarter-Kelly hard. Negative Sharpe never tilts *below* base (that is the caps'
    job, and demotion happens by others tilting up). Result is re-normalized and
    cap-projected.
    """
    if base_weights.empty:
        return base_weights
    sharpe = oos_sharpe.reindex(base_weights.index).fillna(0.0).clip(lower=0.0)
    mult = 1.0 + kelly_fraction * sharpe
    tilted = base_weights * mult
    total = float(tilted.sum())
    tilted = base_weights if total <= 0 else tilted / total
    lo, hi = feasible_caps(len(base_weights))
    return _apply_caps(tilted, lo, hi)


# --------------------------------------------------------- correlation monitor

@dataclass
class CorrelationReport:
    matrix: pd.DataFrame
    hot_pairs: list[tuple[str, str, float]]    # pairs above PAIR_ALERT
    fleet_avg: float                           # mean pairwise correlation
    freeze: bool                               # fleet_avg above FLEET_FREEZE

    def summary(self) -> str:
        parts = [f"fleet avg corr {self.fleet_avg:+.2f}"]
        if self.hot_pairs:
            parts.append(
                "hot: " + ", ".join(f"{a}/{b} {c:+.2f}" for a, b, c in self.hot_pairs)
            )
        if self.freeze:
            parts.append("FREEZE new entries")
        return " | ".join(parts)


def correlation_monitor(
    returns: pd.DataFrame,
    window: int = CORR_WINDOW,
    pair_alert: float = PAIR_ALERT,
    freeze_threshold: float = FLEET_FREEZE,
) -> CorrelationReport:
    """Pairwise return correlation over the trailing `window`; flag crowding."""
    r = returns.tail(window)
    corr = r.corr()
    names = list(corr.columns)
    hot: list[tuple[str, str, float]] = []
    vals: list[float] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = corr.loc[a, b]
            if pd.notna(c):
                vals.append(float(c))
                if c > pair_alert:
                    hot.append((a, b, float(c)))
    fleet_avg = float(np.mean(vals)) if vals else 0.0
    return CorrelationReport(
        matrix=corr, hot_pairs=hot, fleet_avg=fleet_avg,
        freeze=fleet_avg > freeze_threshold,
    )


# ------------------------------------------------------------------- allocate

@dataclass
class Allocation:
    at: datetime
    weights: dict[str, float]
    capital_usd: dict[str, float]
    base_weights: dict[str, float]
    oos_sharpe: dict[str, float]
    correlation: CorrelationReport
    frozen: bool
    reason: str


def allocate(
    returns: pd.DataFrame,
    total_usd: float,
    at: datetime,
    oos_sharpe: pd.Series | dict[str, float] | None = None,
    *,
    corr_window: int = CORR_WINDOW,
    sharpe_window: int = SHARPE_WINDOW,
    kelly_fraction: float = KELLY_FRACTION,
) -> Allocation:
    """One monthly rebalance: risk-parity base -> bandit tilt -> caps, plus the
    correlation freeze check.

    `oos_sharpe` should be each bot's rolling out-of-sample Sharpe (from the
    walk-forward harness). If omitted we approximate it with the trailing
    `sharpe_window`-day Sharpe of the supplied returns — convenient for a fleet
    backtest, but true Tier-A OOS Sharpe is the honest input in production.
    """
    base = risk_parity_weights(returns)
    if oos_sharpe is None:
        sharpe = _rolling_sharpe(returns, sharpe_window)
    elif isinstance(oos_sharpe, dict):
        sharpe = pd.Series(oos_sharpe)
    else:
        sharpe = oos_sharpe

    weights = bandit_tilt(base, sharpe, kelly_fraction)
    corr = correlation_monitor(returns, corr_window)
    capital = (weights * total_usd)

    reason = (
        f"risk-parity + quarter-Kelly tilt; {corr.summary()}"
        if not corr.freeze
        else f"FROZEN: {corr.summary()} — new entries paused pending review"
    )
    return Allocation(
        at=at,
        weights=weights.round(4).to_dict(),
        capital_usd=capital.round(2).to_dict(),
        base_weights=base.round(4).to_dict(),
        oos_sharpe=sharpe.reindex(base.index).fillna(0.0).round(3).to_dict(),
        correlation=corr,
        frozen=corr.freeze,
        reason=reason,
    )
