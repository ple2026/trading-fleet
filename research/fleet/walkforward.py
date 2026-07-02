"""Walk-forward harness (§7b / §7e) and the Tier-A parameter tuner.

Two jobs, both out-of-sample by construction:

  * ``walk_forward`` runs a bot over rolling, *non-overlapping* test windows,
    chaining equity across them, and reports metrics on the concatenated
    out-of-sample curve only. The first ``train_years`` of history is burned as
    warm-up/training and never traded, so no in-sample bar reaches the result.

  * ``tune`` is Tier A: it searches a bot's ``param_space`` within the per-param
    drift bound, scores each candidate by walk-forward OOS Sharpe, and applies the
    §7b acceptance gates (Sharpe gain, drawdown guard, drift cap, minimum trades).
    It never deploys — it returns a :class:`TuningProposal` that a human reviews
    and merges as a config PR. Git history is the parameter audit trail.

Design note: because ``run_backtest`` slices point-in-time history from the full
panel, a bot run over a *test* window still sees all bars before it — indicators
warm up correctly. The window's ``start`` gates only which days are simulated and
traded, which is exactly what makes the OOS split honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .backtest import Trade, compute_metrics, run_backtest
from .bot import Bot, ParamSpec

# Trading-day approximations used to size windows from a sorted date index. Using
# bar counts (not calendar deltas) keeps test windows contiguous and gap-free.
_DAYS_PER_YEAR = 252
_DAYS_PER_MONTH = 21


@dataclass(frozen=True)
class Window:
    """One anchored walk-forward split: fit on [train_start, train_end], validate
    out-of-sample on (train_end, test_end]."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class WalkForwardResult:
    oos_equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        """Metrics on the concatenated OOS curve. Empty dict if nothing traded."""
        return compute_metrics(self.oos_equity, self.trades)


def make_windows(
    dates: list[datetime],
    train_years: float = 3.0,
    test_months: int = 6,
    anchored: bool = True,
) -> list[Window]:
    """Tile [dates] into contiguous, non-overlapping test windows after an initial
    training span.

    ``anchored`` (default) grows the training window from a fixed anchor at the
    first date (expanding-window walk-forward); otherwise the training window
    rolls at a fixed ``train_years`` width. Either way the test windows never
    overlap, so concatenating their equity curves double-counts no bar.
    """
    n = len(dates)
    train_bars = max(int(train_years * _DAYS_PER_YEAR), 1)
    test_bars = max(int(test_months * _DAYS_PER_MONTH), 1)
    if n <= train_bars + 1:
        return []

    windows: list[Window] = []
    i = train_bars
    while i < n:
        j = min(i + test_bars, n)  # exclusive end index for this test window
        train_start = dates[0] if anchored else dates[max(0, i - train_bars)]
        windows.append(
            Window(
                train_start=train_start,
                train_end=dates[i - 1],
                test_start=dates[i],
                test_end=dates[j - 1],
            )
        )
        i = j
    return windows


def _panel_dates(
    panel: dict[str, pd.DataFrame], start: datetime, end: datetime
) -> list[datetime]:
    return sorted({d for df in panel.values() for d in df.index if start <= d <= end})


def walk_forward(
    bot_class: type[Bot],
    panel: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    params: dict[str, float] | None = None,
    *,
    train_years: float = 3.0,
    test_months: int = 6,
    anchored: bool = True,
    starting_equity: float = 10_000.0,
    benchmark: str = "SPY",
    macro=None,
    fundamentals=None,
) -> WalkForwardResult:
    """Run ``bot_class`` walk-forward with fixed ``params``; return OOS-only result.

    A fresh bot is constructed per window so stateful bots reset cleanly, and
    equity is chained across windows to give one continuous out-of-sample curve.
    Point-in-time ``macro`` / ``fundamentals`` planes are forwarded to each window.
    """
    dates = _panel_dates(panel, start, end)
    windows = make_windows(dates, train_years, test_months, anchored)

    equity = starting_equity
    curves: list[pd.Series] = []
    trades: list[Trade] = []
    for w in windows:
        bot = bot_class(params)
        res = run_backtest(
            bot, panel, w.test_start, w.test_end,
            starting_equity=equity, benchmark=benchmark,
            macro=macro, fundamentals=fundamentals,
        )
        if res.equity_curve.empty:
            continue
        curves.append(res.equity_curve)
        trades.extend(res.trades)
        equity = float(res.equity_curve.iloc[-1])

    oos = pd.concat(curves) if curves else pd.Series(dtype=float)
    return WalkForwardResult(oos_equity=oos, trades=trades, windows=windows)


def oos_sharpe(
    bot_class: type[Bot],
    panel: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    params: dict[str, float] | None = None,
    **wf_kwargs: Any,
) -> float:
    """A bot's out-of-sample Sharpe from the walk-forward harness (0.0 if flat).

    This is the honest edge signal the meta-allocator should tilt on — measured
    on data the bot never fit, not the in-sample or trailing-window Sharpe.
    """
    return float(walk_forward(bot_class, panel, start, end, params, **wf_kwargs)
                 .metrics().get("sharpe", 0.0))


def fleet_oos_sharpe(
    bots: dict[str, type[Bot]],
    panel: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    **wf_kwargs: Any,
) -> dict[str, float]:
    """Per-bot walk-forward OOS Sharpe — the allocator's bandit-tilt input."""
    return {
        name: oos_sharpe(cls, panel, start, end, **wf_kwargs)
        for name, cls in bots.items()
    }


# ---------------------------------------------------------------- Tier A tuner

# §7b acceptance gates — all required for a proposal to be accepted.
MIN_SHARPE_GAIN = 0.2          # OOS Sharpe must improve by at least this
MAX_DD_WORSENING = 0.10        # OOS max drawdown may not deepen by more than 10%
MIN_VALIDATION_TRADES = 30     # need enough OOS trades to infer anything


@dataclass
class TuningProposal:
    """The output of a Tier-A re-fit. Deploys nothing — it is a reviewable diff."""

    bot_id: str
    accepted: bool
    reason: str
    current_params: dict[str, float]
    current_metrics: dict[str, float]
    proposed_params: dict[str, float] | None = None
    proposed_metrics: dict[str, float] | None = None
    candidates_evaluated: int = 0

    def diff(self) -> dict[str, tuple[float, float]]:
        """Per-param (current -> proposed) for the params that actually moved."""
        if not self.proposed_params:
            return {}
        return {
            k: (self.current_params[k], self.proposed_params[k])
            for k in self.proposed_params
            if self.proposed_params[k] != self.current_params[k]
        }

    def config_diff(self) -> str:
        """Render the parameter change as a reviewable config diff — the artifact a
        human merges. Empty string when nothing is proposed."""
        moved = self.diff()
        if not moved:
            return ""
        lines = [f"# {self.bot_id} — Tier-A param re-fit ({self.reason})"]
        for k, (cur, new) in sorted(moved.items()):
            lines.append(f"-{k}: {cur:g}")
            lines.append(f"+{k}: {new:g}")
        return "\n".join(lines)

    def as_row(self, created_at: datetime | None = None) -> dict[str, Any]:
        """A `proposals`-table row (tier A). Rejections are logged too, with the
        gate that failed as `human_reason`, so the audit trail is complete."""
        return {
            "bot_id": self.bot_id,
            "tier": "A",
            "created_at": created_at,
            "description": (
                f"[{self.bot_id}] Tier-A re-fit: {self.reason}"
                if self.accepted
                else f"[{self.bot_id}] Tier-A re-fit rejected: {self.reason}"
            ),
            "diff": self.config_diff(),
            "backtest_report": {
                "current": self.current_metrics,
                "proposed": self.proposed_metrics,
                "candidates_evaluated": self.candidates_evaluated,
            },
            "status": "draft" if self.accepted else "rejected",
            "human_reason": None if self.accepted else self.reason,
        }


def _sample_candidate(
    specs: list[ParamSpec], current: dict[str, float], rng: np.random.Generator
) -> dict[str, float]:
    """Perturb each param within its drift bound, clipped to [lo, hi].

    Drift is measured against the parameter's *range* so a param whose current
    value is 0 (a valid default) can still be explored. This keeps every generated
    candidate inside the §7b drift cap by construction.
    """
    cand: dict[str, float] = {}
    for spec in specs:
        step = spec.max_step_pct * (spec.hi - spec.lo)
        lo = max(spec.lo, current[spec.name] - step)
        hi = min(spec.hi, current[spec.name] + step)
        val = float(rng.uniform(lo, hi))
        if spec.integer:
            val = float(int(round(val)))
        cand[spec.name] = val
    return cand


def _within_drift(
    cand: dict[str, float], current: dict[str, float], specs: list[ParamSpec]
) -> bool:
    for spec in specs:
        step = spec.max_step_pct * (spec.hi - spec.lo)
        if abs(cand[spec.name] - current[spec.name]) > step + 1e-9:
            return False
    return True


def _gate(
    baseline: dict[str, float], cand: dict[str, float]
) -> tuple[bool, str]:
    """Apply the §7b acceptance gates. Returns (passed, human-readable reason)."""
    if not cand:
        return False, "candidate produced no OOS metrics"
    if cand.get("trades", 0) < MIN_VALIDATION_TRADES:
        return False, (
            f"only {cand.get('trades', 0)} OOS trades "
            f"(< {MIN_VALIDATION_TRADES}) — insufficient to infer"
        )
    sharpe_gain = cand["sharpe"] - baseline.get("sharpe", 0.0)
    if sharpe_gain < MIN_SHARPE_GAIN:
        return False, (
            f"OOS Sharpe gain {sharpe_gain:+.2f} < required {MIN_SHARPE_GAIN}"
        )
    # max_dd is stored as a negative percent; "worse" means more negative.
    base_dd = abs(baseline.get("max_dd", 0.0))
    cand_dd = abs(cand["max_dd"])
    if base_dd > 0 and cand_dd > base_dd * (1 + MAX_DD_WORSENING):
        return False, (
            f"OOS max-DD worsened {cand_dd:.1f}% vs {base_dd:.1f}% "
            f"(> {int(MAX_DD_WORSENING * 100)}% cap)"
        )
    return True, (
        f"OOS Sharpe {baseline.get('sharpe', 0.0):.2f} -> {cand['sharpe']:.2f} "
        f"(+{sharpe_gain:.2f}) with {cand.get('trades', 0)} OOS trades"
    )


def tune(
    bot_class: type[Bot],
    panel: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    *,
    n_candidates: int = 16,
    seed: int = 0,
    train_years: float = 3.0,
    test_months: int = 6,
    anchored: bool = True,
    benchmark: str = "SPY",
    macro=None,
    fundamentals=None,
) -> TuningProposal:
    """Tier-A walk-forward re-fit. Searches ``param_space`` within drift bounds,
    scores candidates by OOS Sharpe, and accepts one only if it clears every §7b
    gate. Returns a reviewable proposal — it deploys nothing.
    """
    probe = bot_class()
    specs = probe.param_space()
    current = {s.name: float(s.default) for s in specs}
    bot_id = probe.id

    wf_kwargs = dict(
        train_years=train_years, test_months=test_months,
        anchored=anchored, benchmark=benchmark,
        macro=macro, fundamentals=fundamentals,
    )

    baseline = walk_forward(bot_class, panel, start, end, current, **wf_kwargs).metrics()

    rng = np.random.default_rng(seed)
    survivors: list[tuple[dict[str, float], dict[str, float], str]] = []
    best_effort: tuple[dict[str, float], dict[str, float]] | None = None
    for _ in range(n_candidates):
        cand = _sample_candidate(specs, current, rng)
        if not _within_drift(cand, current, specs):  # belt-and-suspenders
            continue
        m = walk_forward(bot_class, panel, start, end, cand, **wf_kwargs).metrics()
        if not m:
            continue
        # Track the highest-Sharpe candidate regardless of gates, so a rejection
        # can explain the *closest miss* rather than a random failure.
        if best_effort is None or m["sharpe"] > best_effort[1]["sharpe"]:
            best_effort = (cand, m)
        passed, reason = _gate(baseline, m)
        if passed:
            survivors.append((cand, m, reason))

    if survivors:
        # Among candidates that clear EVERY gate, deploy the best OOS Sharpe.
        params, metrics, reason = max(survivors, key=lambda t: t[1]["sharpe"])
        return TuningProposal(
            bot_id=bot_id, accepted=True, reason=reason,
            current_params=current, current_metrics=baseline,
            proposed_params=params, proposed_metrics=metrics,
            candidates_evaluated=n_candidates,
        )

    if best_effort is None:
        reason = "no candidate produced tradeable OOS results"
    else:
        _, reason = _gate(baseline, best_effort[1])  # why the best miss failed
    return TuningProposal(
        bot_id=bot_id, accepted=False, reason=reason,
        current_params=current, current_metrics=baseline,
        candidates_evaluated=n_candidates,
    )
