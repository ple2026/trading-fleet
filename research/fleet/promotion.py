"""Champion / challenger promotion (§7c).

An accepted Tier-A (or Tier-B) proposal does not deploy on the strength of the
re-fit alone — that would trust the tuner's own out-of-sample window, which it
optimized against and could have overfit. Instead the challenger runs as a shadow
on a *held-out* period the tuner never saw. Only if it beats the champion there,
by the same gates, is it promoted.

`shadow_compare` runs champion and challenger over an identical held-out window and
returns a promote/hold decision. Because the shadow window is disjoint from the
tuning window, a single-pass backtest over it is genuinely out-of-sample — the bot
still sees full point-in-time history, so indicators warm up correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .backtest import run_backtest
from .bot import Bot
from .walkforward import MAX_DD_WORSENING, MIN_SHARPE_GAIN

# A shadow window is short (weeks–months), so fewer trades are expected than a full
# walk-forward — but there must be enough to mean anything.
MIN_SHADOW_TRADES = 10


@dataclass
class ShadowResult:
    bot_id: str
    promote: bool
    reason: str
    shadow_start: datetime
    shadow_end: datetime
    champion_metrics: dict[str, float]
    challenger_metrics: dict[str, float]


def split_date(
    panel: dict[str, pd.DataFrame], end: datetime, holdout_days: int = 252
) -> datetime | None:
    """The date that carves off the last `holdout_days` trading days as a shadow
    set. None if the panel is too short to hold anything out."""
    dates = sorted({d for df in panel.values() for d in df.index if d <= end})
    if len(dates) <= holdout_days:
        return None
    return dates[-holdout_days]


def _promotion_gate(
    champ: dict[str, float],
    chal: dict[str, float],
    *,
    min_trades: int,
    sharpe_margin: float,
    dd_worsening: float,
) -> tuple[bool, str]:
    """Promote only if the challenger clearly beats the champion out-of-sample."""
    if not chal:
        return False, "challenger produced no shadow trades"
    if chal.get("trades", 0) < min_trades:
        return False, (
            f"only {chal.get('trades', 0)} shadow trades (< {min_trades}) — hold"
        )
    gain = chal["sharpe"] - champ.get("sharpe", 0.0)
    if gain < sharpe_margin:
        return False, (
            f"shadow Sharpe gain {gain:+.2f} < required {sharpe_margin} — hold champion"
        )
    champ_dd = abs(champ.get("max_dd", 0.0))
    chal_dd = abs(chal["max_dd"])
    if champ_dd > 0 and chal_dd > champ_dd * (1 + dd_worsening):
        return False, (
            f"shadow max-DD worsened {chal_dd:.1f}% vs {champ_dd:.1f}% — hold champion"
        )
    return True, (
        f"challenger beats champion OOS: Sharpe {champ.get('sharpe', 0.0):.2f} -> "
        f"{chal['sharpe']:.2f} (+{gain:.2f}) over {chal['trades']} shadow trades — PROMOTE"
    )


def shadow_compare(
    bot_class: type[Bot],
    panel: dict[str, pd.DataFrame],
    champion_params: dict[str, float],
    challenger_params: dict[str, float],
    shadow_start: datetime,
    shadow_end: datetime,
    *,
    benchmark: str = "SPY",
    min_trades: int = MIN_SHADOW_TRADES,
    sharpe_margin: float = MIN_SHARPE_GAIN,
    dd_worsening: float = MAX_DD_WORSENING,
    macro=None,
    fundamentals=None,
) -> ShadowResult:
    """Run champion and challenger over the held-out shadow window and decide."""
    champ = run_backtest(
        bot_class(champion_params), panel, shadow_start, shadow_end,
        benchmark=benchmark, macro=macro, fundamentals=fundamentals,
    ).metrics()
    chal = run_backtest(
        bot_class(challenger_params), panel, shadow_start, shadow_end,
        benchmark=benchmark, macro=macro, fundamentals=fundamentals,
    ).metrics()
    promote, reason = _promotion_gate(
        champ, chal, min_trades=min_trades,
        sharpe_margin=sharpe_margin, dd_worsening=dd_worsening,
    )
    return ShadowResult(
        bot_id=bot_class().id, promote=promote, reason=reason,
        shadow_start=shadow_start, shadow_end=shadow_end,
        champion_metrics=champ, challenger_metrics=chal,
    )


def promote_from_tuning(
    proposal,  # walkforward.TuningProposal
    bot_class: type[Bot],
    panel: dict[str, pd.DataFrame],
    shadow_start: datetime,
    shadow_end: datetime,
    **kwargs,
) -> ShadowResult | None:
    """Shadow-validate an accepted Tier-A proposal's params vs the champion. Returns
    None when there is nothing to promote (rejected proposal / no proposed params)."""
    if not proposal.accepted or not proposal.proposed_params:
        return None
    return shadow_compare(
        bot_class, panel, proposal.current_params, proposal.proposed_params,
        shadow_start, shadow_end, **kwargs,
    )
