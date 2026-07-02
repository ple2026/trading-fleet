"""Event-driven, point-in-time backtest engine with a walk-forward harness.

Design rules baked in (the anti-overfitting constitution):
  - No lookahead: bots only ever see bars dated <= the simulated `now`.
  - T+1 execution: a signal generated on day D fills at day D+1's open.
  - Slippage floor: min 5 bps equities on every fill, always.

This module is the single-pass simulator. The anchored walk-forward harness that
wraps it (fit -> validate -> roll, out-of-sample metrics only) lives in
`walkforward.py`; in-sample results are never reported from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .bot import Bot, MarketContext
from .data import slice_pit
from .journal import Journal
from .kelly import size_order
from .regime import classify_price_regime
from .types import FleetSignal, RegimeState

SLIPPAGE_BPS_EQUITY = 5.0
TRADING_DAYS = 252
# Unleveraged $10k cash book: total gross exposure (longs + shorts) may not exceed
# equity. Without this a bot that shorts many names can drive the account negative.
MAX_GROSS = 1.0


@dataclass
class Position:
    symbol: str
    qty: float
    entry_px: float
    entry_at: datetime
    stop: float
    target: float
    side: str
    max_favorable_bps: float = 0.0
    max_adverse_bps: float = 0.0
    # Regime label at entry — Tier C reads this off closed positions to measure
    # each bot's regime-conditional edge.
    entry_regime: str = ""
    # The decision snapshot the entry signal carried. Tier B replays these features
    # against the trade's outcome to find recurring failure modes.
    entry_features: dict = field(default_factory=dict)
    thesis: str = ""


@dataclass
class Trade:
    symbol: str
    side: str
    entry_at: datetime
    exit_at: datetime
    entry_px: float
    exit_px: float
    qty: float
    pnl_usd: float
    r_multiple: float
    exit_reason: str
    max_favorable_bps: float
    max_adverse_bps: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return compute_metrics(self.equity_curve, self.trades)


def _apply_slippage(px: float, side: str, bps: float = SLIPPAGE_BPS_EQUITY) -> float:
    adj = px * bps / 10_000
    return px + adj if side == "buy" else px - adj


def _close_cash(cash: float, pos: "Position", fill: float) -> tuple[float, float]:
    """Apply a closing fill's cash effect. Returns (pnl, new_cash).

    Long close sells shares back into cash; short close buys shares to cover,
    paying out of cash. Consistent with entry (long pays, short receives), so a
    round-trip's net cash change equals its P&L — no phantom notional.
    """
    if pos.side == "buy":
        pnl = (fill - pos.entry_px) * pos.qty
        cash += fill * pos.qty
    else:
        pnl = (pos.entry_px - fill) * pos.qty
        cash -= fill * pos.qty
    return pnl, cash


def _regime_payload(regime: RegimeState) -> dict:
    """Flatten a RegimeState into the JSON blob the journal stores per decision."""
    return {"label": regime.label.value, "conviction": regime.conviction, **regime.vector}


def run_backtest(
    bot: Bot,
    panel: dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
    starting_equity: float = 10_000.0,
    benchmark: str = "SPY",
    journal: Journal | None = None,
) -> BacktestResult:
    """Simulate one bot over [start, end] on a fixed price panel.

    `panel` must include the benchmark symbol for regime classification. Bots that
    need fundamentals/calendar/macro receive them as None here — a stock-only
    smoke run — and their concrete data planes wire those in for full backtests.

    When a `journal` is supplied, every signal is recorded with its disposition
    (taken / rejected_*) — including the counterfactuals the improvement engine
    needs: signals blocked because the regime was unfavorable are scanned and
    logged as `rejected_regime` even though they never trade. Every closed
    position is journaled with MFE/MAE, exit reason, and entry regime.
    """
    all_dates = sorted(
        {d for df in panel.values() for d in df.index if start <= d <= end}
    )
    spy = panel.get(benchmark)
    if spy is None:
        raise ValueError(f"panel must include benchmark '{benchmark}'")

    cash = starting_equity
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    curve: dict[datetime, float] = {}

    def _record_trade(
        pos: Position, exit_fill: float, pnl: float, reason: str, exit_at: datetime
    ) -> None:
        """Build the Trade record, append it, and journal the position close."""
        per_share_risk = abs(pos.entry_px - pos.stop) or 1e-9
        r_multiple = pnl / (per_share_risk * pos.qty)
        trades.append(Trade(
            symbol=pos.symbol, side=pos.side, entry_at=pos.entry_at, exit_at=exit_at,
            entry_px=pos.entry_px, exit_px=exit_fill, qty=pos.qty, pnl_usd=pnl,
            r_multiple=r_multiple, exit_reason=reason,
            max_favorable_bps=pos.max_favorable_bps, max_adverse_bps=pos.max_adverse_bps,
        ))
        if journal is not None:
            journal.record_position_close({
                "bot_id": bot.id, "symbol": pos.symbol, "opened_at": pos.entry_at,
                "closed_at": exit_at, "entry_px": pos.entry_px, "exit_px": exit_fill,
                "qty": pos.qty, "max_favorable_bps": pos.max_favorable_bps,
                "max_adverse_bps": pos.max_adverse_bps, "exit_reason": reason,
                "pnl_usd": pnl, "r_multiple": r_multiple, "side": pos.side,
                "entry_regime": pos.entry_regime,
                "entry_features": pos.entry_features, "thesis": pos.thesis,
            })

    for i, today in enumerate(all_dates):
        pit = slice_pit(panel, today)
        regime = classify_price_regime(spy, today)

        # --- mark-to-market + intrabar stop/target using today's H/L ---
        for sym, pos in list(positions.items()):
            bar = panel[sym].loc[panel[sym].index == today]
            if bar.empty:
                continue
            hi, lo, close = float(bar["h"].iloc[0]), float(bar["l"].iloc[0]), float(bar["c"].iloc[0])
            fav = (hi - pos.entry_px) / pos.entry_px if pos.side == "buy" else (pos.entry_px - lo) / pos.entry_px
            adv = (pos.entry_px - lo) / pos.entry_px if pos.side == "buy" else (hi - pos.entry_px) / pos.entry_px
            pos.max_favorable_bps = max(pos.max_favorable_bps, fav * 10_000)
            pos.max_adverse_bps = max(pos.max_adverse_bps, adv * 10_000)

            exit_px = exit_reason = None
            if pos.side == "buy":
                if lo <= pos.stop:
                    exit_px, exit_reason = pos.stop, "stop"
                elif hi >= pos.target:
                    exit_px, exit_reason = pos.target, "target"
            else:
                if hi >= pos.stop:
                    exit_px, exit_reason = pos.stop, "stop"
                elif lo <= pos.target:
                    exit_px, exit_reason = pos.target, "target"

            if exit_px is not None:
                fill = _apply_slippage(exit_px, "sell" if pos.side == "buy" else "buy")
                pnl, cash = _close_cash(cash, pos, fill)
                _record_trade(pos, fill, pnl, exit_reason, today)
                del positions[sym]

        # --- management actions from the bot (trend-break, time stops, etc.) ---
        ctx = MarketContext(
            now=today, prices=pit, regime=regime, equity_usd=cash,
            open_symbols=set(positions.keys()),
        )
        for action in bot.manage(ctx):
            if action.kind == "exit" and action.symbol in positions:
                pos = positions[action.symbol]
                bar = panel[pos.symbol].loc[panel[pos.symbol].index == today]
                if bar.empty:
                    continue
                px = _apply_slippage(float(bar["c"].iloc[0]), "sell" if pos.side == "buy" else "buy")
                pnl, cash = _close_cash(cash, pos, px)
                _record_trade(pos, px, pnl, action.reason, today)
                del positions[action.symbol]

        # --- new entries: scan today, fill at NEXT day's open (T+1) ---
        may = bot.may_trade(regime)
        regime_json = _regime_payload(regime) if journal is not None else None

        def _reject(sig: FleetSignal, reason_code: str, why: str) -> None:
            if journal is not None:
                journal.record_signal(sig, reason_code, reject_reason=why, regime=regime_json)

        # Scan even when the regime gate is shut IFF we are journaling, so the
        # improvement engine sees what the bot *would* have done (Tier C data).
        if (may or journal is not None) and i + 1 < len(all_dates):
            next_day = all_dates[i + 1]
            signals: list[FleetSignal] = bot.scan(ctx)
            # Rank by confidence so that when we hit a cap we keep the bot's best
            # ideas, not whatever happened to be iterated first.
            signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

            if not may:
                for sig in signals:
                    _reject(sig, "rejected_regime",
                            f"regime {regime.label.value} not in favorable set")
            else:
                equity_now = cash + sum(
                    (float(panel[s].loc[panel[s].index == today, "c"].iloc[0])
                     * (p.qty if p.side == "buy" else -p.qty))
                    for s, p in positions.items()
                    if today in panel[s].index
                )
                gross_open = sum(p.entry_px * p.qty for p in positions.values())
                for sig in signals:
                    if len(positions) >= bot.max_concurrent:
                        _reject(sig, "rejected_cap", "max_concurrent positions held")
                        continue
                    if sig.symbol in positions:
                        _reject(sig, "rejected_dup", "already holding symbol")
                        continue
                    order = size_order(sig, cash)
                    if order is None:
                        _reject(sig, "rejected_risk", "no risk budget (size_order=None)")
                        continue
                    nb = panel.get(sig.symbol)
                    if nb is None or next_day not in nb.index:
                        _reject(sig, "rejected_nodata", "no next-day bar to fill")
                        continue
                    fill = _apply_slippage(float(nb.loc[next_day, "o"]), sig.side)
                    notional = fill * order.qty
                    # Unleveraged gross-exposure cap (longs + shorts <= equity).
                    if gross_open + notional > equity_now * MAX_GROSS:
                        _reject(sig, "rejected_gross", "gross-exposure cap")
                        continue
                    if sig.side == "buy":
                        if notional > cash:
                            _reject(sig, "rejected_cash", "insufficient cash")
                            continue
                        cash -= notional          # pay for the shares
                    else:
                        cash += notional          # short sale: receive proceeds
                    gross_open += notional
                    positions[sig.symbol] = Position(
                        symbol=sig.symbol, qty=order.qty, entry_px=fill, entry_at=next_day,
                        stop=sig.stop_loss, target=sig.take_profit, side=sig.side,
                        entry_regime=regime.label.value,
                        entry_features=dict(sig.features), thesis=sig.thesis,
                    )
                    if journal is not None:
                        journal.record_signal(sig, "taken", regime=regime_json)

        # --- record equity: cash + long marks - short liabilities ---
        mkt = 0.0
        for sym, pos in positions.items():
            bar = panel[sym].loc[panel[sym].index == today]
            if not bar.empty:
                close = float(bar["c"].iloc[0])
                mkt += close * pos.qty if pos.side == "buy" else -close * pos.qty
        curve[today] = cash + mkt

    return BacktestResult(equity_curve=pd.Series(curve), trades=trades)


def compute_metrics(equity: pd.Series, trades: list[Trade]) -> dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    years = len(equity) / TRADING_DAYS
    # Account ruin: if equity ever hits zero/negative, CAGR is -100% (total loss),
    # not a complex number from a fractional power of a negative base.
    if equity.iloc[-1] <= 0 or years <= 0:
        cagr = -1.0
    else:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    sharpe = (rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0
    cummax = equity.cummax()
    max_dd = float(((equity - cummax) / cummax).min())
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_win = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    return {
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd * 100, 2),
        "trades": len(trades),
        "hit_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "avg_mfe_capture": round(
            np.mean([t.r_multiple for t in trades]) if trades else 0.0, 2
        ),
    }
