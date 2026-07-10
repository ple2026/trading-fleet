"""Champion/challenger shadow-promotion tests (§7c).

Pin the gate: a challenger is promoted only when it beats the champion on the
held-out shadow window by the Sharpe margin, without worse drawdown, on enough
trades. Everything else holds the champion.
"""

from __future__ import annotations

from datetime import datetime

from research.fleet.bots.macro import Macro
from research.fleet.data import SyntheticProvider, load_panel
from research.fleet.promotion import (
    MIN_SHADOW_TRADES,
    _promotion_gate,
    shadow_compare,
    split_date,
)
from research.fleet.walkforward import MIN_SHARPE_GAIN

U = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "UUP", "DBC", "EEM"]
START = datetime(2019, 1, 1)
END = datetime(2024, 12, 31)


def _gate(champ, chal):
    return _promotion_gate(
        champ, chal, min_trades=MIN_SHADOW_TRADES,
        sharpe_margin=MIN_SHARPE_GAIN, dd_worsening=0.10,
    )


def test_gate_promotes_clear_winner():
    champ = {"sharpe": 0.5, "max_dd": -10.0, "trades": 40}
    chal = {"sharpe": 0.9, "max_dd": -9.0, "trades": 40}   # +0.4 Sharpe, better DD
    promote, reason = _gate(champ, chal)
    assert promote and "PROMOTE" in reason


def test_gate_holds_on_thin_sample():
    champ = {"sharpe": 0.5, "max_dd": -10.0, "trades": 40}
    chal = {"sharpe": 2.0, "max_dd": -5.0, "trades": 3}    # great, but too few trades
    promote, reason = _gate(champ, chal)
    assert not promote and "shadow trades" in reason


def test_gate_holds_on_insufficient_edge():
    champ = {"sharpe": 0.5, "max_dd": -10.0, "trades": 40}
    chal = {"sharpe": 0.6, "max_dd": -10.0, "trades": 40}  # +0.1 < 0.2 margin
    promote, reason = _gate(champ, chal)
    assert not promote and "Sharpe gain" in reason


def test_gate_holds_on_worse_drawdown():
    champ = {"sharpe": 0.5, "max_dd": -10.0, "trades": 40}
    chal = {"sharpe": 0.9, "max_dd": -13.0, "trades": 40}  # +0.4 Sharpe but DD +30%
    promote, reason = _gate(champ, chal)
    assert not promote and "max-DD" in reason


def test_split_date_carves_off_holdout():
    panel = load_panel(U, START, END, SyntheticProvider(seed=5))
    split = split_date(panel, END, holdout_days=252)
    assert split is not None and split < END
    # Everything before the split is available for tuning; the split is ~1y of bars in.
    dates = sorted({d for df in panel.values() for d in df.index})
    assert dates[0] < split <= dates[-1]


def test_shadow_compare_runs_and_decides():
    panel = load_panel(U, START, END, SyntheticProvider(seed=5))
    split = split_date(panel, END, holdout_days=252)
    champion = {p.name: p.default for p in Macro().param_space()}
    # A challenger identical to the champion cannot beat it -> must HOLD.
    res = shadow_compare(Macro, panel, champion, dict(champion), split, END)
    assert res.bot_id == "macro"
    assert res.promote is False           # identical params never clear the margin
    assert res.champion_metrics == res.challenger_metrics
    assert res.shadow_start == split and res.shadow_end == END
