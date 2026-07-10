"""Regime-rotation portfolio experiment: stay invested in good tape, rotate in bad.

The one approach with a century of evidence behind it isn't out-picking stocks —
it's owning the market while it is healthy and stepping aside (or into defensives)
when it breaks. This script tests exactly that, honestly:

PRE-REGISTERED RULES — decided before running, zero parameters fit to this window
(the regime classifier predates this experiment and is used unmodified):

  A  hold    SPY buy & hold                          (the bar to beat)
  B  cash    SPY unless regime = RISK_OFF -> cash    (de-risk)
  C  rotate  SPY unless RISK_OFF -> 50/50 TLT + GLD  (rotate, don't hide)

Discipline:
  * No lookahead: the regime is measured at day t's close; the resulting weights
    earn day t+1's return.
  * 5 bps per side of turnover charged on every switch.
  * Run once over 2005-2024 (GFC, COVID, 2022 inflation bear). No post-hoc variant
    shopping — if these rules fail, that result stands.
  * Cash earns 0% here (T-bills averaged ~1.5%/yr over the window, so rule B is
    slightly UNDERSTATED — noted rather than modeled).

    python -m research.scripts.regime_portfolio
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from research.fleet.regime import classify_price_regime
from research.fleet.types import RegimeTag

SYMBOLS = ["SPY", "TLT", "GLD"]
DEFENSIVE = {"TLT": 0.5, "GLD": 0.5}
WARMUP = 200          # bars the regime classifier needs before it stops saying CHOP
COST_BPS = 5.0        # per side, on turnover

CRISES = [
    ("GFC 2008", "2007-10-01", "2009-06-30"),
    ("COVID 2020", "2020-01-01", "2020-12-31"),
    ("2022 bear", "2022-01-01", "2022-12-31"),
]


def target_weights(label: RegimeTag, mode: str) -> dict[str, float]:
    """Map a regime label to portfolio weights under one of the three rules."""
    if mode == "hold":
        return {"SPY": 1.0}
    if label == RegimeTag.RISK_OFF:
        return {} if mode == "cash" else dict(DEFENSIVE)
    return {"SPY": 1.0}


def run_rotation(
    panel: dict[str, pd.DataFrame], mode: str, cost_bps: float = COST_BPS,
    warmup: int = WARMUP,
) -> tuple[pd.Series, pd.DataFrame, int]:
    """Simulate one rule. Returns (equity curve, daily weights, switch count).

    Weights decided from the regime at close t apply to day t+1's close-to-close
    return (T+1, no lookahead). Turnover is charged at `cost_bps` per side.
    """
    spy = panel["SPY"]
    dates = spy.index[warmup:]
    rets = {s: panel[s]["c"].pct_change() for s in panel}

    equity = 1.0
    curve: dict[pd.Timestamp, float] = {}
    weights_hist: dict[pd.Timestamp, dict[str, float]] = {}
    w = {"SPY": 1.0}              # every rule starts invested
    pending: dict[str, float] | None = None
    n_switch = 0

    for t in dates:
        if pending is not None:
            turnover = sum(
                abs(pending.get(s, 0.0) - w.get(s, 0.0))
                for s in set(pending) | set(w)
            )
            if turnover > 1e-9:
                equity *= 1.0 - turnover * cost_bps / 10_000.0
                n_switch += 1
            w = pending
        day_ret = 0.0
        for s, wt in w.items():
            r = rets[s].get(t, np.nan)
            if not np.isnan(r):
                day_ret += wt * float(r)
        equity *= 1.0 + day_ret
        curve[t] = equity
        weights_hist[t] = dict(w)
        # Decide tomorrow's weights from today's close (point-in-time classifier).
        label = classify_price_regime(spy, t).label
        pending = target_weights(label, mode)

    return (
        pd.Series(curve),
        pd.DataFrame(weights_hist).T.fillna(0.0),
        n_switch,
    )


def _metrics(curve: pd.Series) -> dict[str, float]:
    rets = curve.pct_change().dropna()
    years = len(curve) / 252
    cummax = curve.cummax()
    return {
        "cagr": round((curve.iloc[-1] ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(252)), 2)
        if rets.std() > 0 else 0.0,
        "max_dd": round(float(((curve - cummax) / cummax).min()) * 100, 2),
    }


def _crisis_dd(curve: pd.Series, start: str, end: str) -> float:
    c = curve.loc[(curve.index >= start) & (curve.index <= end)]
    if len(c) < 2:
        return float("nan")
    cummax = c.cummax()
    return round(float(((c - cummax) / cummax).min()) * 100, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2004-01-01",
                    help="panel start (warmup consumes the first ~200 bars)")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    from research.fleet.data import default_provider, load_panel

    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    provider = default_provider()
    print(f"Data provider: {type(provider).__name__}")
    panel = load_panel(SYMBOLS, start, end, provider)
    missing = [s for s in SYMBOLS if s not in panel]
    if missing:
        print(f"ERROR: missing {missing}")
        return
    # Align all series to SPY's calendar so day returns line up.
    idx = panel["SPY"].index
    panel = {s: df.reindex(idx).ffill() for s, df in panel.items()}
    print(f"Panel: {idx.min().date()}..{idx.max().date()} ({len(idx)} bars; "
          f"first {WARMUP} consumed as classifier warmup)\n")

    rules = [("A hold", "hold"), ("B cash", "cash"), ("C rotate", "rotate")]
    results = {}
    print(f"{'rule':<10}{'CAGR%':>7}{'Sharpe':>8}{'MaxDD%':>9}{'switches':>10}{'%invested':>11}")
    print("-" * 55)
    for name, mode in rules:
        curve, wts, n_sw = run_rotation(panel, mode, args.cost_bps)
        m = _metrics(curve)
        pct_inv = round(float((wts.sum(axis=1) > 0).mean()) * 100, 1)
        results[name] = curve
        print(f"{name:<10}{m['cagr']:>7}{m['sharpe']:>8}{m['max_dd']:>9}"
              f"{n_sw:>10}{pct_inv:>10}%")

    print("\nDrawdown inside each crisis:")
    print(f"{'rule':<10}" + "".join(f"{c[0]:>13}" for c in CRISES))
    for name, curve in results.items():
        row = "".join(f"{_crisis_dd(curve, s, e):>12}%" for _, s, e in CRISES)
        print(f"{name:<10}{row}")

    print("\nPre-registered rules, one run, no variant shopping. Cash earns 0% here "
          "(T-bills would add ~1-1.5%/yr to rule B). 'Beating the market' claimed "
          "only on Sharpe / drawdown, never on raw CAGR.")


if __name__ == "__main__":
    main()
