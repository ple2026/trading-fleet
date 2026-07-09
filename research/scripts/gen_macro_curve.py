"""Regenerate the MACRO out-of-sample daily equity curve (walk-forward, FRED-driven).

This is the sleeve the handoff calls "the survivor" (18y OOS 2007-24, Sharpe 0.88,
corr to SPY -0.15). We need its *daily* curve to blend against the v20 book, so
this script runs the same walk-forward the handoff verdict came from and writes the
concatenated OOS equity to CSV.

    python -m research.scripts.gen_macro_curve --start 2007-01-01 --end 2026-07-01

Writes:  data/macro_oos_equity.csv   (date, macro_equity)
Prints:  OOS metrics on the full span AND on the 2007-24 sub-window (to confirm we
         reproduce the handoff's Sharpe 0.88 / MaxDD -6.9%).
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from research.fleet.bots.macro import Macro
from research.fleet.data import TiingoProvider
from research.fleet.walkforward import walk_forward

# MACRO's expression universe + SPY (regime benchmark). Same names backtest.py uses.
UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "XLE", "OIH", "SMH", "XLF", "KRE", "GLD", "GDX", "XLK",
    "TLT", "IEF", "SLV", "UUP", "DBC", "USO", "EEM", "EWJ",
]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _resilient_panel(symbols, start, end):
    provider = TiingoProvider()
    panel, skipped = {}, []
    for s in symbols:
        try:
            df = provider.daily_bars(s, start, end)
        except Exception as e:  # noqa: BLE001
            skipped.append((s, str(e)[:60]))
            continue
        if len(df) > 200:
            panel[s] = df
        else:
            skipped.append((s, f"only {len(df)} bars"))
    return panel, skipped


def _metrics(curve: pd.Series, span: str) -> dict:
    if curve is None or curve.empty or len(curve) < 2:
        return {}
    rets = curve.pct_change().dropna()
    years = len(curve) / 252
    cummax = curve.cummax()
    dd = ((curve - cummax) / cummax).min()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {
        "span": span,
        "start": curve.index.min().date(),
        "end": curve.index.max().date(),
        "days": len(curve),
        "cagr_pct": round(((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(sharpe, 2),
        "maxdd_pct": round(float(dd) * 100, 2),
        "final": round(float(curve.iloc[-1]), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--out", default="data/macro_oos_equity.csv")
    ap.add_argument("--train-years", type=float, default=3.0)
    ap.add_argument("--test-months", type=int, default=6)
    args = ap.parse_args()
    _load_env()

    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    print(f"Loading Tiingo ETF panel {start.date()}..{end.date()} ({len(UNIVERSE)} names)…")
    panel, skipped = _resilient_panel(UNIVERSE, start, end)
    print(f"  loaded {len(panel)}; skipped {len(skipped)}: {skipped}")
    if "SPY" not in panel:
        raise SystemExit("SPY failed to load — cannot classify regime.")

    from research.fleet.macro_data import MacroPanel
    print("Building FRED macro panel (point-in-time)…")
    macro = MacroPanel.from_fred(start, end)

    print(f"Walk-forward MACRO ({args.train_years}y train -> {args.test_months}m test, anchored)…")
    res = walk_forward(
        Macro, panel, start, end,
        train_years=args.train_years, test_months=args.test_months,
        anchored=True, macro=macro,
    )
    curve = res.oos_equity
    if curve.empty:
        raise SystemExit("MACRO produced no OOS trades.")

    curve.to_frame("macro_equity").to_csv(args.out, index_label="date")
    print(f"\nWrote {len(curve)} rows -> {args.out}")

    full = _metrics(curve, "full OOS")
    sub = _metrics(curve.loc[curve.index <= pd.Timestamp("2024-12-31")], "OOS 2007-24 (handoff)")
    print(f"\n{'span':<26}{'start':>12}{'end':>12}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'final':>14}")
    for m in (full, sub):
        if m:
            print(f"{m['span']:<26}{str(m['start']):>12}{str(m['end']):>12}"
                  f"{m['cagr_pct']:>8}{m['sharpe']:>8}{m['maxdd_pct']:>9}{m['final']:>14,.0f}")
    print(f"\nOOS trades: {len(res.trades)} across {len(res.windows)} windows")
    print("Handoff reference: 2007-24 Sharpe 0.88, MaxDD -6.9%, CAGR 3.5% unlevered.")


if __name__ == "__main__":
    main()
