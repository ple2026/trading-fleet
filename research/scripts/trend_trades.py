"""Show the ACTUAL trades of the TREND_NE sleeve and attribute P&L — then show,
concretely, why the v20+TREND_NE blend beats v20.

A "trade" here = a continuous run in one instrument on one side (long/short) until the
trend signal flips it. We rebuild the exact monthly weight path the sleeve trades,
split each instrument's holding into sign-runs, and total the P&L of each run.

    python -m research.scripts.trend_trades
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.fleet.data import TiingoProvider
from research.scripts.trend_sleeve import (
    INSTRUMENTS, LOOKBACKS, _target_weights,
)

TD = 252
EQUITIES = {"SPY", "QQQ", "IWM", "EFA", "EEM"}
UNIVERSE = [s for s in INSTRUMENTS if s not in EQUITIES]  # non-equity = TREND_NE
V20_CSV = "/Users/phuongle/Documents/Trading Agent/backtest-output/v20/equity-layers.csv"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def load_prices(symbols, start="2004-01-01", end="2026-07-01") -> pd.DataFrame:
    prov = TiingoProvider()
    from datetime import datetime
    s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    cols = {}
    for sym in symbols:
        try:
            cols[sym] = prov.daily_bars(sym, s, e)["c"]
        except Exception:
            pass
    return pd.DataFrame(cols).sort_index()


def weight_path(px, start="2007-01-01"):
    """Rebuild the monthly weight vector actually held each day (no lookahead)."""
    rets = px.pct_change()
    dates = px.index
    mk = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first()
    rebal = {d for d in mk.values if d >= pd.Timestamp(start)}
    W = pd.DataFrame(0.0, index=dates, columns=px.columns)
    w = pd.Series(0.0, index=px.columns)
    for i, d in enumerate(dates):
        if d < pd.Timestamp(start):
            continue
        if d in rebal:
            hpx, hret = px.loc[:dates[i - 1]], rets.loc[:dates[i - 1]]
            if len(hpx) > max(LOOKBACKS) + 5:
                w = _target_weights(hpx, hret)
        W.loc[d] = w
    return W.loc[W.index >= pd.Timestamp(start)]


def main() -> None:
    _load_env()
    px = load_prices(UNIVERSE)
    print(f"TREND_NE universe: {list(px.columns)}")
    W = weight_path(px)
    rets = px.pct_change().reindex(W.index)
    # daily P&L contribution per instrument = weight held * next-day return
    contrib = (W.shift(1) * rets).fillna(0.0)

    # ---- instrument-level attribution -------------------------------------
    print("\n=== Per-instrument P&L attribution (sum of daily weight*return) ===")
    print(f"{'inst':<6}{'total P&L%':>11}{'avg |wt|':>10}{'% days long':>12}{'% days short':>13}")
    tot = contrib.sum().sort_values(ascending=False)
    for s in tot.index:
        w = W[s]
        held = w[w.abs() > 1e-6]
        pl = tot[s] * 100
        avgw = held.abs().mean() if len(held) else 0.0
        plong = (w > 1e-6).mean() * 100
        pshort = (w < -1e-6).mean() * 100
        print(f"{s:<6}{pl:>11.1f}{avgw:>10.2f}{plong:>12.0f}{pshort:>13.0f}")
    print(f"{'TOTAL':<6}{tot.sum()*100:>11.1f}  (sum of contributions = sleeve gross return)")

    # ---- split each instrument into sign-run trades -----------------------
    trades = []
    for s in px.columns:
        w = W[s]
        sign = np.sign(w).replace(0, np.nan)
        grp = (sign != sign.shift()).cumsum()
        for _, idx in w.groupby(grp).groups.items():
            seg = w.loc[idx]
            sg = np.sign(seg.iloc[0])
            if sg == 0 or seg.abs().max() < 1e-6:
                continue
            pl = contrib.loc[idx, s].sum() * 100
            trades.append({"inst": s, "side": "LONG" if sg > 0 else "SHORT",
                           "start": idx[0].date(), "end": idx[-1].date(),
                           "months": max(1, len(idx) // 21), "pnl_pct": pl,
                           "avg_wt": seg.abs().mean()})
    tdf = pd.DataFrame(trades)
    print(f"\nTotal trades (sign-runs): {len(tdf)}   "
          f"winners {int((tdf.pnl_pct>0).sum())} / losers {int((tdf.pnl_pct<0).sum())}")
    print("\n=== 15 biggest WINNING trades ===")
    print(f"{'inst':<6}{'side':<6}{'start':>12}{'end':>12}{'mths':>6}{'wt':>6}{'P&L%':>8}")
    for _, t in tdf.sort_values("pnl_pct", ascending=False).head(15).iterrows():
        print(f"{t['inst']:<6}{t['side']:<6}{str(t['start']):>12}{str(t['end']):>12}"
              f"{t['months']:>6}{t['avg_wt']:>6.2f}{t['pnl_pct']:>8.1f}")
    print("\n=== 8 biggest LOSING trades ===")
    for _, t in tdf.sort_values("pnl_pct").head(8).iterrows():
        print(f"{t['inst']:<6}{t['side']:<6}{str(t['start']):>12}{str(t['end']):>12}"
              f"{t['months']:>6}{t['avg_wt']:>6.2f}{t['pnl_pct']:>8.1f}")

    # ---- WHY it beats v20: behaviour on v20's worst days ------------------
    v20 = pd.read_csv(V20_CSV, encoding="utf-8-sig")
    v20["date"] = pd.to_datetime(v20["date"], format="mixed")
    v20 = v20.set_index("date")["v20_equity"].sort_index()
    sleeve = (1 + contrib.sum(axis=1)).cumprod()
    j = pd.DataFrame({"v20": v20, "NE": sleeve}).dropna().pct_change().dropna()
    worst = j.nsmallest(40, "v20")
    print("\n=== On v20's 40 WORST days, what did TREND_NE do? ===")
    print(f"  v20 avg day: {worst['v20'].mean()*100:+.2f}%   "
          f"TREND_NE avg those same days: {worst['NE'].mean()*100:+.2f}%")
    print(f"  TREND_NE was UP on {int((worst['NE']>0).sum())}/40 of v20's worst days")
    best = j.nlargest(40, "v20")
    print(f"  (on v20's 40 BEST days TREND_NE avg: {best['NE'].mean()*100:+.2f}% — "
          f"it doesn't give the upside back)")
    print(f"\n  full-period daily corr(v20, TREND_NE) = {j['v20'].corr(j['NE']):+.3f}")


if __name__ == "__main__":
    main()
