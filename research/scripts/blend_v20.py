"""Step 1 (pre-registered): blend v20 + MACRO + index-beta and ask whether the
blend beats v20 alone on a RISK-ADJUSTED basis.

The handoff's meta-finding is that returns come from time-in-market x beta x timing
x risk-sizing, and that MACRO (corr ~ -0.15 to SPY) is v20's natural complement
because v20 is "QQQ-plus with -50%+ possible DD." This script tests that claim on
real curves:

  1. corr(v20, MACRO), corr(v20, SPY), corr(MACRO, SPY) on daily returns.
  2. The allocator's own blend (risk-parity + quarter-Kelly OOS-Sharpe tilt +
     [10%,40%] caps + corr freeze), rebalanced monthly, point-in-time.
  3. A fixed-weight frontier, for context (the allocator picks ONE point; the
     frontier shows the tradeoff curve it sits on).
  4. A VOL-MATCHED comparison: lever the blend to v20's realized vol. This is the
     honest "better than v20?" number -- a higher-Sharpe blend levered to v20's
     risk should out-CAGR v20 at equal-or-lower drawdown.

Inputs:
    --v20-csv   Trading Agent/backtest-output/v20/equity-layers.csv (col v20_equity)
    --macro-csv data/macro_oos_equity.csv (from gen_macro_curve.py; col macro_equity)
    SPY is fetched from Tiingo.

One run, results stand.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from research.fleet.allocator import allocate

TRADING_DAYS = 252
V20_DEFAULT = "/Users/phuongle/Documents/Trading Agent/backtest-output/v20/equity-layers.csv"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _curve_metrics(curve: pd.Series) -> dict:
    """CAGR/Sharpe/MaxDD/vol on a daily equity curve."""
    rets = curve.pct_change().dropna()
    years = len(curve) / TRADING_DAYS
    cummax = curve.cummax()
    dd = float(((curve - cummax) / cummax).min())
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0
    cagr = float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1)
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": dd, "vol": vol,
            "final": float(curve.iloc[-1])}


def _fmt(name: str, m: dict) -> str:
    return (f"{name:<22}{m['cagr']*100:>8.1f}{m['sharpe']:>8.2f}"
            f"{m['maxdd']*100:>9.1f}{m['vol']*100:>8.1f}")


def _equity_from_returns(rets: pd.Series, seed: float = 100_000.0) -> pd.Series:
    return seed * (1 + rets).cumprod()


def _load_spy(start, end) -> pd.Series:
    from research.fleet.data import TiingoProvider
    df = TiingoProvider().daily_bars("SPY", start, end)
    return df["c"].rename("SPY")


def monthly_rebalance_dates(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First trading day of each month in the index."""
    s = pd.Series(idx, index=idx)
    firsts = s.groupby([idx.year, idx.month]).first()
    return list(firsts.values)


def allocator_blend(returns: pd.DataFrame, warmup: int = 90) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly-rebalanced allocator blend, point-in-time.

    At each month start we call allocate() on returns observed UP TO that day only
    (expanding window -> no look-ahead), hold the weights through the month, and
    accumulate the blended daily return. Returns (blended_daily_returns, weight_log).
    """
    dates = returns.index
    rebal = [d for d in monthly_rebalance_dates(dates) if returns.index.get_loc(d) >= warmup]
    weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)  # equal until first rebal
    blended = pd.Series(0.0, index=dates)
    wlog = {}
    rebal_set = set(rebal)
    for i, d in enumerate(dates):
        if d in rebal_set:
            hist = returns.loc[:dates[i - 1]] if i > 0 else returns.iloc[:0]
            if len(hist) >= warmup:
                alloc = allocate(hist, total_usd=1.0, at=d.to_pydatetime())
                weights = pd.Series(alloc.weights).reindex(returns.columns).fillna(0.0)
                wlog[d] = {**alloc.weights, "_frozen": alloc.frozen,
                           "_fleet_corr": round(alloc.correlation.fleet_avg, 3)}
        blended.loc[d] = float((weights * returns.loc[d]).sum())
    return blended, pd.DataFrame(wlog).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v20-csv", default=V20_DEFAULT)
    ap.add_argument("--macro-csv", default="data/macro_oos_equity.csv")
    ap.add_argument("--out", default="data/blend_report.csv")
    args = ap.parse_args()
    _load_env()

    # ---- load the three sleeves as daily equity ----------------------------
    v20 = pd.read_csv(args.v20_csv, encoding="utf-8-sig")
    v20["date"] = pd.to_datetime(v20["date"], utc=False, format="mixed")
    v20 = v20.set_index("date")["v20_equity"].sort_index()

    macro = pd.read_csv(args.macro_csv)
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date")["macro_equity"].sort_index()

    spy = _load_spy(v20.index.min().to_pydatetime(), v20.index.max().to_pydatetime())
    spy.index = pd.to_datetime(spy.index)

    # ---- align on common trading days --------------------------------------
    eq = pd.DataFrame({"v20": v20, "MACRO": macro, "SPY": spy}).dropna()
    print(f"Common span: {eq.index.min().date()} .. {eq.index.max().date()} "
          f"({len(eq)} trading days)")
    rets = eq.pct_change().dropna()

    # ---- per-sleeve stats + correlations -----------------------------------
    print(f"\n{'sleeve':<22}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    for c in ["v20", "MACRO", "SPY"]:
        print(_fmt(c, _curve_metrics(eq[c] / eq[c].iloc[0])))

    corr = rets.corr()
    print("\nDaily-return correlations:")
    print(f"  corr(v20, MACRO) = {corr.loc['v20','MACRO']:+.3f}")
    print(f"  corr(v20, SPY)   = {corr.loc['v20','SPY']:+.3f}")
    print(f"  corr(MACRO, SPY) = {corr.loc['MACRO','SPY']:+.3f}")

    v20m = _curve_metrics(eq["v20"] / eq["v20"].iloc[0])

    # ---- (2) the pre-registered allocator blend ----------------------------
    blended, wlog = allocator_blend(rets)
    blend_curve = _equity_from_returns(blended.loc[rets.index])
    bm = _curve_metrics(blend_curve)
    print("\n=== Allocator blend (risk-parity + quarter-Kelly + caps, monthly) ===")
    if not wlog.empty:
        avg_w = wlog[["v20", "MACRO", "SPY"]].astype(float).mean()
        print(f"  avg weights: v20 {avg_w['v20']:.2f} | MACRO {avg_w['MACRO']:.2f} | "
              f"SPY {avg_w['SPY']:.2f}   frozen months: {int(wlog['_frozen'].sum())}/{len(wlog)}")
    print(f"\n{'strategy':<22}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    print(_fmt("v20 alone", v20m))
    print(_fmt("allocator blend", bm))

    # ---- (3) fixed-weight frontier (context) -------------------------------
    print("\n=== Fixed-weight frontier (static, whole-span) ===")
    print(f"{'w_v20 / w_MACRO / w_SPY':<22}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    grid = [
        (1.00, 0.00, 0.00), (0.70, 0.20, 0.10), (0.60, 0.30, 0.10),
        (0.50, 0.30, 0.20), (0.40, 0.40, 0.20), (0.34, 0.33, 0.33),
    ]
    best = None
    for wv, wm, ws in grid:
        r = wv * rets["v20"] + wm * rets["MACRO"] + ws * rets["SPY"]
        m = _curve_metrics(_equity_from_returns(r))
        tag = f"{wv:.2f}/{wm:.2f}/{ws:.2f}"
        print(_fmt(tag, m))
        if best is None or m["sharpe"] > best[1]["sharpe"]:
            best = (tag, m, (wv, wm, ws))

    # ---- (3b) v20 + MACRO only (drop SPY — it is +0.63 corr, low Sharpe) ---
    print("\n=== v20 + MACRO only (no SPY) ===")
    print(f"{'w_v20 / w_MACRO':<22}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    for wv in (0.80, 0.70, 0.60, 0.50):
        r = wv * rets["v20"] + (1 - wv) * rets["MACRO"]
        m = _curve_metrics(_equity_from_returns(r))
        print(_fmt(f"{wv:.2f}/{1-wv:.2f}", m))
        if m["sharpe"] > best[1]["sharpe"]:
            best = (f"{wv:.2f}/{1-wv:.2f} (v20+MACRO)", m, (wv, 1 - wv, 0.0))

    # ---- (3c) in-sample max-Sharpe tangency (the improvement CEILING) ------
    # NB: fit on the whole span => look-ahead. This is the ex-post ceiling, NOT a
    # deployable weight. The honest deployable number is the monthly allocator above.
    from scipy.optimize import minimize
    mu = rets[["v20", "MACRO", "SPY"]].mean().values * TRADING_DAYS
    cov = rets[["v20", "MACRO", "SPY"]].cov().values * TRADING_DAYS

    def neg_sharpe(w):
        port_mu = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        return -port_mu / port_vol if port_vol > 0 else 0.0

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * 3
    sol = minimize(neg_sharpe, np.array([0.5, 0.3, 0.2]), bounds=bnds,
                   constraints=cons, method="SLSQP")
    tw = sol.x
    tr = tw[0] * rets["v20"] + tw[1] * rets["MACRO"] + tw[2] * rets["SPY"]
    tm = _curve_metrics(_equity_from_returns(tr))
    print(f"\n=== In-sample max-Sharpe tangency (CEILING, look-ahead) ===")
    print(f"  weights: v20 {tw[0]:.2f} | MACRO {tw[1]:.2f} | SPY {tw[2]:.2f}")
    print(f"{'strategy':<22}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    print(_fmt("v20 alone", v20m))
    print(_fmt("tangency (unlevered)", tm))
    ktan = v20m["vol"] / tm["vol"]
    tmlev = _curve_metrics(_equity_from_returns(ktan * tr))
    print(_fmt(f"tangency levered x{ktan:.2f}", tmlev))

    # ---- (4) vol-matched: lever the max-Sharpe blend to v20's vol ----------
    tag, m, (wv, wm, ws) = best
    r = wv * rets["v20"] + wm * rets["MACRO"] + ws * rets["SPY"]
    k = v20m["vol"] / m["vol"]                     # leverage to match v20 vol
    levered = _curve_metrics(_equity_from_returns(k * r))
    print(f"\n=== Vol-matched: best-Sharpe blend {tag} levered x{k:.2f} to v20's vol ===")
    print(f"{'strategy':<22}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    print(_fmt("v20 alone", v20m))
    print(_fmt(f"blend levered x{k:.2f}", levered))
    print("\n(Leverage ignores financing cost; Sharpe is the leverage-invariant "
          "verdict. A higher blend Sharpe => more CAGR per unit of drawdown than v20.)")

    # ---- (5) sub-period robustness: is the edge structural or one crisis? ---
    print("\n=== Sub-period robustness: v20 vs 50/50 v20+MACRO (unlevered) ===")
    print(f"{'window':<16}{'v20 Shrp':>9}{'v20 DD%':>9}{'blend Shrp':>11}{'blend DD%':>10}"
          f"{'corr':>7}")
    windows = [("2007-2009", "2007", "2009"), ("2010-2014", "2010", "2014"),
               ("2015-2019", "2015", "2019"), ("2020-2022", "2020", "2022"),
               ("2023-2026", "2023", "2026")]
    for label, lo, hi in windows:
        sub = rets.loc[f"{lo}-01-01":f"{hi}-12-31"]
        if len(sub) < 30:
            continue
        vm = _curve_metrics(_equity_from_returns(sub["v20"]))
        bl = 0.5 * sub["v20"] + 0.5 * sub["MACRO"]
        bmv = _curve_metrics(_equity_from_returns(bl))
        c = sub["v20"].corr(sub["MACRO"])
        print(f"{label:<16}{vm['sharpe']:>9.2f}{vm['maxdd']*100:>9.1f}"
              f"{bmv['sharpe']:>11.2f}{bmv['maxdd']*100:>10.1f}{c:>7.2f}")

    # ---- persist the blended curve for later use ---------------------------
    out = pd.DataFrame({"v20": eq["v20"], "MACRO": eq["MACRO"], "SPY": eq["SPY"],
                        "allocator_blend": blend_curve.reindex(eq.index)})
    out.to_csv(args.out, index_label="date")
    print(f"\nWrote curves -> {args.out}")


if __name__ == "__main__":
    main()
