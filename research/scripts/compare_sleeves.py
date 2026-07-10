"""Honest comparison of defensive sleeves for the v20 book: CASH vs MACRO vs TREND.

The Step-1 finding was that MACRO barely beats CASH as v20's defensive half. This
asks whether the Step-2 TREND sleeve (vol-targeted long/short managed-futures) is a
materially better defensive half. The fair test is VOL-MATCHED: at the same blend
risk, which defensive sleeve delivers the most CAGR and the least drawdown?

    python -m research.scripts.compare_sleeves
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
V20 = "/Users/phuongle/Documents/Trading Agent/backtest-output/v20/equity-layers.csv"
RF_ANNUAL = 0.02


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def M(x: pd.Series) -> dict:
    """Metrics on a daily-return series."""
    c = 100_000 * (1 + x).cumprod()
    r = c.pct_change().dropna()
    yrs = len(c) / TRADING_DAYS
    dd = float(((c - c.cummax()) / c.cummax()).min())
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    shrp = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    return {"cagr": float((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1),
            "sharpe": shrp, "maxdd": dd, "vol": vol}


def row(name: str, m: dict) -> str:
    return (f"{name:<26}{m['cagr']*100:>8.1f}{m['sharpe']:>8.2f}"
            f"{m['maxdd']*100:>9.1f}{m['vol']*100:>8.1f}")


def vol_matched_blend(v20r: pd.Series, dr: pd.Series, target_vol: float) -> tuple:
    """Find w on v20 (1-w on the defensive sleeve) whose blend vol ~ target_vol."""
    best = None
    for w in np.linspace(0.0, 1.0, 101):
        b = w * v20r + (1 - w) * dr
        m = M(b)
        err = abs(m["vol"] - target_vol)
        if best is None or err < best[0]:
            best = (err, w, m, b)
    return best[1], best[2], best[3]


def tangency(v20r: pd.Series, dr: pd.Series) -> tuple:
    """Long-only 2-asset max-Sharpe weights (in-sample ceiling)."""
    from scipy.optimize import minimize_scalar
    mu = np.array([v20r.mean(), dr.mean()]) * TRADING_DAYS
    cov = np.cov(np.vstack([v20r.values, dr.values])) * TRADING_DAYS

    def negS(w):
        wv = np.array([w, 1 - w])
        pv = np.sqrt(wv @ cov @ wv)
        return -(wv @ mu) / pv if pv > 0 else 0.0
    r = minimize_scalar(negS, bounds=(0, 1), method="bounded")
    w = r.x
    return w, M(w * v20r + (1 - w) * dr)


def main() -> None:
    _load_env()
    # ---- load sleeves ------------------------------------------------------
    v20 = pd.read_csv(V20, encoding="utf-8-sig")
    v20["date"] = pd.to_datetime(v20["date"], format="mixed")
    v20 = v20.set_index("date")["v20_equity"].sort_index()
    macro = pd.read_csv("data/macro_oos_equity.csv")
    macro["date"] = pd.to_datetime(macro["date"]); macro = macro.set_index("date")["macro_equity"]
    trend = pd.read_csv("data/trend_oos_equity.csv")
    trend["date"] = pd.to_datetime(trend["date"]); trend = trend.set_index("date")["trend_equity"]
    trend_ne = pd.read_csv("data/trend_noeq_oos_equity.csv")
    trend_ne["date"] = pd.to_datetime(trend_ne["date"])
    trend_ne = trend_ne.set_index("date")["trend_equity"]

    eq = pd.DataFrame({"v20": v20, "MACRO": macro, "TREND": trend, "TREND_NE": trend_ne}).dropna()
    r = eq.pct_change().dropna()
    rf = pd.Series(RF_ANNUAL / TRADING_DAYS, index=r.index)
    print(f"Common span: {eq.index.min().date()} .. {eq.index.max().date()} ({len(eq)} days)\n")

    # ---- standalone + correlations ----------------------------------------
    print(f"{'sleeve':<26}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    for c in ["v20", "MACRO", "TREND", "TREND_NE"]:
        print(row(c, M(r[c])))
    print("\ncorrelation matrix (daily returns):")
    print(r.corr().round(3).to_string())

    v20m = M(r["v20"])
    sleeves = {"CASH": rf, "MACRO": r["MACRO"], "TREND": r["TREND"], "TREND_NE": r["TREND_NE"]}

    # ---- (1) vol-matched at half v20's vol (apples-to-apples) --------------
    target = v20m["vol"] / 2
    print(f"\n=== Defensive half, VOL-MATCHED to {target*100:.0f}% (= half of v20's "
          f"{v20m['vol']*100:.0f}%) ===")
    print(f"{'blend (w_v20 / sleeve)':<26}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    print(row("v20 alone", v20m))
    for name, dr in sleeves.items():
        w, m, _ = vol_matched_blend(r["v20"], dr, target)
        print(row(f"{w:.2f} v20 / {1-w:.2f} {name}", m))

    # ---- (2) in-sample max-Sharpe tangency {v20, sleeve} (ceiling) ---------
    print("\n=== Max-Sharpe tangency {v20, sleeve} (in-sample CEILING, unlevered) ===")
    print(f"{'sleeve':<26}{'w_v20':>7}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    for name, dr in sleeves.items():
        w, m = tangency(r["v20"], dr)
        print(f"{name:<26}{w:>7.2f}{m['cagr']*100:>8.1f}{m['sharpe']:>8.2f}"
              f"{m['maxdd']*100:>9.1f}{m['vol']*100:>8.1f}")

    # ---- (3) sub-period drawdown protection (50/50, per sleeve) ------------
    print("\n=== Sub-period MaxDD: v20 alone vs 50/50 v20+sleeve ===")
    print(f"{'window':<12}{'v20 DD%':>9}{'+CASH':>9}{'+MACRO':>9}{'+TREND':>9}{'+TREND_NE':>11}"
          f"{'  corr(NE,v20)':>15}")
    wins = [("2007-2009", "2007", "2009"), ("2010-2014", "2010", "2014"),
            ("2015-2019", "2015", "2019"), ("2020-2022", "2020", "2022"),
            ("2023-2026", "2023", "2026")]
    for label, lo, hi in wins:
        sub = r.loc[f"{lo}-01-01":f"{hi}-12-31"]
        subrf = rf.loc[sub.index]
        if len(sub) < 30:
            continue
        vdd = M(sub["v20"])["maxdd"] * 100
        cdd = M(0.5 * sub["v20"] + 0.5 * subrf)["maxdd"] * 100
        mdd = M(0.5 * sub["v20"] + 0.5 * sub["MACRO"])["maxdd"] * 100
        tdd = M(0.5 * sub["v20"] + 0.5 * sub["TREND"])["maxdd"] * 100
        ndd = M(0.5 * sub["v20"] + 0.5 * sub["TREND_NE"])["maxdd"] * 100
        cn = sub["TREND_NE"].corr(sub["v20"])
        print(f"{label:<12}{vdd:>9.1f}{cdd:>9.1f}{mdd:>9.1f}{tdd:>9.1f}{ndd:>11.1f}{cn:>15.2f}")

    # ---- (4) crisis calendar-year returns ---------------------------------
    print("\n=== Crisis years — sleeve standalone total return (crisis alpha?) ===")
    print(f"{'year':<8}{'v20%':>9}{'MACRO%':>9}{'TREND%':>9}{'TREND_NE%':>11}")
    for yr in ["2008", "2011", "2018", "2020", "2022"]:
        sub = r.loc[f"{yr}-01-01":f"{yr}-12-31"]
        if len(sub) < 30:
            continue
        vy = (1 + sub["v20"]).prod() - 1
        my = (1 + sub["MACRO"]).prod() - 1
        ty = (1 + sub["TREND"]).prod() - 1
        ny = (1 + sub["TREND_NE"]).prod() - 1
        print(f"{yr:<8}{vy*100:>9.1f}{my*100:>9.1f}{ty*100:>9.1f}{ny*100:>11.1f}")


if __name__ == "__main__":
    main()
