"""ALLOC-BOOK — compose the ALLOC core with the TREND_NE sleeve, then layer
leverage over the BLEND (the user's 'amazing returns' test), with real financing.

Book return: r = L * [w*r_core + (1-w)*r_trend] - (L-1)*(TB3MS+60bps)/252
Frontier over w (core weight) x L (book leverage). Costs/financing inside the core
are already charged by alloc_agent; blend rebalancing is ~free (monthly ~ daily).

    python -m research.scripts.alloc_book
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.scripts.alloc_agent import (
    build_planes, load_index, metrics, run_alloc, _load_env, TD,
)

V20_CSV = "/Users/phuongle/Documents/Trading Agent/backtest-output/v20/equity-layers.csv"


def row(name, m):
    return (f"{name:<30}{m['cagr']*100:>8.1f}{m['sharpe']:>7.2f}{m['maxdd']*100:>8.1f}"
            f"{m['mar']:>6.2f}{m['vol']*100:>7.1f}")


def main():
    _load_env()
    start, end = "1999-03-10", "2026-07-01"
    idx = load_index("qqq", start, end)
    rf_d, def_d, cool_d = build_planes(idx, start, end)
    core, _ = run_alloc(idx, rf_d, def_d, cool_d,
                        use_dial=True, use_def=True, use_reb=False)  # A3 core
    core_r = core.pct_change()

    tr = pd.read_csv("data/trend_noeq_oos_equity.csv")
    tr["date"] = pd.to_datetime(tr["date"])
    tr_r = tr.set_index("date")["trend_equity"].pct_change()

    j = pd.DataFrame({"core": core_r, "trend": tr_r, "rf": rf_d}).dropna()
    print(f"Blend span {j.index.min().date()}..{j.index.max().date()} ({len(j)} days)")
    print(f"corr(core, trend) = {j['core'].corr(j['trend']):+.3f}\n")

    spy = load_index("spy", "2007-01-01", end); spy = spy.loc[j.index.min():]
    qqq = idx.loc[j.index.min():]
    v20 = pd.read_csv(V20_CSV, encoding="utf-8-sig")
    v20["date"] = pd.to_datetime(v20["date"], format="mixed")
    v20 = v20.set_index("date")["v20_equity"].loc[j.index.min():]

    header = f"{'book (w_core / L)':<30}{'CAGR%':>8}{'Shrp':>7}{'MaxDD%':>8}{'MAR':>6}{'Vol%':>7}"
    print(header + "\n" + "-" * len(header))
    print(row("SPY buy&hold", metrics(spy / spy.iloc[0])))
    print(row("QQQ buy&hold", metrics(qqq / qqq.iloc[0])))
    print(row("v20 (context)", metrics(v20 / v20.iloc[0])))
    print()
    fin = (j["rf"] * TD + 0.006) / TD
    for w in (1.0, 0.8, 0.7, 0.6, 0.5):
        blend = w * j["core"] + (1 - w) * j["trend"]
        for L in (1.0, 1.5, 2.0):
            r = L * blend - max(L - 1, 0) * fin
            m = metrics(100_000 * (1 + r).cumprod())
            tag = f"w={w:.1f} L={L:.2f}"
            print(row(tag, m))
        print()


if __name__ == "__main__":
    main()
