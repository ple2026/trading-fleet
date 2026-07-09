"""ALLOC-BOOK daily ledger — v11-style holdings + buy/sell record.

Produces data/alloc_book_daily_ledger.csv: one row per trading day for the
DEPLOYABLE book (60% ALLOC core + 40% TREND_NE, no account-level leverage),
expressed in tradable instruments:

  QQQ        core equity leg (weight = 0.6 x dial leverage, so >60% when the
             dial is levered — financed via margin, that cost is in the equity)
  IEF        core defensive leg when gate-off in cool inflation
  TLT/IEF/GLD/SLV/DBC/USO/DBA/UUP   the trend sleeve (signed weights, x0.4)
  CASH_NET   1 - sum(all instrument weights): positive = T-bills, negative =
             margin borrowing
  TRADES     the day's position changes ">= 1pp of book" (BUY/SELL, in pp)

Convention: each row = the position decided at that day's close (held next
session) + the book equity at that close, per $100k start.

    python -m research.scripts.alloc_ledger
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.scripts.alloc_agent import (
    _load_env, build_planes, load_index, run_alloc,
)
from research.scripts.trend_trades import UNIVERSE, load_prices, weight_path

W_CORE, W_TREND = 0.6, 0.4
EPS = 0.01                      # report trades >= 1pp of book


def main() -> None:
    _load_env()
    start, end = "1999-03-10", "2026-07-01"

    # ---- core: daily leverage + defensive state + returns -------------------
    idx = load_index("qqq", start, end)
    rf_d, def_d, cool_d = build_planes(idx, start, end)
    core, diag = run_alloc(idx, rf_d, def_d, cool_d,
                           use_dial=True, use_def=True, use_reb=True)
    st = pd.DataFrame(diag["state_rows"], columns=["date", "lev", "state", "eq"]
                      ).set_index("date")

    # ---- trend sleeve: daily signed weights per instrument + returns --------
    px = load_prices(UNIVERSE)
    tw = weight_path(px)                                  # daily weights, 2007+
    tr = pd.read_csv("data/trend_noeq_oos_equity.csv")
    tr["date"] = pd.to_datetime(tr["date"])
    trend_r = tr.set_index("date")["trend_equity"].pct_change()

    # ---- book equity (60/40 daily-rebalanced) -------------------------------
    j = pd.DataFrame({"core_r": core.pct_change(), "trend_r": trend_r}).dropna()
    days = j.index.intersection(st.index).intersection(tw.index)
    j = j.loc[days]
    eq = 100_000 * (1 + W_CORE * j["core_r"] + W_TREND * j["trend_r"]).cumprod()

    # ---- assemble instrument weights (share of BOOK) ------------------------
    rows = []
    prev_w: dict[str, float] = {}
    for d in days:
        w: dict[str, float] = {}
        lev, state = float(st.loc[d, "lev"]), str(st.loc[d, "state"])
        w["QQQ"] = round(W_CORE * lev, 4)
        w["IEF_DEF"] = round(W_CORE if (lev == 0 and state == "BONDS") else 0.0, 4)
        for sym in UNIVERSE:
            w[sym] = round(W_TREND * float(tw.loc[d, sym]), 4)
        gross = sum(abs(v) for v in w.values())
        cash_net = round(1.0 - sum(w.values()), 4)
        trades = []
        for k in w:
            delta = w[k] - prev_w.get(k, 0.0)
            if abs(delta) >= EPS:
                trades.append(f"{'BUY' if delta > 0 else 'SELL'} {k} {delta*100:+.1f}pp")
        prev_w = w
        rows.append({"date": d, "equity": round(float(eq.loc[d]), 2), **w,
                     "GROSS": round(gross, 3), "CASH_NET": cash_net,
                     "TRADES": "; ".join(trades)})

    out = pd.DataFrame(rows).set_index("date")
    out.to_csv("data/alloc_book_daily_ledger.csv")
    ntr = int((out["TRADES"] != "").sum())
    print(f"wrote data/alloc_book_daily_ledger.csv: {len(out)} days "
          f"{days.min().date()}..{days.max().date()}, {ntr} trade days")
    print(f"final equity per $100k: ${out['equity'].iloc[-1]:,.0f}")
    print("\nsample (first trade days):")
    smp = out[out["TRADES"] != ""].head(4)
    for d, r in smp.iterrows():
        print(f"  {d.date()}  eq ${r['equity']:,.0f}  QQQ {r['QQQ']*100:.0f}%  "
              f"GROSS {r['GROSS']:.2f}x  | {r['TRADES'][:90]}")


if __name__ == "__main__":
    main()
