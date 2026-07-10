"""ALLOC status emitter — one JSON snapshot of the agent's current posture, for
consumption by the trading-agent daily dashboard (read-only data hand-off; no code
crosses the repo boundary in either direction).

Writes data/alloc-status.json:
  as_of, gate, lev_held, lev_desired, vol20, rebound{...}, defensive, cpi_cool,
  tqqq_fraction, book weights (QQQ-equiv + defensive + 8 trend ETFs), paper
  {equity per $100k-2007, ret_12m, qqq_ret_12m, maxdd}, action, generated_at.

ACTION is the operator flag: NONE most days; REBALANCE-DUE on the monthly sleeve
day; RESIZE when the dial has drifted past the band; plus gate-flip notices.

    python -m research.scripts.alloc_status
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from research.scripts.alloc_agent import (
    _load_env, build_planes, load_index, run_alloc,
    SIGMA_TGT, VOL_WIN, BAND, LMAX, REB_FLOOR, TD,
)
from research.scripts.trend_trades import UNIVERSE, load_prices, weight_path

OUT = "data/alloc-status.json"
W_CORE, W_TREND = 0.6, 0.4


def main() -> None:
    _load_env()
    start, end = "1999-03-10", datetime.now().strftime("%Y-%m-%d")
    idx = load_index("qqq", start, end)
    rf_d, def_d, cool_d = build_planes(idx, start, end)
    core, diag = run_alloc(idx, rf_d, def_d, cool_d,
                           use_dial=True, use_def=True, use_reb=True)

    st = pd.DataFrame(diag["state_rows"], columns=["date", "lev", "state", "eq"]
                      ).set_index("date")
    last = st.index[-1]
    lev_held = float(st["lev"].iloc[-1])

    # gate + rebound state from the event log (source of truth = the engine)
    gate_on, reb_armed = False, False
    for d, ev, _ in diag["events"]:
        if ev == "GATE-ON":
            gate_on = True
        elif ev == "GATE-OFF":
            gate_on = False
        elif ev == "REBOUND-ARM":
            reb_armed = True
        elif ev in ("REBOUND-STOP", "REBOUND-EXPIRE"):
            reb_armed = False
    reb_active = reb_armed  # armed events already respect stop/expiry ordering

    vol20 = float(idx.pct_change().iloc[-VOL_WIN:].std() * np.sqrt(TD))
    desired = 0.0 if not gate_on else float(np.clip(SIGMA_TGT / max(vol20, 1e-9), 0, LMAX))
    if reb_active and last.year == st.index[-1].year:
        desired = max(desired, min(REB_FLOOR, LMAX))

    cool = bool(cool_d.iloc[-1])
    # trend sleeve current weights
    px = load_prices(UNIVERSE)
    tw = weight_path(px).iloc[-1]

    # paper track: 60/40 book vs QQQ, trailing 12m
    tr = pd.read_csv("data/trend_noeq_oos_equity.csv")
    tr["date"] = pd.to_datetime(tr["date"])
    trend_r = tr.set_index("date")["trend_equity"].pct_change()
    j = pd.DataFrame({"c": core.pct_change(), "t": trend_r}).dropna()
    book = (1 + W_CORE * j["c"] + W_TREND * j["t"]).cumprod()
    ret12 = float(book.iloc[-1] / book.iloc[-min(252, len(book))] - 1)
    q12 = float(idx.iloc[-1] / idx.iloc[-252] - 1)
    dd = float(((book - book.cummax()) / book.cummax()).min())

    # the operator ACTION flag
    actions = []
    if abs(desired - lev_held) > BAND:
        actions.append(f"RESIZE core: dial {lev_held:.2f}x -> {desired:.2f}x "
                       f"(TQQQ {W_CORE*lev_held/3*100:.0f}% -> {W_CORE*desired/3*100:.0f}% of book)")
    nxt_bday = last + pd.tseries.offsets.BDay(1)
    if nxt_bday.month != last.month or (last.day <= 3 and last.month != (last - pd.tseries.offsets.BDay(1)).month):
        actions.append("TREND sleeve monthly REBALANCE due")
    recent = [(d, ev) for d, ev, _ in diag["events"] if d >= last - pd.Timedelta(days=7)]
    for d, ev in recent:
        actions.append(f"{ev} on {d.date()}")

    out = {
        "as_of": str(last.date()),
        "gate": "ON" if gate_on else "OFF",
        "lev_held": round(lev_held, 2),
        "lev_desired": round(desired, 2),
        "vol20_pct": round(vol20 * 100, 1),
        "rebound_active": reb_active,
        "defensive": ("IEF (bonds)" if cool else "T-bills (CPI hot)"),
        "cpi_cool": cool,
        "tqqq_fraction_pct": round(W_CORE * lev_held / 3 * 100, 1),
        "weights_pct": {
            "QQQ_equiv": round(W_CORE * lev_held * 100, 1),
            **{s: round(W_TREND * float(tw[s]) * 100, 1) for s in UNIVERSE},
        },
        "paper": {
            "book_equity_per_100k_2007": round(float(book.iloc[-1]) * 100_000, 0),
            "ret_12m_pct": round(ret12 * 100, 1),
            "qqq_ret_12m_pct": round(q12 * 100, 1),
            "maxdd_pct": round(dd * 100, 1),
        },
        "action": actions if actions else ["NONE"],
        "mode": "PAPER",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}: gate {out['gate']}, dial {out['lev_held']}x, "
          f"action={out['action'][0] if out['action'] else 'NONE'}")


if __name__ == "__main__":
    main()
