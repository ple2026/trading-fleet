"""ALLOC implementation comparison — margin-QQQ vs 2x ETF (QLD) vs 3x ETF (TQQQ).

The dial says HOW MUCH exposure; this decides HOW to hold it:
  margin : QQQ notional = lev x equity, borrow (lev-1) at TB3MS+60bps  [current model]
  qld    : hold f = min(lev,2)/2 of the sleeve in 2x-daily-reset ETF, rest in T-BILLS
  tqqq   : hold f = lev/3 of the sleeve in 3x-daily-reset ETF, rest in T-BILLS

ETF modes carry NO margin-call risk (gap loss capped at the ETF stake) and their
cash buffer EARNS the T-bill rate — the margin model credits nothing on residuals.
Both synthetic LETFs are validated against the real ETF before use.

    python -m research.scripts.alloc_impl
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.scripts.alloc_agent import (
    _load_env, build_planes, load_index, metrics, run_alloc,
    SIGMA_TGT, VOL_WIN, BAND, LMAX, REB_FLOOR, REB_STOP, COST_BPS, SPREAD,
    ER_3X, EMA_EXIT, EMA_ENTER, TD,
)

ER_2X = 0.0095          # QLD expense ratio ~0.95%


def synth_letf(r_idx: pd.Series, rf_d: pd.Series, L: float, er: float) -> pd.Series:
    """Daily-reset Lx ETF return: L*r - (L-1)*(rf+spread) - ER."""
    return L * r_idx - (L - 1) * (rf_d * TD + SPREAD) / TD - er / TD


def validate(which: str, L: float, er: float, start: str) -> None:
    idx = load_index("qqq", "1999-03-10", "2026-07-01")
    rf_d, _, _ = build_planes(idx, "1999-03-10", "2026-07-01")
    syn_r = synth_letf(idx.pct_change(), rf_d, L, er)
    try:
        real = pd.read_parquet(f"data/tiingo/{which}.parquet")["c"]
    except FileNotFoundError:
        import sqlite3
        conn = sqlite3.connect("/Users/phuongle/Documents/Trading Agent/data/daily-bars.db")
        rows = conn.execute("SELECT date, close FROM bars WHERE symbol=? AND "
                            "source='tiingo' ORDER BY date", (which,)).fetchall()
        real = pd.Series({pd.Timestamp(d): v for d, v in rows}).sort_index()
    common = syn_r.index.intersection(real.index)
    common = common[common >= pd.Timestamp(start)][1:]
    syn = (1 + syn_r.loc[common]).cumprod()
    rl = real.loc[common] / real.loc[common].iloc[0]
    yrs = len(common) / TD
    gap = abs(syn.iloc[-1] ** (1 / yrs) - rl.iloc[-1] ** (1 / yrs))
    print(f"  syn-{L:g}x vs {which} {common[0].date()}..{common[-1].date()}: "
          f"CAGR gap {gap*100:.2f}pp/yr")


def run_impl(idx, rf_d, def_d, cool_d, mode: str, use_reb=True, warmup=EMA_EXIT):
    """Same gate/dial/rebound brain as run_alloc, but exposure held via `mode`."""
    c, dates = idx.values, idx.index
    r = idx.pct_change()
    ema_f = idx.ewm(span=EMA_ENTER, adjust=False).mean().values
    ema_s = idx.ewm(span=EMA_EXIT, adjust=False).mean().values
    vol = r.rolling(VOL_WIN).std().values * np.sqrt(TD)
    letf_L = 2.0 if mode == "qld" else 3.0
    letf_r = synth_letf(r, rf_d, letf_L, ER_2X if mode == "qld" else ER_3X)

    eq = np.ones(len(dates))
    gate_on, lev_held = False, 0.0
    reb_active, reb_entry_eq, reb_year = False, 1.0, None
    yfp = c[0]
    for t in range(1, len(dates)):
        y = dates[t].year
        # earn with yesterday's position
        if lev_held > 0:
            f = lev_held / letf_L                      # ETF fraction of sleeve
            ret = f * letf_r.iloc[t] + (1 - f) * rf_d.iloc[t]
            eq[t] = eq[t - 1] * (1 + ret)
        else:
            ret = def_d.iloc[t] if cool_d.iloc[t] else rf_d.iloc[t]
            eq[t] = eq[t - 1] * (1 + ret)
        if reb_active and reb_year == y and eq[t] / reb_entry_eq - 1 <= -REB_STOP:
            reb_active = False
        # decide tomorrow's position
        nxt = dates[t + 1].year if t + 1 < len(dates) else y
        if nxt != y:
            pr = c[t] / yfp - 1
            yfp = c[t]
            if use_reb and t >= warmup and pr < 0:
                reb_active, reb_entry_eq, reb_year = True, eq[t], nxt
            else:
                reb_active = False
        if t < warmup:
            continue
        if gate_on and c[t] < ema_s[t]:
            gate_on = False
        elif not gate_on and c[t] > ema_f[t] and c[t] > ema_s[t]:
            gate_on = True
        cap = letf_L if mode == "qld" else LMAX
        if not gate_on:
            desired = 0.0
        else:
            v = vol[t] if vol[t] and not np.isnan(vol[t]) else SIGMA_TGT
            desired = float(np.clip(SIGMA_TGT / v, 0.0, min(LMAX, cap)))
        if reb_active and nxt == reb_year:
            desired = max(desired, min(REB_FLOOR, cap))
        if abs(desired - lev_held) > BAND or (desired == 0) != (lev_held == 0):
            eq[t] *= (1 - abs(desired - lev_held) * COST_BPS / 10_000)
            lev_held = desired
    return pd.Series(eq, index=dates)


def row(name, m):
    return (f"{name:<30}{m['cagr']*100:>8.1f}{m['sharpe']:>7.2f}{m['maxdd']*100:>8.1f}"
            f"{m['mar']:>6.2f}{m['vol']*100:>7.1f}")


def main() -> None:
    _load_env()
    print("Validating synthetic LETFs vs the real ETFs:")
    validate("QLD", 2.0, ER_2X, "2006-06-22")
    validate("TQQQ", 3.0, ER_3X, "2010-02-11")

    start, end = "1999-03-10", "2026-07-01"
    idx = load_index("qqq", start, end)
    rf_d, def_d, cool_d = build_planes(idx, start, end)
    margin, _ = run_alloc(idx, rf_d, def_d, cool_d, use_dial=True, use_def=True, use_reb=True)
    cores = {"margin-QQQ (current)": margin.pct_change(),
             "2x ETF (QLD)+T-bills": run_impl(idx, rf_d, def_d, cool_d, "qld").pct_change(),
             "3x ETF (TQQQ)+T-bills": run_impl(idx, rf_d, def_d, cool_d, "tqqq").pct_change()}

    tr = pd.read_csv("data/trend_noeq_oos_equity.csv")
    tr["date"] = pd.to_datetime(tr["date"])
    trend_r = tr.set_index("date")["trend_equity"].pct_change()

    hdr = f"{'implementation':<30}{'CAGR%':>8}{'Shrp':>7}{'MaxDD%':>8}{'MAR':>6}{'Vol%':>7}"
    print(f"\nCORE 1999-2026:\n{hdr}\n" + "-" * len(hdr))
    for name, cr in cores.items():
        print(row(name, metrics(100_000 * (1 + cr.dropna()).cumprod())))
    print(f"\nBOOK (60/40 with TREND_NE) 2007-2026:\n{hdr}\n" + "-" * len(hdr))
    for name, cr in cores.items():
        j = pd.DataFrame({"c": cr, "t": trend_r}).dropna()
        book = 0.6 * j["c"] + 0.4 * j["t"]
        print(row(name, metrics(100_000 * (1 + book).cumprod())))


if __name__ == "__main__":
    main()
