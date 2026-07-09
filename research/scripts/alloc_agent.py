"""ALLOC — the risk-allocation agent (fleet pivot). Spec: docs/ALLOC_AGENT.md.

One machine, four century-validated components, ALL parameters a-priori:
  gate      asymmetric EMA200-exit / EMA50-reenter (century-validated)
  dial      lev = clip(sigma_tgt / sigma_20d, 0, 3)   sigma_tgt=30%
  defensive gate-OFF capital -> IEF if trailing-3y CPI <= 3% else cash
  rebound   prior index year < 0 -> dial floored at 3.0, -10% stop, EOY expiry
Financing charged on borrowed notional (TB3MS + 60bps), ER 0.20->0.95%/yr,
5 bps turnover cost, 0.25 leverage band. No lookahead: signals at close t earn t+1.

  python -m research.scripts.alloc_agent                    # QQQ 1999-2026 ablation
  INDEX=nasdaq python -m research.scripts.alloc_agent       # 1971+ era-robustness
  python -m research.scripts.alloc_agent --validate-syn     # synthetic 3x vs TQQQ

Env overrides for 1-D sweeps: SIGMA_TGT VOL_WIN BAND REB_FLOOR LMAX COST_BPS SPREAD.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

TD = 252
DB = "/Users/phuongle/Documents/Trading Agent/data/daily-bars.db"
SECULAR = "/Users/phuongle/Documents/Trading Agent/data/secular"
V20_CSV = "/Users/phuongle/Documents/Trading Agent/backtest-output/v20/equity-layers.csv"

# ---- a-priori parameters (env-overridable ONLY for pre-registered sweeps) ----
SIGMA_TGT = float(os.environ.get("SIGMA_TGT", "0.30"))
VOL_WIN   = int(os.environ.get("VOL_WIN", "20"))
BAND      = float(os.environ.get("BAND", "0.25"))
LMAX      = float(os.environ.get("LMAX", "3.0"))
REB_FLOOR = float(os.environ.get("REB_FLOOR", "3.0"))
REB_STOP  = float(os.environ.get("REB_STOP", "0.10"))
COST_BPS  = float(os.environ.get("COST_BPS", "5.0"))
SPREAD    = float(os.environ.get("SPREAD", "0.006"))     # borrow spread over TB3MS
ER_BASE, ER_3X = 0.0020, 0.0095                           # ER at 1x .. 3x (linear)
CPI_SWITCH = 3.0                                          # defensive: bonds iff CPI3y <= 3%
EMA_EXIT  = int(os.environ.get("EMA_EXIT", "200"))
EMA_ENTER = int(os.environ.get("EMA_ENTER", "50"))


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


# ------------------------------------------------------------------ data plane
def load_index(which: str, start: str, end: str) -> pd.Series:
    if which in ("qqq", "spy"):
        conn = sqlite3.connect(DB)
        rows = conn.execute(
            "SELECT date, close FROM bars WHERE symbol=? AND source='tiingo' "
            "AND date BETWEEN ? AND ? ORDER BY date", (which.upper(), start, end)).fetchall()
        s = pd.Series({pd.Timestamp(d): c for d, c in rows}, name="c")
    elif which == "dow":  # spliced dja (1885-2019) + djia (2016-2026)
        a = pd.read_csv(f"{SECULAR}/dja.csv"); a["date"] = pd.to_datetime(a["date"])
        b = pd.read_csv(f"{SECULAR}/djia.csv"); b["date"] = pd.to_datetime(b["date"])
        a, b = a.set_index("date")["close"], b.set_index("date")["close"]
        s = pd.concat([a[a.index < b.index.min()], b]).rename("c")
        s = s.loc[(s.index >= start) & (s.index <= end)]
    else:  # nasdaq composite from the secular CSV
        df = pd.read_csv(f"{SECULAR}/nasdaqcom.csv")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].rename("c")
        s = s.loc[(s.index >= start) & (s.index <= end)]
    return s.sort_index()


def fred_monthly(series_id: str, start: str, end: str) -> pd.Series:
    from research.fleet.macro_data import FredClient
    return FredClient().series(series_id,
                               datetime.fromisoformat(start), datetime.fromisoformat(end))


def build_planes(idx: pd.Series, start: str, end: str):
    """Daily rf, defensive-asset daily TR, and the CPI<=3% condition (lagged 1mo)."""
    days = idx.index
    tb3 = fred_monthly("TB3MS", "1934-01-01", end).shift(1)   # month-avg known at month-END
    # pre-TB3MS (Dow century run): flat 4%/yr — approximate call-money-era funding
    rf_d = (tb3.reindex(days, method="ffill") / 100 / TD).fillna(0.04 / TD)

    cpi = fred_monthly("CPIAUCNS", "1950-01-01", end)
    cpi3y = (cpi / cpi.shift(36)) ** (1 / 3) - 1
    cool = (cpi3y * 100 <= CPI_SWITCH).shift(2)            # BLS publishes mid-next-month
    cool_d = cool.reindex(days, method="ffill").fillna(False).astype(bool)

    # defensive TR: IEF daily (2004+) else GS10-implied monthly spread across days
    try:
        ief = pd.read_parquet("data/tiingo/IEF.parquet")["c"].pct_change()
    except Exception:
        ief = pd.Series(dtype=float)
    gs10 = fred_monthly("GS10", "1953-01-01", end)
    # month-m TR is known only at month-END -> shift so month m uses m-1's TR
    bond_m = ((gs10.shift(1) / 12 - 7.5 * gs10.diff()) / 100).shift(1)
    mdays = pd.Series(days, index=days).groupby([days.year, days.month]).transform("count")
    bond_d_proxy = (bond_m.reindex(days, method="ffill") / mdays).fillna(0.0)
    def_d = ief.reindex(days)
    def_d = def_d.where(def_d.notna(), bond_d_proxy).fillna(0.0)
    return rf_d, def_d, cool_d


# ------------------------------------------------------------------ the engine
def run_alloc(idx: pd.Series, rf_d, def_d, cool_d, *,
              use_dial=True, use_def=True, use_reb=True, fixed_lev=None,
              ens=False, gold_d=None, warmup=EMA_EXIT):
    """Daily simulation. Returns (equity Series, diag dict).

    ens=True blends three classic exposure votes (asymmetric-EMA gate, close >
    210d SMA ~ 10-mo, 12-1 TSMOM sign) into a fractional exposure — reduces
    single-signal timing luck. gold_d: hot-inflation defensive returns (else cash).
    """
    c = idx.values
    dates = idx.index
    r = idx.pct_change().values
    ema_fast = idx.ewm(span=EMA_ENTER, adjust=False).mean().values
    ema_slow = idx.ewm(span=EMA_EXIT, adjust=False).mean().values
    sma210 = idx.rolling(210).mean().values
    tsm = np.full(len(idx), np.nan)
    tsm[252:] = c[252:] / c[:-252] - 1.0
    vol = pd.Series(r, index=dates).rolling(VOL_WIN).std().values * np.sqrt(TD)

    eq = np.ones(len(dates))
    gate_on, lev_held = False, 0.0
    reb_active, reb_entry_eq, reb_year = False, 1.0, None
    year_first_px = c[0]
    turn_total, fin_total, day_count = 0.0, 0.0, 0
    lev_log = []
    events = []                      # (date, event, detail) — the trade history
    state_rows = []                  # daily state for CSV export

    for t in range(1, len(dates)):
        y = dates[t].year
        # ---- earn today's return with YESTERDAY's position (no lookahead) ----
        if lev_held > 0:
            fin = max(lev_held - 1, 0) * (rf_d.iloc[t] * TD + SPREAD) / TD
            er = (ER_BASE + max(lev_held - 1, 0) / 2 * (ER_3X - ER_BASE)) / TD
            ret = lev_held * r[t] - fin - er
            fin_total += fin + er
        else:
            if use_def and cool_d.iloc[t]:
                ret = def_d.iloc[t]
            elif use_def and gold_d is not None and not np.isnan(gold_d.iloc[t]):
                ret = gold_d.iloc[t]           # hot inflation: gold, not cash
            else:
                ret = rf_d.iloc[t]
        eq[t] = eq[t - 1] * (1 + ret)
        day_count += 1

        # ---- rebound stop check (on the levered book since floor entry) ----
        if reb_active and reb_year == y and eq[t] / reb_entry_eq - 1 <= -REB_STOP:
            reb_active = False                      # stopped: revert to dial
            events.append((dates[t], "REBOUND-STOP",
                           f"book {(eq[t]/reb_entry_eq-1)*100:+.1f}% since floor entry -> revert to dial"))

        # ---- decide TOMORROW's position from today's close ----
        # year-boundary bookkeeping at the ENDING year's final close, so the
        # rebound floor covers day 1 of the new year and expires INTO year-end
        eff_next_year = dates[t + 1].year if t + 1 < len(dates) else y
        if eff_next_year != y:
            prev_ret = c[t] / year_first_px - 1
            year_first_px = c[t]
            if use_reb and t >= warmup and prev_ret < 0:
                if reb_active:
                    events.append((dates[t], "REBOUND-EXPIRE", "EOY exit; re-armed (year negative again)"))
                reb_active, reb_entry_eq, reb_year = True, eq[t], eff_next_year
                events.append((dates[t], "REBOUND-ARM",
                               f"{y} index {prev_ret*100:+.1f}% -> floor {min(REB_FLOOR, LMAX):g}x for {eff_next_year}, stop -{REB_STOP*100:.0f}%"))
            else:
                if reb_active:
                    events.append((dates[t], "REBOUND-EXPIRE", "EOY exit (hold-to-year-end)"))
                reb_active = False                  # floor never crosses a boundary
        if t < warmup:
            continue
        if gate_on and c[t] < ema_slow[t]:
            gate_on = False
            events.append((dates[t], "GATE-OFF",
                           f"close < EMA{EMA_EXIT} -> defensive ({'bonds' if (use_def and cool_d.iloc[t]) else 'cash'})"))
        elif not gate_on and c[t] > ema_fast[t] and c[t] > ema_slow[t]:
            gate_on = True
            events.append((dates[t], "GATE-ON", f"close > EMA{EMA_ENTER} -> risk back on"))
        if ens:   # fractional exposure: mean of 3 classic votes
            votes = [1.0 if gate_on else 0.0,
                     1.0 if (not np.isnan(sma210[t]) and c[t] > sma210[t]) else 0.0,
                     1.0 if (not np.isnan(tsm[t]) and tsm[t] > 0) else 0.0]
            expo = float(np.mean(votes))
        else:
            expo = 1.0 if gate_on else 0.0
        if expo == 0.0:
            desired = 0.0
        elif fixed_lev is not None:
            desired = fixed_lev * expo
        elif use_dial:
            v = vol[t] if vol[t] and not np.isnan(vol[t]) else SIGMA_TGT
            desired = float(np.clip(SIGMA_TGT / v, 0.0, LMAX)) * expo
        else:
            desired = expo
        # rebound floor OVERRIDES the gate (v20 semantics: the Jan bet fires even
        # while the regime gate is off — that's the whole point of the rebound).
        # Keyed to the year of t+1 = the day this position actually earns.
        if reb_active and eff_next_year == reb_year:
            desired = max(desired, min(REB_FLOOR, LMAX))
        if abs(desired - lev_held) > BAND or (desired == 0) != (lev_held == 0):
            cost = abs(desired - lev_held) * COST_BPS / 10_000
            eq[t] *= (1 - cost)
            turn_total += abs(desired - lev_held)
            lev_held = desired
        lev_log.append(lev_held)
        state_rows.append((dates[t], lev_held,
                           "LONG" if lev_held > 0 else
                           ("BONDS" if (use_def and cool_d.iloc[t]) else "CASH"),
                           eq[t]))

    curve = pd.Series(eq, index=dates)
    diag = {"avg_lev": float(np.mean(lev_log)) if lev_log else 0.0,
            "turnover_yr": turn_total / (day_count / TD),
            "drag_yr": fin_total / (day_count / TD),
            "events": events, "state_rows": state_rows}
    return curve, diag


# ------------------------------------------------------------------- metrics
def metrics(curve: pd.Series) -> dict:
    r = curve.pct_change().dropna()
    yrs = len(curve) / TD
    dd = float(((curve - curve.cummax()) / curve.cummax()).min())
    cagr = float((curve.iloc[-1] / curve.iloc[0]) ** (1 / yrs) - 1)
    vol = float(r.std() * np.sqrt(TD))
    sharpe = float(r.mean() / r.std() * np.sqrt(TD)) if r.std() > 0 else 0.0
    # top-5-year share of log-wealth (the concentration metric)
    ye = curve.resample("YE").last()
    ylog = np.log(ye / ye.shift(1)).dropna()
    tot = ylog.sum()
    top5 = float(ylog.sort_values(ascending=False).head(5).sum() / tot) if tot > 0 else np.nan
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": dd, "vol": vol,
            "mar": cagr / abs(dd) if dd else np.nan, "top5_share": top5}


def row(name, m, diag=None):
    d = f"  lev {diag['avg_lev']:.2f} turn {diag['turnover_yr']:.1f}/yr drag {diag['drag_yr']*100:.2f}%/yr" if diag else ""
    return (f"{name:<26}{m['cagr']*100:>8.1f}{m['sharpe']:>7.2f}{m['maxdd']*100:>8.1f}"
            f"{m['mar']:>6.2f}{m['vol']*100:>7.1f}{m['top5_share']*100 if m['top5_share']==m['top5_share'] else float('nan'):>7.0f}{d}")


# ------------------------------------------------------------------- runners
def validate_syn(start="2010-02-11", end="2026-07-01"):
    """Synthetic 3x QQQ with our financing model vs real TQQQ."""
    _load_env()
    conn = sqlite3.connect(DB)
    q = load_index("qqq", "2009-01-01", end)
    rows = conn.execute("SELECT date, close FROM bars WHERE symbol='TQQQ' AND source='tiingo' "
                        "AND date BETWEEN ? AND ? ORDER BY date", (start, end)).fetchall()
    tqqq = pd.Series({pd.Timestamp(d): c for d, c in rows}).sort_index()
    rf_d, _, _ = build_planes(q, "2009-01-01", end)
    r = q.pct_change()
    syn_r = 3 * r - 2 * (rf_d * TD + SPREAD) / TD - ER_3X / TD
    common = syn_r.index.intersection(tqqq.index)[1:]
    syn = (1 + syn_r.loc[common]).cumprod()
    real = tqqq.loc[common] / tqqq.loc[common].iloc[0]
    yrs = len(common) / TD
    cs, cr = (syn.iloc[-1]) ** (1 / yrs) - 1, (real.iloc[-1] / real.iloc[0]) ** (1 / yrs) - 1
    te = float((syn.pct_change() - real.pct_change()).std() * np.sqrt(TD))
    print(f"Synthetic-3x validation {common[0].date()}..{common[-1].date()}: "
          f"syn CAGR {cs*100:.1f}% vs TQQQ {cr*100:.1f}%  (gap {abs(cs-cr)*100:.2f}pp/yr, TE {te*100:.1f}%)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate-syn", action="store_true")
    ap.add_argument("--history", default=None, metavar="FROM_DATE",
                    help="print the ALLOC-core trade/state history from this date "
                         "(e.g. 2007-01-01) + yearly summary; writes daily state CSV")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    _load_env()
    if args.validate_syn:
        validate_syn()
        return
    if args.history:
        which = os.environ.get("INDEX", "qqq")
        start = args.start or ("1999-03-10" if which == "qqq" else "1971-02-05")
        idx = load_index(which, start, args.end)
        rf_d, def_d, cool_d = build_planes(idx, start, args.end)
        curve, diag = run_alloc(idx, rf_d, def_d, cool_d,
                                use_dial=True, use_def=True, use_reb=True)
        frm = pd.Timestamp(args.history)
        print(f"ALLOC-core trade history on {which.upper()} from {args.history} "
              f"(engine warms up from {start}; every event is a real position change)\n")
        for d, ev, detail in diag["events"]:
            if d >= frm:
                print(f"  {d.date()}  {ev:<15} {detail}")
        st = pd.DataFrame(diag["state_rows"], columns=["date", "lev", "state", "eq"]).set_index("date")
        st.to_csv("data/alloc_state_history.csv")
        ev = pd.DataFrame(diag["events"], columns=["date", "event", "detail"]).set_index("date")
        ev.to_csv("data/alloc_trade_history.csv")
        print(f"\n(wrote {len(ev)} events -> data/alloc_trade_history.csv; "
              f"full daily state -> data/alloc_state_history.csv, {len(st)} rows)")
        # yearly summary
        st = st.loc[st.index >= frm]
        idx_y = idx.loc[idx.index >= frm]
        print(f"\n{'year':<6}{'avg lev':>8}{'% days long':>12}{'book ret%':>11}{'index ret%':>12}")
        for yy, g in st.groupby(st.index.year):
            ref = st["eq"].shift(1).reindex(g.index).iloc[0]
            if pd.isna(ref):
                ref = g["eq"].iloc[0]
            br = g["eq"].iloc[-1] / ref - 1
            iy = idx_y[idx_y.index.year == yy]
            ir = iy.iloc[-1] / iy.iloc[0] - 1 if len(iy) > 1 else 0.0
            print(f"{yy:<6}{g['lev'].mean():>8.2f}{(g['lev']>0).mean()*100:>12.0f}"
                  f"{br*100:>11.1f}{ir*100:>12.1f}")
        return

    which = os.environ.get("INDEX", "qqq")
    start = args.start or ("1999-03-10" if which == "qqq" else "1971-02-05")
    idx = load_index(which, start, args.end)
    rf_d, def_d, cool_d = build_planes(idx, start, args.end)
    print(f"ALLOC on {which.upper()} {idx.index.min().date()}..{idx.index.max().date()} "
          f"({len(idx)} days)  sigma_tgt={SIGMA_TGT} vol_win={VOL_WIN} band={BAND} "
          f"reb_floor={REB_FLOOR} spread={SPREAD}")

    try:
        gld = pd.read_parquet("data/tiingo/GLD.parquet")["c"].pct_change()
        gold_d = gld.reindex(idx.index)
    except Exception:
        gold_d = None

    header = f"{'variant':<26}{'CAGR%':>8}{'Shrp':>7}{'MaxDD%':>8}{'MAR':>6}{'Vol%':>7}{'top5%':>7}"
    print("\n" + header + "\n" + "-" * len(header))
    bh = metrics(idx / idx.iloc[0]); bh_row = {"cagr": bh["cagr"], "sharpe": bh["sharpe"],
        "maxdd": bh["maxdd"], "vol": bh["vol"], "mar": bh["mar"], "top5_share": bh["top5_share"]}
    print(row(f"A0 {which.upper()} buy&hold", bh_row))
    if which == "qqq":  # SPY benchmark on the same span
        spy = load_index("spy", start, args.end)
        print(row("A0b SPY buy&hold", metrics(spy / spy.iloc[0])))
    variants = [
        ("A1 gate 1x",          dict(use_dial=False, use_def=False, use_reb=False)),
        ("A2 gate+dial",        dict(use_dial=True,  use_def=False, use_reb=False)),
        ("A3 +defensive",       dict(use_dial=True,  use_def=True,  use_reb=False)),
        ("A3e ens-gate",        dict(use_dial=True,  use_def=True,  use_reb=False, ens=True)),
        ("A3g +gold-hot",       dict(use_dial=True,  use_def=True,  use_reb=False, gold_d=gold_d)),
        ("A4 +rebound = ALLOC", dict(use_dial=True,  use_def=True,  use_reb=True)),
        ("A5 gated FIXED 3x",   dict(use_dial=False, use_def=True,  use_reb=False, fixed_lev=3.0)),
    ]
    curves = {}
    for name, kw in variants:
        curve, diag = run_alloc(idx, rf_d, def_d, cool_d, **kw)
        curves[name] = curve
        print(row(name, metrics(curve), diag))

    # v20 comparison + trend blend on the common span (QQQ runs only)
    if which == "qqq":
        v20 = pd.read_csv(V20_CSV, encoding="utf-8-sig")
        v20["date"] = pd.to_datetime(v20["date"], format="mixed")
        v20 = v20.set_index("date")["v20_equity"].sort_index()
        alloc = curves["A4 +rebound = ALLOC"]
        common = alloc.index.intersection(v20.index)
        common = common[common >= pd.Timestamp("2007-01-04")]
        a, v = alloc.loc[common], v20.loc[common]
        print(f"\n=== vs v20, common span {common[0].date()}..{common[-1].date()} ===")
        print(header)
        print(row("v20", metrics(v)))
        print(row("ALLOC core", metrics(a)))
        try:
            tr = pd.read_csv("data/trend_noeq_oos_equity.csv")
            tr["date"] = pd.to_datetime(tr["date"])
            trr = tr.set_index("date")["trend_equity"].pct_change()
            ar = a.pct_change()
            j = pd.DataFrame({"a": ar, "t": trr}).dropna()
            blend = (1 + 0.8 * j["a"] + 0.2 * j["t"]).cumprod()
            print(row("ALLOC 80/20 TREND_NE", metrics(blend)))
            print(f"\ncorr(ALLOC, v20) daily = {a.pct_change().corr(v.pct_change()):+.3f}")
        except Exception as e:
            print(f"(trend blend skipped: {e})")

    if args.out:
        pd.DataFrame({k: v for k, v in curves.items()}).to_csv(args.out, index_label="date")
        print(f"\nwrote curves -> {args.out}")


if __name__ == "__main__":
    main()
