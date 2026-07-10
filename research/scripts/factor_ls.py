"""FACTOR-LS — diversified dollar-neutral multi-factor long-short sleeve.

The hunt for a higher-Sharpe UNCORRELATED sleeve for v20 (Sharpe ~1.0+, |corr|<0.3
including IN THE TAIL). Dollar-neutral by construction → structurally low beta, not
incidentally. Combines the canonical PRICE factors (a-priori, no tuning):

  MOM    12-1 month return   (t-252 .. t-21)     long winners
  LOWVOL -trailing 120d vol                       long low-vol
  REV    -last-21d return                          long 1-month losers (reversal)

Composite = mean of cross-sectional z-scores (winsorized ±3). Long top quintile /
short bottom quintile, equal-weight, dollar-neutral (0.5 long + 0.5 short), monthly
rebalance. Costs: turnover x COST_BPS one-way + short borrow. Strictly point-in-time
(factors use data before the rebalance day). Survivorship-free universe.

    python -m research.scripts.factor_ls --sample 400 --min-dv 20e6

Reports standalone Sharpe/vol/MaxDD, corr to v20 + SPY, and the TAIL test (what the
sleeve does on v20's worst-decile days).
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from research.fleet.universe import (
    load_supported_tickers, survivorship_free_universe, sample_universe,
)

TD = 252
V20_CSV = "/Users/phuongle/Documents/Trading Agent/backtest-output/v20/equity-layers.csv"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def build_panel(symbols, start, end):
    from research.fleet.data import TiingoProvider
    prov = TiingoProvider()
    close, dvol = {}, {}
    ok = 0
    for s in symbols:
        try:
            df = prov.daily_bars(s, start, end)
        except Exception:
            continue
        if len(df) > 300:
            close[s] = df["c"]
            dvol[s] = (df["c"] * df["v"]).median()
            ok += 1
    px = pd.DataFrame(close).sort_index()
    return px, pd.Series(dvol)


def zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    mu, sd = s.mean(), s.std()
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-3, 3)


def factor_scores(px_hist: pd.DataFrame) -> pd.Series:
    """Composite factor z at the rebalance day, from history ending strictly before it."""
    close = px_hist
    last = close.iloc[-1]
    # MOM 12-1
    if len(close) < 252:
        return pd.Series(dtype=float)
    mom = close.iloc[-21] / close.iloc[-252] - 1.0
    # LOWVOL (negative trailing 120d vol)
    rets = close.pct_change()
    lowvol = -rets.iloc[-120:].std()
    # REV (negative last-21d return)
    rev = -(last / close.iloc[-21] - 1.0)
    # require all three defined
    df = pd.DataFrame({"MOM": mom, "LOWVOL": lowvol, "REV": rev}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    comp = (zscore(df["MOM"]) + zscore(df["LOWVOL"]) + zscore(df["REV"])) / 3.0
    return comp


def run(px: pd.DataFrame, start, cost_bps=10.0, borrow_ann=0.01, quantile=0.2):
    rets = px.pct_change()
    dates = px.index
    mk = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first()
    rebal = {d for d in mk.values if d >= pd.Timestamp(start)}
    w = pd.Series(0.0, index=px.columns)
    sleeve = pd.Series(0.0, index=dates, dtype=float)
    borrow_daily = borrow_ann / TD
    nlog = []
    for i, d in enumerate(dates):
        if d < pd.Timestamp(start):
            continue
        cost = 0.0
        if d in rebal:
            hist = px.loc[:dates[i - 1]]
            # only names with a live price at the rebalance and enough history
            valid = hist.columns[hist.iloc[-1].notna() & (hist.notna().sum() > 252)]
            comp = factor_scores(hist[valid])
            if len(comp) >= 20:
                comp = comp.dropna().sort_values()
                n = max(1, int(len(comp) * quantile))
                shorts, longs = comp.index[:n], comp.index[-n:]
                nw = pd.Series(0.0, index=px.columns)
                nw[longs] = 0.5 / len(longs)
                nw[shorts] = -0.5 / len(shorts)
                cost = float((nw - w).abs().sum()) * cost_bps / 10_000.0
                w = nw
                nlog.append(len(longs))
        gross_short = float(w[w < 0].abs().sum())
        r = float((w * rets.loc[d]).fillna(0.0).sum()) - (cost if d in rebal else 0.0) \
            - gross_short * borrow_daily
        sleeve.loc[d] = r
    curve = 100_000 * (1 + sleeve.loc[sleeve.index >= pd.Timestamp(start)]).cumprod()
    if nlog:
        print(f"  avg {np.mean(nlog):.0f} long / {np.mean(nlog):.0f} short per month")
    return curve, sleeve.loc[sleeve.index >= pd.Timestamp(start)]


def metrics(sl: pd.Series) -> dict:
    c = (1 + sl).cumprod()
    dd = float(((c - c.cummax()) / c.cummax()).min())
    vol = float(sl.std() * np.sqrt(TD))
    shrp = float(sl.mean() / sl.std() * np.sqrt(TD)) if sl.std() > 0 else 0.0
    ann = float((1 + sl.mean()) ** TD - 1)
    return {"ann_ret": ann, "sharpe": shrp, "maxdd": dd, "vol": vol}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--trade-start", default="2008-06-01")   # ~18mo warmup for 12-1 + vol
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--min-dv", type=float, default=20e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--out", default="data/factor_ls_equity.csv")
    args = ap.parse_args()
    _load_env()

    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    recs = load_supported_tickers()
    free = survivorship_free_universe(recs, start, end)
    picked = sample_universe(free, args.sample, args.seed)
    print(f"Survivorship-free sample: {len(picked)} names; fetching (cached)…")
    px, dvol = build_panel(picked, start, end)
    liquid = dvol[dvol >= args.min_dv].index
    px = px[liquid]
    print(f"  loaded {px.shape[1]} names; {len(liquid)} pass ${args.min_dv:,.0f}/day liquidity")
    if px.shape[1] < 40:
        raise SystemExit("too few liquid names for a factor book — raise --sample")

    curve, sl = run(px, datetime.fromisoformat(args.trade_start), cost_bps=args.cost_bps)
    curve.to_frame("factor_ls_equity").to_csv(args.out, index_label="date")
    m = metrics(sl)
    print(f"\n{'FACTOR-LS (dollar-neutral)':<28}{'AnnRet%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    print(f"{'standalone':<28}{m['ann_ret']*100:>8.1f}{m['sharpe']:>8.2f}"
          f"{m['maxdd']*100:>9.1f}{m['vol']*100:>8.1f}")
    print(f"span {sl.index.min().date()}..{sl.index.max().date()}")

    # correlation to v20 + tail test
    v20 = pd.read_csv(V20_CSV, encoding="utf-8-sig")
    v20["date"] = pd.to_datetime(v20["date"], format="mixed")
    v20 = v20.set_index("date")["v20_equity"].sort_index()
    j = pd.DataFrame({"F": (1 + sl).cumprod(), "v20": v20}).dropna().pct_change().dropna()
    print(f"\ncorr(FACTOR-LS, v20) = {j['F'].corr(j['v20']):+.3f}   bar: Sharpe>~1.0 & |corr|<0.3")
    worst = j.nsmallest(int(len(j) * 0.1), "v20")   # v20's worst decile
    print(f"TAIL: on v20's worst-decile days (avg {worst['v20'].mean()*100:.1f}%), "
          f"FACTOR-LS avg {worst['F'].mean()*100:+.2f}%, up {int((worst['F']>0).mean()*100)}% of them")
    # sub-period Sharpe stability
    print("\nSub-period Sharpe:")
    for lo, hi in [("2008","2012"),("2013","2017"),("2018","2021"),("2022","2026")]:
        s = sl.loc[f"{lo}-01-01":f"{hi}-12-31"]
        if len(s) > 60:
            print(f"  {lo}-{hi}: Sharpe {s.mean()/s.std()*np.sqrt(TD):.2f}  ann {((1+s.mean())**TD-1)*100:.1f}%")


if __name__ == "__main__":
    main()
