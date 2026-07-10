"""TREND_NE2 — pre-registered improvement of the non-equity trend sleeve.

Two STRUCTURAL changes decided a priori (no tuning to this dataset):

  1. Breadth: ~2x the non-equity universe — short/intl/EM/inflation bonds, natgas,
     and FX — so the sleeve doesn't lean ~30% on gold.
  2. Signal ensemble: equal-weight the three CANONICAL trend signals instead of one —
       * multi-horizon momentum : mean(sign 3/6/12m return)   (the v1 signal)
       * MA crossover           : sign(SMA20 - SMA100)
       * channel/breakout       : sign(close - midpoint of 252d high/low)

Everything else is identical to v1 (inverse-vol sizing, covariance-based 15% vol
target, monthly rebalance, 10 bps cost, point-in-time, gross cap 5x). One build,
keep-or-kill vs TREND_NE v1.

    python -m research.scripts.trend_sleeve_v2
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from research.fleet.data import TiingoProvider
from research.scripts.trend_sleeve import _load_prices, _metrics

TRADING_DAYS = 252
LOOKBACKS = [63, 126, 252]
MA_FAST, MA_SLOW, CHANNEL = 20, 100, 252
VOL_WINDOW = 63
TARGET_VOL = 0.15
MAX_GROSS = 5.0
COST_BPS = 10.0

# Wider non-equity universe (rates/bonds x6, commodities x6, FX x4).
WIDE_NONEQ = [
    "TLT", "IEF", "SHY", "TIP", "BWX", "EMB",        # rates / bonds
    "GLD", "SLV", "DBC", "USO", "UNG", "DBA",         # commodities
    "UUP", "FXE", "FXY", "FXB",                        # FX
]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _ensemble_signal(close: pd.Series) -> float:
    """Equal-weight of the three canonical trend signals, in [-1, 1]. Uses only the
    history in `close` (caller passes data strictly before the rebalance day)."""
    if len(close) < CHANNEL + 1:
        return 0.0
    c = close.iloc[-1]
    # 1) multi-horizon momentum
    mom = np.mean([np.sign(c / close.iloc[-lb - 1] - 1.0) for lb in LOOKBACKS
                   if not np.isnan(close.iloc[-lb - 1])])
    # 2) MA crossover
    ma = np.sign(close.iloc[-MA_FAST:].mean() - close.iloc[-MA_SLOW:].mean())
    # 3) channel position (breakout)
    hi, lo = close.iloc[-CHANNEL:].max(), close.iloc[-CHANNEL:].min()
    ch = np.sign(c - (hi + lo) / 2.0)
    return float((mom + ma + ch) / 3.0)


def _target_weights_v2(hist_px: pd.DataFrame, hist_ret: pd.DataFrame) -> pd.Series:
    valid = [col for col in hist_px.columns
             if hist_px[col].notna().sum() > CHANNEL + VOL_WINDOW]
    if not valid:
        return pd.Series(0.0, index=hist_px.columns)
    sig, vol = {}, {}
    for col in valid:
        sig[col] = _ensemble_signal(hist_px[col].dropna())
        v = hist_ret[col].iloc[-VOL_WINDOW:].std() * np.sqrt(TRADING_DAYS)
        vol[col] = v if v and not np.isnan(v) and v > 0 else np.nan
    raw = pd.Series({c: sig[c] / vol[c] for c in valid if not np.isnan(vol[c])})
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty or raw.abs().sum() == 0:
        return pd.Series(0.0, index=hist_px.columns)
    cov = hist_ret[raw.index].iloc[-VOL_WINDOW:].cov() * TRADING_DAYS
    w = raw.values
    pv = float(w @ cov.values @ w)
    scale = TARGET_VOL / np.sqrt(pv) if pv > 0 else 0.0
    weights = raw * scale
    g = weights.abs().sum()
    if g > MAX_GROSS:
        weights = weights * (MAX_GROSS / g)
    return weights.reindex(hist_px.columns).fillna(0.0)


def build(px: pd.DataFrame, start: datetime) -> pd.Series:
    rets = px.pct_change()
    dates = px.index
    mk = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first()
    rebal = {d for d in mk.values if d >= pd.Timestamp(start)}
    w = pd.Series(0.0, index=px.columns)
    sr = pd.Series(0.0, index=dates)
    gl = []
    for i, d in enumerate(dates):
        if d < pd.Timestamp(start):
            continue
        cost = 0.0
        if d in rebal:
            hpx, hret = px.loc[:dates[i - 1]], rets.loc[:dates[i - 1]]
            if len(hpx) > CHANNEL + 5:
                nw = _target_weights_v2(hpx, hret)
                cost = float((nw - w).abs().sum()) * COST_BPS / 10_000.0
                w = nw
                gl.append(float(w.abs().sum()))
        sr.loc[d] = float((w * rets.loc[d]).fillna(0.0).sum()) - cost
    if gl:
        print(f"  avg gross leverage: {np.mean(gl):.2f}x (max {np.max(gl):.2f}x)")
    return 100_000 * (1 + sr.loc[sr.index >= pd.Timestamp(start)]).cumprod()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--trade-start", default="2007-01-01")
    ap.add_argument("--out", default="data/trend_ne2_oos_equity.csv")
    ap.add_argument("--narrow", action="store_true",
                    help="ensemble signal on the ORIGINAL 8 non-equity ETFs (isolate "
                         "the ensemble from the breadth expansion)")
    args = ap.parse_args()
    _load_env()

    universe = ["TLT", "IEF", "GLD", "SLV", "DBC", "USO", "DBA", "UUP"] if args.narrow \
        else WIDE_NONEQ
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    print(f"Loading {len(universe)} ETFs…")
    px = _load_prices(universe, start, end)
    print(f"  loaded {list(px.columns)}")
    curve = build(px, datetime.fromisoformat(args.trade_start))
    curve.to_frame("trend_equity").to_csv(args.out, index_label="date")
    print(f"Wrote {len(curve)} rows -> {args.out}")

    m = _metrics(curve)
    print(f"\n{'TREND_NE2':<16}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    print(f"{'standalone':<16}{m['cagr']*100:>8.1f}{m['sharpe']:>8.2f}"
          f"{m['maxdd']*100:>9.1f}{m['vol']*100:>8.1f}")

    v20 = pd.read_csv("/Users/phuongle/Documents/Trading Agent/backtest-output/"
                      "v20/equity-layers.csv", encoding="utf-8-sig")
    v20["date"] = pd.to_datetime(v20["date"], format="mixed")
    v20 = v20.set_index("date")["v20_equity"].sort_index()
    j = pd.DataFrame({"NE2": curve, "v20": v20}).dropna().pct_change().dropna()
    print(f"corr(TREND_NE2, v20) = {j['NE2'].corr(j['v20']):+.3f}   "
          f"(v1 was -0.11)   bar: Sharpe>0.5 & |corr|<0.3")


if __name__ == "__main__":
    main()
