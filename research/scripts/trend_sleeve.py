"""TREND — a vol-targeted, long/short, multi-asset time-series-momentum sleeve
(managed-futures / CTA replication) on liquid ETFs.

This is Step 2 of docs/HANDOFF.md: a *higher-quality* defensive sleeve than MACRO.
The design is the textbook Moskowitz-Ooi-Pedersen TSMOM, applied BLIND with
literature-standard parameters (no fit to this data => out-of-sample by construction):

  * signal_i   = mean( sign(3m ret), sign(6m ret), sign(12m ret) )  in [-1, +1]
  * size_i     = signal_i / vol_i        (inverse-vol; each name ~equal risk)
  * portfolio  = scaled so ex-ante vol (covariance-based) hits TARGET_VOL
  * long AND short (shorting the ETF); monthly rebalance; costs charged.
  * strictly point-in-time: every signal/vol/cov uses data BEFORE the rebalance day.

Leverage is allowed (the user can express it via 2-3x ETFs / margin); we cap gross
notional at MAX_GROSS and note the implied leverage.

    python -m research.scripts.trend_sleeve --start 2004-01-01 --end 2026-07-01

Writes data/trend_oos_equity.csv (date, trend_equity) and prints standalone metrics
plus correlation to v20 and SPY.
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from research.fleet.data import TiingoProvider

TRADING_DAYS = 252

# --- FIXED textbook parameters (NOT tuned to this data) ---------------------
LOOKBACKS = [63, 126, 252]     # 3 / 6 / 12-month trend
VOL_WINDOW = 63                # ~3-month ex-ante vol / covariance window
TARGET_VOL = 0.15              # 15% annualized portfolio vol (standard CTA)
MAX_GROSS = 5.0                # cap gross notional (leverage ETFs allowed)
COST_BPS = 10.0               # per unit turnover, one-way-ish (liquid ETFs)

# Diversified ETF universe across asset classes. Breadth is what makes TSMOM work;
# ETF-land is thinner than a real futures book (hence a degraded CTA ceiling).
INSTRUMENTS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",   # equities (US + intl)
    "TLT", "IEF",                          # rates
    "GLD", "SLV", "DBC", "USO", "DBA",     # commodities
    "UUP",                                  # USD
]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _load_prices(symbols, start, end) -> pd.DataFrame:
    prov = TiingoProvider()
    cols = {}
    for s in symbols:
        try:
            df = prov.daily_bars(s, start, end)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {s}: {str(e)[:50]}")
            continue
        if len(df) > 260:
            cols[s] = df["c"]
    px = pd.DataFrame(cols).sort_index()
    return px


def _metrics(curve: pd.Series) -> dict:
    rets = curve.pct_change().dropna()
    years = len(curve) / TRADING_DAYS
    cummax = curve.cummax()
    dd = float(((curve - cummax) / cummax).min())
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else 0.0
    cagr = float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1)
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": dd, "vol": vol}


def build_trend(px: pd.DataFrame, start: datetime) -> pd.Series:
    """Monthly-rebalanced, point-in-time vol-targeted TSMOM. Returns a daily equity
    curve seeded at 100k over [start, end]."""
    rets = px.pct_change()
    dates = px.index
    # month-start rebalance days within the tradeable span
    month_key = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).first()
    rebal_days = [d for d in month_key.values if d >= pd.Timestamp(start)]

    weights = pd.Series(0.0, index=px.columns)
    sleeve_ret = pd.Series(0.0, index=dates, dtype=float)
    rebal_set = set(rebal_days)
    gross_log = []

    for i, d in enumerate(dates):
        if d < pd.Timestamp(start):
            continue
        cost = 0.0
        if d in rebal_set:
            hist_px = px.loc[:dates[i - 1]] if i > 0 else px.iloc[:0]
            hist_ret = rets.loc[:dates[i - 1]]
            if len(hist_px) > max(LOOKBACKS) + 5:
                new_w = _target_weights(hist_px, hist_ret)
                cost = float((new_w - weights).abs().sum()) * COST_BPS / 10_000.0
                weights = new_w
                gross_log.append(float(weights.abs().sum()))
        # daily return from weights set at/prior to today (no lookahead: weights
        # use data strictly before their rebalance day), less rebalance turnover cost
        r = float((weights * rets.loc[d]).fillna(0.0).sum()) - cost
        sleeve_ret.loc[d] = r

    curve = 100_000 * (1 + sleeve_ret.loc[sleeve_ret.index >= pd.Timestamp(start)]).cumprod()
    if gross_log:
        print(f"  avg gross leverage: {np.mean(gross_log):.2f}x  "
              f"(max {np.max(gross_log):.2f}x, cap {MAX_GROSS}x)")
    return curve


def _target_weights(hist_px: pd.DataFrame, hist_ret: pd.DataFrame) -> pd.Series:
    """Vol-targeted TSMOM weights from history ending strictly before the rebal day."""
    last = hist_px.iloc[-1]
    # instruments with enough history for the longest lookback and a vol window
    valid = [c for c in hist_px.columns
             if hist_px[c].notna().sum() > max(LOOKBACKS) + VOL_WINDOW]
    if not valid:
        return pd.Series(0.0, index=hist_px.columns)

    sig = {}
    vol = {}
    for c in valid:
        s = 0.0
        for lb in LOOKBACKS:
            past = hist_px[c].iloc[-lb - 1]
            if past and not np.isnan(past):
                s += np.sign(last[c] / past - 1.0)
        sig[c] = s / len(LOOKBACKS)
        v = hist_ret[c].iloc[-VOL_WINDOW:].std() * np.sqrt(TRADING_DAYS)
        vol[c] = v if v and not np.isnan(v) and v > 0 else np.nan

    raw = pd.Series({c: sig[c] / vol[c] for c in valid if not np.isnan(vol[c])})
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty or raw.abs().sum() == 0:
        return pd.Series(0.0, index=hist_px.columns)

    # ex-ante portfolio vol from trailing covariance -> scale to TARGET_VOL
    cov = hist_ret[raw.index].iloc[-VOL_WINDOW:].cov() * TRADING_DAYS
    w = raw.values
    port_var = float(w @ cov.values @ w)
    port_vol = np.sqrt(port_var) if port_var > 0 else np.nan
    scale = TARGET_VOL / port_vol if port_vol and not np.isnan(port_vol) else 0.0
    weights = raw * scale
    gross = weights.abs().sum()
    if gross > MAX_GROSS:                    # cap leverage
        weights = weights * (MAX_GROSS / gross)
    return weights.reindex(hist_px.columns).fillna(0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2004-01-01")   # 3y warmup -> trades from 2007
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--trade-start", default="2007-01-01")
    ap.add_argument("--out", default="data/trend_oos_equity.csv")
    ap.add_argument("--no-equity", action="store_true",
                    help="drop equity ETFs — a cleaner diversifier to an equity-beta book")
    ap.add_argument("--symbols", default=None,
                    help="comma-separated universe override (v1 momentum signal)")
    args = ap.parse_args()
    _load_env()

    universe = INSTRUMENTS
    if args.symbols:
        universe = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.no_equity:
        equities = {"SPY", "QQQ", "IWM", "EFA", "EEM"}
        universe = [s for s in INSTRUMENTS if s not in equities]

    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    tstart = datetime.fromisoformat(args.trade_start)
    print(f"Loading {len(universe)} ETFs {start.date()}..{end.date()}…")
    px = _load_prices(universe, start, end)
    print(f"  loaded {list(px.columns)}")

    curve = build_trend(px, tstart)
    curve.to_frame("trend_equity").to_csv(args.out, index_label="date")
    print(f"\nWrote {len(curve)} rows -> {args.out}")

    m = _metrics(curve)
    print(f"\n{'TREND sleeve':<16}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Vol%':>8}")
    print(f"{'standalone':<16}{m['cagr']*100:>8.1f}{m['sharpe']:>8.2f}"
          f"{m['maxdd']*100:>9.1f}{m['vol']*100:>8.1f}")
    print(f"span {curve.index.min().date()}..{curve.index.max().date()}")

    # correlation to v20 + SPY (the acceptance bar: Sharpe>0.5 AND |corr|<0.3)
    try:
        v20 = pd.read_csv("/Users/phuongle/Documents/Trading Agent/backtest-output/"
                          "v20/equity-layers.csv", encoding="utf-8-sig")
        v20["date"] = pd.to_datetime(v20["date"], format="mixed")
        v20 = v20.set_index("date")["v20_equity"].sort_index()
        spy = px["SPY"] if "SPY" in px else None
        j = pd.DataFrame({"TREND": curve, "v20": v20, "SPY": spy}).dropna()
        jr = j.pct_change().dropna()
        print(f"\ncorr(TREND, v20) = {jr['TREND'].corr(jr['v20']):+.3f}")
        print(f"corr(TREND, SPY) = {jr['TREND'].corr(jr['SPY']):+.3f}")
        print(f"\nAcceptance bar: OOS Sharpe > 0.5 AND |corr| < 0.3  ->  "
              f"Sharpe {m['sharpe']:.2f}, |corr v20| {abs(jr['TREND'].corr(jr['v20'])):.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"(corr check skipped: {e})")


if __name__ == "__main__":
    main()
