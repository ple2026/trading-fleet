"""First non-rigged out-of-sample run: walk-forward on a SURVIVORSHIP-FREE sample.

Every other backtest in this repo uses a hand-picked (survivor-biased) universe.
This one samples names that were *live during the window* from Tiingo's full
supported-ticker list — delisted ones included — so BREAKOUT/CATALYST finally see
the failures and buyouts, not just the winners. It then reports out-of-sample
metrics from the walk-forward harness (the first numbers here that aren't rigged by
hindsight — though still not edge until a paid liquidity/index-membership feed
replaces the raw ticker pool).

    python -m research.scripts.oos --bot breakout --sample 80
    python -m research.scripts.oos --bot catalyst --start 2017-01-01 --sample 120
"""

from __future__ import annotations

import argparse
from datetime import datetime

from research.fleet.bots.breakout import Breakout
from research.fleet.bots.catalyst import Catalyst
from research.fleet.universe import (
    filter_panel_by_liquidity,
    load_supported_tickers,
    sample_universe,
    survivorship_free_universe,
)
from research.fleet.walkforward import walk_forward

BOTS = {"breakout": Breakout, "catalyst": Catalyst}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _resilient_panel(symbols, start, end, provider=None):
    """Load Tiingo bars per symbol, skipping names that fail (rate-limit, no data,
    reused ticker). A partial universe still produces a valid backtest."""
    if provider is None:
        from research.fleet.data import TiingoProvider
        provider = TiingoProvider()
    panel, skipped = {}, 0
    for s in symbols:
        try:
            df = provider.daily_bars(s, start, end)
        except Exception:
            skipped += 1
            continue
        if len(df) > 200:          # need enough history to warm up indicators
            panel[s] = df
        else:
            skipped += 1
    return panel, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bot", choices=list(BOTS), default="breakout")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--sample", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-dv", type=float, default=0.0,
                    help="min median daily $ volume (liquidity screen; 0=off)")
    ap.add_argument("--fundamentals", choices=["none", "edgar"], default="none",
                    help="apply the CAN SLIM EPS gate from a point-in-time source")
    ap.add_argument("--circuit-pct", type=float, default=15.0,
                    help="per-bot drawdown circuit breaker %% (0=off)")
    args = ap.parse_args()
    _load_env()

    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    records = load_supported_tickers()
    free = survivorship_free_universe(records, start, end)
    picked = sample_universe(free, args.sample, args.seed)
    # SPY is the regime benchmark and must be present.
    symbols = sorted(set(["SPY", *picked]))
    delisted_end = {r.ticker: r.end_date for r in records}
    n_delisted = sum(
        1 for t in picked
        if delisted_end.get(t) is not None and delisted_end[t] < end.date()
    )

    print(f"Survivorship-free sample: {len(picked)} names "
          f"({n_delisted} already delisted before {end.date()}) + SPY")
    print("Loading Tiingo bars (cached after first run)…")
    panel, skipped = _resilient_panel(symbols, start, end)
    print(f"Loaded {len(panel)} symbols ({skipped} skipped: no data / rate-limit / reused ticker)")
    if args.min_dv > 0:
        before = len(panel)
        panel = filter_panel_by_liquidity(panel, args.min_dv)
        print(f"Liquidity screen (median $vol >= ${args.min_dv:,.0f}/day): "
              f"{len(panel)} kept, {before - len(panel)} illiquid dropped")
    if "SPY" not in panel:
        print("ERROR: SPY (benchmark) failed to load — cannot classify regime.")
        return

    fundamentals = None
    if args.fundamentals == "edgar":
        from research.fleet.edgar_fundamentals import edgar_fundamentals_panel
        names = [s for s in panel if s != "SPY"]
        fundamentals = edgar_fundamentals_panel(names)
        covered = sum(1 for s in names if s.upper() in fundamentals.frames)
        print(f"EDGAR CAN SLIM gate ON: {covered}/{len(names)} names have EPS history "
              f"(point-in-time, filed dates; uncovered names pass the gate ungated)")

    bot_class = BOTS[args.bot]
    cb = args.circuit_pct / 100.0 if args.circuit_pct > 0 else None
    cb_note = f", {args.circuit_pct:.0f}% drawdown circuit breaker" if cb else ""
    print(f"\nWalk-forward OOS for {args.bot} (3y train -> 6m test, rolling{cb_note})…")
    res = walk_forward(bot_class, panel, start, end, fundamentals=fundamentals,
                       circuit_breaker_pct=cb)
    m = res.metrics()
    if not m:
        print("(no OOS trades — sample too small / too illiquid)")
        return
    oos_span = (f"{res.oos_equity.index.min().date()}..{res.oos_equity.index.max().date()}"
                if not res.oos_equity.empty else "n/a")
    print(f"  OOS window : {oos_span} across {len(res.windows)} rolls")
    print(f"  {args.bot:<9} CAGR%={m['cagr']}  Sharpe={m['sharpe']}  MaxDD%={m['max_dd']}  "
          f"Trades={m['trades']}  Hit%={m['hit_rate']}  PF={m['profit_factor']}")

    # Two benchmarks over the SAME OOS window:
    #   SPY b&h   — cap-weighted index; the "just buy the index" bar.
    #   EW pond   — equal-weight buy-hold of the SAME liquid universe; the fair
    #               opportunity-set bar. The gap SPY - EW is the mega-cap premium
    #               the strategy never had access to, so beating EW (not SPY) is the
    #               honest test of stock SELECTION within its pond.
    spy = _spy_benchmark(panel["SPY"], res.oos_equity)
    if spy:
        print(f"  {'SPY b&h':<9} CAGR%={spy['cagr']}  Sharpe={spy['sharpe']}  "
              f"MaxDD%={spy['max_dd']}   <- cap-weighted index")
    ew = _ew_benchmark(panel, res.oos_equity)
    if ew:
        print(f"  {'EW pond':<9} CAGR%={ew['cagr']}  Sharpe={ew['sharpe']}  "
              f"MaxDD%={ew['max_dd']}   <- equal-weight SAME universe (fair bar for selection)")

    print("\nNote: survivorship-free prices, walk-forward OOS — bias-corrected. Beat EW "
          "pond to prove stock selection; beat SPY to prove the whole thing beats indexing.")


def _spy_benchmark(spy, oos_equity) -> dict | None:
    """Buy-and-hold SPY metrics over the OOS curve's span (same start/end days)."""
    import numpy as np
    if oos_equity is None or oos_equity.empty:
        return None
    c = spy["c"].loc[(spy.index >= oos_equity.index.min()) & (spy.index <= oos_equity.index.max())]
    if len(c) < 2:
        return None
    rets = c.pct_change().dropna()
    years = len(c) / 252
    cummax = c.cummax()
    return {
        "cagr": round(((c.iloc[-1] / c.iloc[0]) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(252)), 2) if rets.std() > 0 else 0.0,
        "max_dd": round(float(((c - cummax) / cummax).min()) * 100, 2),
    }


def _ew_benchmark(panel, oos_equity) -> dict | None:
    """Equal-weight, daily-rebalanced buy-and-hold of the loaded universe (ex-SPY)
    over the OOS span. Each day's return is the mean across names that traded that
    day, so a name that delists simply drops out — the fair 'average liquid name'
    opportunity set the strategy actually fished in."""
    import numpy as np
    import pandas as pd
    if oos_equity is None or oos_equity.empty:
        return None
    lo, hi = oos_equity.index.min(), oos_equity.index.max()
    cols = {}
    for sym, df in panel.items():
        if sym == "SPY":
            continue
        c = df["c"].loc[(df.index >= lo) & (df.index <= hi)]
        if len(c) > 1:
            cols[sym] = c.pct_change()
    if not cols:
        return None
    daily = pd.DataFrame(cols).mean(axis=1).dropna()   # equal-weight daily return
    if daily.empty:
        return None
    curve = (1 + daily).cumprod()
    years = len(curve) / 252
    cummax = curve.cummax()
    return {
        "cagr": round((curve.iloc[-1] ** (1 / years) - 1) * 100, 2) if years > 0 else 0.0,
        "sharpe": round(float(daily.mean() / daily.std() * np.sqrt(252)), 2) if daily.std() > 0 else 0.0,
        "max_dd": round(float(((curve - cummax) / cummax).min()) * 100, 2),
    }


if __name__ == "__main__":
    main()
