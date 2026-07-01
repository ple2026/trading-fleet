"""Backtest one or all bots over a date range on a shared price panel.

Runs fully offline against the synthetic data provider when no ALPACA_* creds are
set, so `python -m research.scripts.backtest` works in a fresh checkout.

Usage:
    python -m research.scripts.backtest                 # all bots, synthetic data
    python -m research.scripts.backtest --bot breakout  # one bot
    python -m research.scripts.backtest --start 2021-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
from datetime import datetime

from research.fleet.backtest import run_backtest
from research.fleet.bots.arb import Arb
from research.fleet.bots.breakout import Breakout
from research.fleet.bots.catalyst import Catalyst
from research.fleet.bots.macro import Macro
from research.fleet.data import default_provider, load_panel

BOTS = {"breakout": Breakout, "arb": Arb, "catalyst": Catalyst, "macro": Macro}

# A compact universe that covers every bot: leaders for BREAKOUT/CATALYST, ETF
# pairs for ARB, and macro expressions for MACRO. SPY is the regime benchmark.
UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
    "XLE", "OIH", "SMH", "XLF", "KRE", "GLD", "GDX", "XLK",
    "TLT", "IEF", "SLV", "UUP", "DBC", "USO", "EEM", "EWJ",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", choices=list(BOTS), default=None)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--equity", type=float, default=10_000.0)
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    provider = default_provider()
    print(f"Data provider: {type(provider).__name__}")
    panel = load_panel(UNIVERSE, start, end, provider)
    print(f"Loaded {len(panel)} symbols\n")

    chosen = [args.bot] if args.bot else list(BOTS)
    header = f"{'bot':<10}{'CAGR%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Trades':>8}{'Hit%':>7}{'PF':>8}"
    print(header)
    print("-" * len(header))
    for name in chosen:
        bot = BOTS[name]()
        result = run_backtest(bot, panel, start, end, starting_equity=args.equity)
        m = result.metrics()
        if not m:
            print(f"{name:<10}{'(no trades / insufficient data)':>40}")
            continue
        print(f"{name:<10}{m['cagr']:>8}{m['sharpe']:>8}{m['max_dd']:>9}"
              f"{m['trades']:>8}{m['hit_rate']:>7}{m['profit_factor']:>8}")

    print("\nNote: synthetic-data results are a plumbing check, NOT an edge estimate.")


if __name__ == "__main__":
    main()
