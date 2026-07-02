"""The improvement run — the fleet's evidence flywheel, executed end-to-end.

This is the job the operating cadence (§11) calls nightly/weekly: it backtests the
whole fleet while journaling every decision, then turns that journal into the four
things a human reviews:

  * **Tier C** — measured regime-conditional edge per bot, with add/drop deltas
    against each bot's declared favorable regimes.
  * **Tier B** — one bounded, falsifiable rule-change proposal per bot (recorded
    to the proposals table for approval).
  * **Meta-allocation** — risk-parity + quarter-Kelly tilt on *walk-forward* OOS
    Sharpe, with the correlation freeze check.
  * **Tier A** (``--tune``) — a walk-forward parameter re-fit per bot, gated on the
    §7b acceptance criteria.

Runs fully offline on synthetic data with no credentials, so the whole loop is
demonstrable in a fresh checkout:

    python -m research.scripts.improve
    python -m research.scripts.improve --tune --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from datetime import datetime

from research.fleet.allocator import allocate, run_fleet_backtest
from research.fleet.bots.arb import Arb
from research.fleet.bots.breakout import Breakout
from research.fleet.bots.catalyst import Catalyst
from research.fleet.bots.macro import Macro
from research.fleet.data import default_provider, load_panel
from research.fleet.improve import propose_rule_change, recommend_regimes
from research.fleet.journal import Journal
from research.fleet.promotion import promote_from_tuning, split_date
from research.fleet.theses import post_mortem, score_theses, theses_from_journal
from research.fleet.walkforward import fleet_oos_sharpe, tune

BOTS = {"breakout": Breakout, "arb": Arb, "catalyst": Catalyst, "macro": Macro}

UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
    "XLE", "OIH", "SMH", "XLF", "KRE", "GLD", "GDX", "XLK",
    "TLT", "IEF", "SLV", "UUP", "DBC", "USO", "EEM", "EWJ",
]


def _priors() -> dict[str, set[str]]:
    """Each bot's currently-declared favorable regimes (Tier-C baseline)."""
    return {name: {r.value for r in cls().favorable_regimes} for name, cls in BOTS.items()}


def _rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def run(start: datetime, end: datetime, total_usd: float, do_tune: bool,
        tune_candidates: int) -> None:
    provider = default_provider()
    print(f"Data provider: {type(provider).__name__}  |  window {start.date()} → {end.date()}")
    panel = load_panel(UNIVERSE, start, end, provider)
    print(f"Loaded {len(panel)} symbols across {len(BOTS)} bots")

    # 1) Fleet backtest, journaling every decision per bot. -------------------
    journals = {name: Journal(keep_in_memory=True) for name in BOTS}
    fleet = run_fleet_backtest(BOTS, panel, start, end, journals=journals)
    positions = [row for j in journals.values() for row in j.records.get("positions", [])]
    signal_counts = {
        name: len(j.records.get("signals", [])) for name, j in journals.items()
    }
    _rule("Journal")
    for name in BOTS:
        print(f"  {name:<9} {signal_counts[name]:>5} signals  "
              f"{len(journals[name].records.get('positions', [])):>4} closed trades")

    # 2) Tier C — measured regime-conditional edge. ---------------------------
    _rule("Tier C — regime-conditional deployment")
    rec = recommend_regimes(positions, favorable_priors=_priors())
    for name in BOTS:
        r = rec.get(name)
        if not r:
            print(f"  {name:<9} (no closed trades yet)")
            continue
        print(f"  {name:<9} trade in: {', '.join(r['recommended']) or '—'}")
        if r.get("add"):
            print(f"            + add {', '.join(r['add'])} (measured edge, not in priors)")
        if r.get("drop"):
            print(f"            - drop {', '.join(r['drop'])} (measured negative)")

    # 3) Tier B — one bounded rule-change proposal per bot. -------------------
    _rule("Tier B — proposed rule changes")
    proposals = Journal(keep_in_memory=True)
    for name in BOTS:
        prop = propose_rule_change(name, positions, created_at=end)
        if prop is None:
            print(f"  {name:<9} no proposal (insufficient evidence / no clean split)")
            continue
        proposals.record_proposal(prop.as_row())
        print(f"  {name:<9} {prop.description}")
    n_props = len(proposals.records.get("proposals", []))
    print(f"  → {n_props} proposal(s) recorded to the proposals table (status=draft)")

    # 3b) MACRO thesis scoring — falsify what was claimed. --------------------
    if "macro" in journals:
        _rule("MACRO theses — 60/120-day scoring & post-mortem")
        thesis_rows = theses_from_journal(journals["macro"].records.get("signals", []))
        scored = score_theses(thesis_rows, panel)
        for st in scored:
            journals["macro"].record_thesis(st.as_row())
        pm = post_mortem(scored)
        scorable = [t for t in scored if t.score_60d is not None]
        print(f"  {len(thesis_rows)} theses, {len(scorable)} scorable at 60d")
        if not pm:
            print("  (not enough per-driver history for a post-mortem yet)")
        for d in sorted(pm, key=lambda x: x.mean_score_60d):
            print(f"  {d.driver_key:<16} {d.verdict}")

    # 4) Meta-allocation on walk-forward OOS Sharpe. --------------------------
    _rule("Meta-allocation (risk-parity + quarter-Kelly tilt on OOS Sharpe)")
    print("  computing walk-forward OOS Sharpe per bot…")
    oos = fleet_oos_sharpe(BOTS, panel, start, end)
    alloc = allocate(fleet.returns, total_usd=total_usd, at=end, oos_sharpe=oos)
    for name in BOTS:
        print(f"  {name:<9} OOS Sharpe {oos[name]:+.2f}  →  "
              f"weight {alloc.weights[name]:>6.1%}  (${alloc.capital_usd[name]:,.0f})")
    print(f"  {alloc.reason}")

    # 5) Tier A — walk-forward re-fit, then shadow-validate on a held-out tail. -
    if do_tune:
        _rule("Tier A — walk-forward re-fit + shadow promotion")
        # Hold out the last ~year: the tuner fits on [start, split]; an accepted
        # challenger must then beat the champion on [split, end], which it never saw.
        split = split_date(panel, end, holdout_days=252)
        if split is None:
            print("  (not enough history to hold out a shadow window)")
        else:
            print(f"  tune on {start.date()}→{split.date()}, shadow on {split.date()}→{end.date()}")
            for name, cls in BOTS.items():
                prop = tune(cls, panel, start, split, n_candidates=tune_candidates)
                proposals.record_proposal(prop.as_row(created_at=end))
                verdict = "ACCEPT" if prop.accepted else "reject"
                print(f"  {name:<9} [{verdict}] {prop.reason}")
                if not prop.accepted:
                    continue
                changes = ", ".join(
                    f"{k}: {a:g}→{b:g}" for k, (a, b) in prop.diff().items()
                )
                print(f"            proposed: {changes}")
                shadow = promote_from_tuning(prop, cls, panel, split, end)
                decision = "PROMOTE" if shadow and shadow.promote else "hold champion"
                print(f"            shadow: [{decision}] {shadow.reason if shadow else ''}")
            n_after = len(proposals.records.get("proposals", []))
            print(f"  → proposals table now holds {n_after} record(s)")

    print("\nNote: synthetic-data results are a plumbing check, NOT an edge estimate.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--total", type=float, default=40_000.0, help="fleet capital USD")
    ap.add_argument("--tune", action="store_true", help="run the slow Tier-A re-fit")
    ap.add_argument("--tune-candidates", type=int, default=12)
    args = ap.parse_args()
    run(
        datetime.fromisoformat(args.start),
        datetime.fromisoformat(args.end),
        args.total,
        args.tune,
        args.tune_candidates,
    )


if __name__ == "__main__":
    main()
