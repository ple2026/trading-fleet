"""Trace the fleet namesakes' latest stock picks from SEC 13F filings.

Pulls the two most recent 13F-HR filings for Druckenmiller (Duquesne) and Cohen
(Point72) from EDGAR, prints the current top holdings, and diffs the quarters to
show new buys / adds / trims / exits — a lagged read on manager conviction.

    python -m research.scripts.smart_money
    python -m research.scripts.smart_money --manager cohen --ua "you you@email.com"

Reminder: 13F is longs-only, quarter-end, filed up to 45 days late. Idea filter,
not a copy-trade. Be polite to EDGAR — pass your own contact via --ua.
"""

from __future__ import annotations

import argparse

from research.fleet.smart_money import (
    DEFAULT_USER_AGENT,
    MANAGERS,
    diff_filings,
    fetch_manager_filings,
)


def _report(key: str, user_agent: str, top_n: int) -> None:
    m = MANAGERS[key]
    print(f"\n{'=' * 72}\n{m['fund']}  (CIK {m['cik']})\n{'=' * 72}")
    filings = fetch_manager_filings(m["cik"], n_quarters=2, user_agent=user_agent)
    if not filings:
        print("  no 13F-HR filings found")
        return
    latest = filings[0]
    period = latest.period.isoformat() if latest.period else "?"
    print(f"Latest 13F — period {period} — {len(latest.holdings)} positions, "
          f"~${latest.reported_usd() / 1e9:,.1f}B long book\n")
    print(f"  {'Top holdings':<38}{'weight':>8}{'shares':>14}")
    for h in latest.top(top_n):
        print(f"  {h.issuer[:36]:<38}{latest.weight(h.cusip) * 100:>7.1f}%{int(h.shares):>14,}")

    if len(filings) < 2:
        return
    changes = diff_filings(latest, filings[1])
    buckets: dict[str, list] = {}
    for c in changes:
        buckets.setdefault(c.action, []).append(c)

    prior = filings[1].period.isoformat() if filings[1].period else "?"
    print(f"\n  Changes vs prior quarter ({prior}):")
    for action, label in (("new", "NEW BUYS"), ("exit", "EXITED")):
        items = buckets.get(action, [])
        names = ", ".join(c.issuer for c in items[:10])
        print(f"    {label} ({len(items)}): {names}")
    print(f"    ADDED: {len(buckets.get('add', []))}   TRIMMED: {len(buckets.get('trim', []))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manager", choices=list(MANAGERS), default=None,
                    help="default: all managers")
    ap.add_argument("--ua", default=DEFAULT_USER_AGENT,
                    help="User-Agent for SEC (use your name + email)")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    for key in ([args.manager] if args.manager else list(MANAGERS)):
        _report(key, args.ua, args.top)


if __name__ == "__main__":
    main()
