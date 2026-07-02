"""13F smart-money parsing/diff tests, against saved real EDGAR filings.

Uses two archived Duquesne (Druckenmiller) information tables as fixtures so the
parse -> diff -> conviction pipeline is verified with zero network. The live
`fetch_manager_filings` path is exercised separately/manually.
"""

from __future__ import annotations

from pathlib import Path

from research.fleet.smart_money import (
    Filing,
    conviction_signals,
    diff_filings,
    parse_information_table,
)

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> Filing:
    xml = (FIX / name).read_bytes()
    return Filing(cik="0001536411", period=None, holdings=parse_information_table(xml))


def test_parse_real_infotable():
    q1 = _load("duquesne_2026q1_infotable.xml")
    assert len(q1.holdings) == 68, "expected 68 aggregated positions in Q1-2026"
    assert q1.total_value > 0
    issuers = {h.issuer.lower() for h in q1.holdings.values()}
    assert any("natera" in i for i in issuers)          # known top holding
    # Weights form a distribution summing to 1.
    assert abs(sum(q1.weight(c) for c in q1.holdings) - 1.0) < 1e-6
    top = q1.top(1)[0]
    assert q1.weight(top.cusip) > 0.10                  # concentrated top name


def test_diff_recovers_new_buys_and_exits():
    q1 = _load("duquesne_2026q1_infotable.xml")
    q0 = _load("duquesne_2025q4_infotable.xml")
    changes = diff_filings(q1, q0)
    by_action: dict[str, list] = {}
    for c in changes:
        by_action.setdefault(c.action, []).append(c)

    # Every action bucket is populated for these two quarters.
    for a in ("new", "add", "trim", "exit"):
        assert by_action.get(a), f"no '{a}' changes found"

    new_issuers = {c.issuer.lower() for c in by_action["new"]}
    exit_issuers = {c.issuer.lower() for c in by_action["exit"]}
    assert any("broadcom" in i for i in new_issuers)    # a real Q1-2026 new buy
    assert any("alphabet" in i for i in exit_issuers)   # a real Q1-2026 exit

    # New buys have no prior shares; exits end at zero.
    assert all(c.prev_shares == 0 for c in by_action["new"])
    assert all(c.new_shares == 0 for c in by_action["exit"])
    # Changes are weight-ranked (newer quarter), so weights are non-increasing.
    weights = [c.new_weight for c in changes]
    assert weights == sorted(weights, reverse=True)


def test_conviction_signal_bounds_and_signs():
    q1 = _load("duquesne_2026q1_infotable.xml")
    q0 = _load("duquesne_2025q4_infotable.xml")
    changes = diff_filings(q1, q0)
    sig = conviction_signals(changes)
    assert all(-1.0 <= v <= 1.0 for v in sig.values())
    action_by_cusip = {c.cusip: c.action for c in changes}
    for cusip, v in sig.items():
        a = action_by_cusip[cusip]
        if a == "new":
            assert v == 1.0
        elif a == "exit":
            assert v == -1.0
        elif a == "add":
            assert v > 0
        elif a == "trim":
            assert v < 0
