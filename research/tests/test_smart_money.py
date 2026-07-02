"""13F smart-money parsing/diff tests, against saved real EDGAR filings.

Uses two archived Duquesne (Druckenmiller) information tables as fixtures so the
parse -> diff -> conviction pipeline is verified with zero network. The live
`fetch_manager_filings` path is exercised separately/manually.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.fleet.bots.catalyst import Catalyst
from research.fleet.smart_money import (
    Filing,
    conviction_signals,
    diff_filings,
    fundamentals_overlay,
    parse_information_table,
    ticker_conviction,
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


def test_ticker_conviction_maps_cusip_to_ticker():
    q1 = _load("duquesne_2026q1_infotable.xml")
    q0 = _load("duquesne_2025q4_infotable.xml")
    changes = diff_filings(q1, q0)
    # Build the CUSIP->ticker map from the real filing (a new buy and an exit).
    avgo = next(c.cusip for c in changes if "broadcom" in c.issuer.lower() and c.action == "new")
    googl = next(c.cusip for c in changes if "alphabet" in c.issuer.lower() and c.action == "exit")
    tc = ticker_conviction({"druckenmiller": [q1, q0]}, {avgo: "AVGO", googl: "GOOGL"})
    assert tc["AVGO"] == 1.0 and tc["GOOGL"] == -1.0
    # Unmapped CUSIPs are dropped, so only the two mapped tickers appear.
    assert set(tc) == {"AVGO", "GOOGL"}


def test_two_managers_agreement_reinforces():
    q1 = _load("duquesne_2026q1_infotable.xml")
    q0 = _load("duquesne_2025q4_infotable.xml")
    changes = diff_filings(q1, q0)
    avgo = next(c.cusip for c in changes if "broadcom" in c.issuer.lower() and c.action == "new")
    # Same manager filings under two keys => same +1 new buy => average stays +1.
    tc = ticker_conviction(
        {"a": [q1, q0], "b": [q1, q0]}, {avgo: "AVGO"}
    )
    assert tc["AVGO"] == 1.0


def test_catalyst_reads_institutional_tile():
    """The overlay feeds CATALYST's mosaic as a signed tile."""
    cat = Catalyst()
    # A flat price series so price-based tiles are near zero; the institutional
    # tile should then dominate the composite's sign.
    idx = pd.bdate_range("2023-01-01", periods=40)
    df = pd.DataFrame({"o": 100.0, "h": 100.0, "l": 100.0, "c": 100.0, "v": 1e6}, index=idx)

    overlay = fundamentals_overlay({"XYZ": 0.9})
    bull, comps_bull = cat._mosaic_score(df, overlay["XYZ"])
    assert "institutional_conviction" in comps_bull
    assert comps_bull["institutional_conviction"] == 0.9

    bear, _ = cat._mosaic_score(df, fundamentals_overlay({"XYZ": -0.9})["XYZ"])
    assert bull > bear                       # the tile moves the composite
    # Absent fundamentals => tile skipped, not imputed.
    _, comps_none = cat._mosaic_score(df, None)
    assert "institutional_conviction" not in comps_none
