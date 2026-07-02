"""13F "smart-money" signal — reconstruct legendary managers' quarterly picks.

Two of the fleet's namesakes still run reportable books: Druckenmiller (Duquesne
Family Office) and Cohen (Point72 Asset Management). Their **13F-HR** filings are
public on SEC EDGAR. By parsing each quarter's information table and diffing
consecutive quarters we recover what they *newly bought, added to, trimmed, or
exited* — a genuine (if lagged) read on where these managers are putting conviction.

Honest limitations — read before trusting this:
  * 13F is **longs only** (no shorts, no hedges, options shown as notional holdings).
  * It is a **quarter-END snapshot filed up to 45 days later**, so a "new buy" you
    see here could already be up 40% and half-sold. Treat it as an *idea filter /
    one weak mosaic tile*, never a copy-trade or a training label. §7e still rules:
    it earns weight only if the journal shows it adds measured, out-of-sample edge.

The parsing/diff/scoring functions are pure and unit-tested against saved fixtures.
`fetch_manager_filings` is the only part that touches the network (opt-in, polite
User-Agent per SEC guidance).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date

# The reportable funds behind two of the fleet's namesakes (SEC CIKs).
MANAGERS: dict[str, dict[str, str]] = {
    "druckenmiller": {"cik": "0001536411", "fund": "Duquesne Family Office LLC"},
    "cohen": {"cik": "0001603466", "fund": "Point72 Asset Management, L.P."},
}

# SEC asks automated clients to identify themselves; override with your own contact.
DEFAULT_USER_AGENT = "trading-fleet-research contact@example.com"


@dataclass(frozen=True)
class Holding:
    cusip: str
    issuer: str
    shares: float
    value: float          # as reported (thousands of USD); unit cancels in weights


@dataclass
class Filing:
    cik: str
    period: date | None
    holdings: dict[str, Holding] = field(default_factory=dict)  # keyed by CUSIP

    @property
    def total_value(self) -> float:
        return sum(h.value for h in self.holdings.values())

    def weight(self, cusip: str) -> float:
        tot = self.total_value
        return self.holdings[cusip].value / tot if tot > 0 and cusip in self.holdings else 0.0

    def top(self, n: int = 10) -> list[Holding]:
        return sorted(self.holdings.values(), key=lambda h: -h.value)[:n]

    def reported_usd(self) -> float:
        """Total long book in whole dollars, normalizing the filer's value unit.

        The 13F `value` column is thousands of dollars for some filers and whole
        dollars for others (the 2023 amendment was adopted unevenly). We infer the
        unit from the median implied share price — a per-share value under ~$5
        means the column is in thousands — and scale to dollars. Weights never
        depend on this; only the headline total does.
        """
        prices = sorted(h.value / h.shares for h in self.holdings.values() if h.shares > 0)
        if not prices:
            return self.total_value
        median_px = prices[len(prices) // 2]
        scale = 1_000.0 if median_px < 5.0 else 1.0
        return self.total_value * scale


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def parse_information_table(xml_text: str | bytes) -> dict[str, Holding]:
    """Parse a 13F information table into {cusip: Holding}, summing duplicate rows.

    A manager often reports one issuer across several rows (share classes, multiple
    managers, put/call legs); we aggregate them by CUSIP into a single position.
    Namespaces vary by filer, so tags are matched by local name.
    """
    root = ET.fromstring(xml_text)
    holds: dict[str, Holding] = {}
    for it in root.iter():
        if it.tag.split("}")[-1] != "infoTable":
            continue
        by_name = {c.tag.split("}")[-1]: c for c in it}
        cusip = _text(by_name.get("cusip")).upper()
        if not cusip:
            continue
        issuer = _text(by_name.get("nameOfIssuer"))
        try:
            value = float(_text(by_name.get("value")) or 0.0)
        except ValueError:
            value = 0.0
        sh_el = it.find(".//{*}sshPrnamt")
        try:
            shares = float(_text(sh_el)) if sh_el is not None else 0.0
        except ValueError:
            shares = 0.0
        if cusip in holds:
            prev = holds[cusip]
            holds[cusip] = Holding(cusip, prev.issuer or issuer,
                                   prev.shares + shares, prev.value + value)
        else:
            holds[cusip] = Holding(cusip, issuer, shares, value)
    return holds


@dataclass(frozen=True)
class HoldingChange:
    cusip: str
    issuer: str
    action: str           # new | add | trim | exit
    prev_shares: float
    new_shares: float
    new_weight: float     # portfolio weight in the newer quarter (0 for exits)

    @property
    def shares_delta(self) -> float:
        return self.new_shares - self.prev_shares

    @property
    def pct_change(self) -> float:
        """Fractional change in share count (inf-safe; 1.0 for a brand-new buy)."""
        if self.prev_shares <= 0:
            return 1.0
        return (self.new_shares - self.prev_shares) / self.prev_shares


def diff_filings(newer: Filing, older: Filing) -> list[HoldingChange]:
    """Quarter-over-quarter position changes, newest-quarter weight-ranked."""
    changes: list[HoldingChange] = []
    for cusip, h in newer.holdings.items():
        prev = older.holdings.get(cusip)
        if prev is None:
            action = "new"
        elif h.shares > prev.shares:
            action = "add"
        elif h.shares < prev.shares:
            action = "trim"
        else:
            continue  # unchanged
        changes.append(HoldingChange(
            cusip, h.issuer, action,
            prev.shares if prev else 0.0, h.shares, newer.weight(cusip),
        ))
    for cusip, h in older.holdings.items():
        if cusip not in newer.holdings:
            changes.append(HoldingChange(cusip, h.issuer, "exit", h.shares, 0.0, 0.0))
    return sorted(changes, key=lambda c: -c.new_weight)


# Conviction weight per action — a fresh position is the loudest signal, an exit the
# most bearish. Adds/trims are scaled by how much the stake moved.
_ACTION_BASE = {"new": 1.0, "add": 0.5, "trim": -0.5, "exit": -1.0}


def conviction_signals(changes: list[HoldingChange]) -> dict[str, float]:
    """Map position changes to a signed conviction score per CUSIP in roughly
    [-1, 1]. New buys and exits are the extremes; adds/trims scale with size.

    This is the raw "smart-money tile" a bot's mosaic (e.g. CATALYST) can consume
    once CUSIPs are mapped to tickers — not a standalone trade trigger.
    """
    out: dict[str, float] = {}
    for c in changes:
        base = _ACTION_BASE[c.action]
        if c.action in ("add", "trim"):
            # Scale by fractional stake change, capped so a doubling saturates.
            base *= min(abs(c.pct_change), 1.0)
        out[c.cusip] = max(-1.0, min(1.0, base))
    return out


# --------------------------------------------------------------- network (opt-in)

def fetch_manager_filings(
    cik: str,
    n_quarters: int = 2,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[Filing]:
    """Fetch and parse a manager's most recent `n_quarters` 13F-HR filings from
    SEC EDGAR. Network + `requests` required; everything else in this module is
    pure and offline. Newest filing first.
    """
    import requests

    cik = cik.zfill(10)
    headers = {"User-Agent": user_agent}
    sub = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers, timeout=30
    )
    sub.raise_for_status()
    recent = sub.json()["filings"]["recent"]
    forms, accs, periods = (
        recent["form"], recent["accessionNumber"], recent["reportDate"],
    )
    picked = [i for i, f in enumerate(forms) if f.startswith("13F-HR")][:n_quarters]

    filings: list[Filing] = []
    for i in picked:
        acc_nodash = accs[i].replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}"
        idx = requests.get(f"{base}/", headers=headers, timeout=30)
        idx.raise_for_status()
        # The information table is the XML that is not primary_doc.xml.
        xml_name = None
        for token in idx.text.split('"'):
            low = token.lower()
            if low.endswith(".xml") and "primary_doc" not in low:
                xml_name = token.split("/")[-1]
                break
        if xml_name is None:
            continue
        info = requests.get(f"{base}/{xml_name}", headers=headers, timeout=30)
        info.raise_for_status()
        period = date.fromisoformat(periods[i]) if periods[i] else None
        filings.append(Filing(cik=cik, period=period,
                              holdings=parse_information_table(info.content)))
    return filings
