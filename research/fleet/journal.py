"""Journal writer. Persists every decision to Postgres (Neon) for the improvement
engine, and falls back to append-only JSONL on disk when DATABASE_URL is unset so
backtests and offline runs still produce a replayable record.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import FleetSignal


def _default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if hasattr(o, "value"):  # Enum
        return o.value
    return str(o)


class Journal:
    """Records every fleet decision to three sinks, layered:

    - `records`: an always-on in-memory buffer (table -> list of rows). Backtests
      and the improvement engine read this directly, so Tier A/B/C analysis is
      testable with no database. Set `keep_in_memory=False` for a long-lived live
      process that should not accumulate an unbounded buffer.
    - Postgres (Neon), when `DATABASE_URL` is set.
    - append-only JSONL on disk otherwise, so offline runs still leave a replayable
      trail.
    """

    def __init__(
        self,
        jsonl_dir: str | Path = "data/journal",
        keep_in_memory: bool = True,
    ) -> None:
        self.db_url = os.environ.get("DATABASE_URL")
        self.jsonl_dir = Path(jsonl_dir)
        self.keep_in_memory = keep_in_memory
        # In-memory mirror of every recorded row, keyed by table name.
        self.records: dict[str, list[dict[str, Any]]] = {}
        # Durable sink policy: a database always wins; otherwise we only touch disk
        # for a durable (non-in-memory) journal. A transient in-memory backtest
        # journal writes no files, so re-running a backtest never appends dupes.
        self.persist_jsonl = not self.db_url and not keep_in_memory
        if self.persist_jsonl:
            self.jsonl_dir.mkdir(parents=True, exist_ok=True)

    def _capture(self, table: str, row: dict[str, Any]) -> None:
        """Append a row to the in-memory buffer (if enabled)."""
        if self.keep_in_memory:
            self.records.setdefault(table, []).append(row)

    def _write_jsonl(self, table: str, row: dict[str, Any]) -> None:
        path = self.jsonl_dir / f"{table}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(row, default=_default) + "\n")

    def _write_pg(self, sql: str, params: tuple) -> None:
        import psycopg

        with psycopg.connect(self.db_url) as conn:  # type: ignore[arg-type]
            conn.execute(sql, params)
            conn.commit()

    def record_signal(
        self,
        signal: FleetSignal,
        action: str,
        reject_reason: str | None = None,
        regime: dict[str, Any] | str | None = None,
    ) -> None:
        """Log a signal AND its disposition. `action` is taken | rejected_*.

        Rejected signals are as valuable as taken ones — they are the
        counterfactuals the improvement engine needs. `regime` is the regime
        vector/label at decision time; Tier C measures regime-conditional edge
        from it, so it is captured on every signal, taken or not.
        """
        row = {
            "bot_id": signal.bot_id,
            "at": signal.at,
            "symbol": signal.symbol,
            "side": signal.side,
            "features": signal.features,
            "thesis": signal.thesis,
            "confidence": signal.confidence,
            "regime": regime,
            "action": action,
            "reject_reason": reject_reason,
        }
        self._capture("signals", row)
        if self.db_url:
            self._write_pg(
                """insert into signals
                   (bot_id, at, symbol, side, features, thesis, confidence, regime, action, reject_reason)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    signal.bot_id, signal.at, signal.symbol, signal.side,
                    json.dumps(signal.features, default=_default), signal.thesis,
                    signal.confidence, json.dumps(regime or {}, default=_default),
                    action, reject_reason,
                ),
            )
        elif self.persist_jsonl:
            self._write_jsonl("signals", row)

    def record_position_close(self, row: dict[str, Any]) -> None:
        """Log a closed position with MFE/MAE/exit-reason for post-mortems."""
        self._capture("positions", row)
        if self.db_url:
            self._write_pg(
                """insert into positions
                   (bot_id, symbol, opened_at, closed_at, entry_px, exit_px, qty,
                    max_favorable_bps, max_adverse_bps, exit_reason, pnl_usd, r_multiple)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                tuple(row.get(k) for k in (
                    "bot_id", "symbol", "opened_at", "closed_at", "entry_px",
                    "exit_px", "qty", "max_favorable_bps", "max_adverse_bps",
                    "exit_reason", "pnl_usd", "r_multiple",
                )),
            )
        elif self.persist_jsonl:
            self._write_jsonl("positions", row)

    def record_proposal(self, row: dict[str, Any]) -> None:
        """Log a Tier A/B proposal — the audit trail for every rule/param change.

        Rejections matter as much as approvals: a `human_reason` on a rejected
        proposal is fed back into future proposal prompts so the proposer stops
        re-suggesting what was already turned down.
        """
        self._capture("proposals", row)
        if self.db_url:
            self._write_pg(
                """insert into proposals
                   (bot_id, tier, created_at, description, diff, backtest_report,
                    status, human_reason, shadow_start, shadow_result)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    row.get("bot_id"), row.get("tier"), row.get("created_at"),
                    row.get("description"), row.get("diff"),
                    json.dumps(row.get("backtest_report") or {}, default=_default),
                    row.get("status"), row.get("human_reason"),
                    row.get("shadow_start"),
                    json.dumps(row.get("shadow_result") or {}, default=_default),
                ),
            )
        elif self.persist_jsonl:
            self._write_jsonl("proposals", row)
