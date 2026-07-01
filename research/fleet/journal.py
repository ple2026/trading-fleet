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
    def __init__(self, jsonl_dir: str | Path = "data/journal") -> None:
        self.db_url = os.environ.get("DATABASE_URL")
        self.jsonl_dir = Path(jsonl_dir)
        if not self.db_url:
            self.jsonl_dir.mkdir(parents=True, exist_ok=True)

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
        self, signal: FleetSignal, action: str, reject_reason: str | None = None
    ) -> None:
        """Log a signal AND its disposition. `action` is taken | rejected_*.

        Rejected signals are as valuable as taken ones — they are the
        counterfactuals the improvement engine needs.
        """
        row = {
            "bot_id": signal.bot_id,
            "at": signal.at,
            "symbol": signal.symbol,
            "side": signal.side,
            "features": signal.features,
            "thesis": signal.thesis,
            "confidence": signal.confidence,
            "action": action,
            "reject_reason": reject_reason,
        }
        if self.db_url:
            self._write_pg(
                """insert into signals
                   (bot_id, at, symbol, side, features, thesis, confidence, regime, action, reject_reason)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    signal.bot_id, signal.at, signal.symbol, signal.side,
                    json.dumps(signal.features, default=_default), signal.thesis,
                    signal.confidence, json.dumps({}, default=_default),
                    action, reject_reason,
                ),
            )
        else:
            self._write_jsonl("signals", row)

    def record_position_close(self, row: dict[str, Any]) -> None:
        """Log a closed position with MFE/MAE/exit-reason for post-mortems."""
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
        else:
            self._write_jsonl("positions", row)
