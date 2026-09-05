"""Lease-based polling relay for the responder outbox."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from responder.states import State
from responder.transitions import transition


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


class Poller:
    """Claim and relay outbox rows without wedging failed work."""

    def __init__(self, connection, *, lease_seconds: int = 60):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.connection = connection
        self.lease_seconds = lease_seconds

    def claim(self, decision_id: str, *, now: datetime | None = None) -> bool:
        """Claim one unclaimed or expired row for the duration of its lease."""
        current = _utcnow() if now is None else now
        now_text = _timestamp(current)
        lease_text = _timestamp(current + timedelta(seconds=self.lease_seconds))
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE action_outbox "
                "SET claimed_until = ?, attempts = attempts + 1 "
                "WHERE decision_id = ? "
                "AND terminal = 0 "
                "AND (claimed_until IS NULL OR claimed_until < ?)",
                (lease_text, decision_id, now_text),
            )
        return cursor.rowcount == 1

    def claim_next(self, *, now: datetime | None = None) -> str | None:
        """Claim and return one eligible decision ID, if any exists."""
        current = _utcnow() if now is None else now
        now_text = _timestamp(current)
        row = self.connection.execute(
            "SELECT decision_id FROM action_outbox "
            "WHERE terminal = 0 "
            "AND (claimed_until IS NULL OR claimed_until < ?) "
            "ORDER BY created_at, decision_id LIMIT 1",
            (now_text,),
        ).fetchone()
        if row is None:
            return None
        decision_id = row[0]
        return decision_id if self.claim(decision_id, now=current) else None

    def abandon(self, decision_id: str, *, reason: str = "send failed") -> None:
        """Mark a failed decision terminal and release its temporary lease."""
        row = self.connection.execute(
            "SELECT state FROM decision_log WHERE decision_id = ? "
            "ORDER BY seq DESC LIMIT 1",
            (decision_id,),
        ).fetchone()
        if row is None:
            raise KeyError(decision_id)
        if row[0] != State.ABANDONED.value:
            transition(self.connection, decision_id, State.ABANDONED, "poller", reason)
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE action_outbox SET terminal = 1, claimed_until = NULL "
                "WHERE decision_id = ?",
                (decision_id,),
            )
        if cursor.rowcount != 1:
            raise KeyError(decision_id)

    def poll_once(self, send: Callable[[str], None]) -> str | None:
        """Claim one row, invoke the relay, and abandon on a permanent failure."""
        decision_id = self.claim_next()
        if decision_id is None:
            return None
        try:
            send(decision_id)
        except Exception:
            self.abandon(decision_id)
            raise
        return decision_id
