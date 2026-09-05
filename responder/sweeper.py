"""Response-window sweeper for sent responder decisions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from responder.states import State
from responder.transitions import TransitionError, transition


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Sweeper:
    """Close expired response windows, escalating at most once."""

    def __init__(
        self,
        connection,
        *,
        response_window: timedelta = timedelta(hours=24),
        dispatch_cutoff: datetime | None = None,
    ):
        if response_window.total_seconds() <= 0:
            raise ValueError("response_window must be positive")
        self.connection = connection
        self.response_window = response_window
        self.dispatch_cutoff = dispatch_cutoff

    def sweep(self, *, now: datetime | None = None) -> list[tuple[str, State]]:
        current = _now() if now is None else now
        rows = self.connection.execute(
            "SELECT decision_id, occurred_at FROM decision_log "
            "WHERE state = ? ORDER BY seq",
            (State.SENT.value,),
        ).fetchall()
        changed: list[tuple[str, State]] = []
        for decision_id, occurred_at in rows:
            if current < _parse(occurred_at) + self.response_window:
                continue
            if self.dispatch_cutoff is not None and current < self.dispatch_cutoff:
                target = State.ESCALATED
                reason = "response window expired; one escalation permitted"
            else:
                target = State.NO_RESPONSE
                reason = "response window expired after dispatch cutoff"
            try:
                transition(self.connection, decision_id, target, "sweeper", reason)
            except TransitionError:
                continue
            changed.append((decision_id, target))
        return changed
