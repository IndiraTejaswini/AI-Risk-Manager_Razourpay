"""The responder state machine declaration."""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    SCORED = "SCORED"
    SUPPRESSED = "SUPPRESSED"
    HELD_EXPLORATION = "HELD_EXPLORATION"
    QUEUED = "QUEUED"
    SENT = "SENT"
    CONFIRMED = "CONFIRMED"
    CANCELLED_AT_PROMPT = "CANCELLED_AT_PROMPT"
    NO_RESPONSE = "NO_RESPONSE"
    ESCALATED = "ESCALATED"
    SEND_FAILED = "SEND_FAILED"
    ABANDONED = "ABANDONED"


TERMINAL = frozenset({
    State.SUPPRESSED,
    State.HELD_EXPLORATION,
    State.CONFIRMED,
    State.CANCELLED_AT_PROMPT,
    State.NO_RESPONSE,
    State.ABANDONED,
})

TRANSITIONS = {
    State.SCORED: frozenset({State.SUPPRESSED, State.HELD_EXPLORATION, State.QUEUED}),
    State.QUEUED: frozenset({State.SENT, State.ABANDONED}),
    State.SENT: frozenset({
        State.CONFIRMED, State.CANCELLED_AT_PROMPT, State.NO_RESPONSE,
        State.ESCALATED, State.SEND_FAILED, State.ABANDONED,
    }),
    State.ESCALATED: frozenset({State.SENT, State.ABANDONED}),
    State.SEND_FAILED: frozenset({State.QUEUED, State.ABANDONED}),
    **{state: frozenset() for state in TERMINAL},
}

MAX_ESCALATIONS = 1
MAX_SEND_ATTEMPTS = 3
