"""Common channel protocol and validation."""

from __future__ import annotations

from typing import Protocol

from responder.candidate import CandidateAction


class Channel(Protocol):
    def send(self, action: CandidateAction) -> None: ...


def validate(action: CandidateAction) -> None:
    if not action.decision_id or not action.rendered:
        raise ValueError("candidate action is incomplete")
