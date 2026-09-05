"""The sole outbound dispatch boundary."""

from __future__ import annotations

from responder.candidate import CandidateAction
from responder.channels.base import Channel, validate


def dispatch(action: CandidateAction, channel: Channel) -> None:
    validate(action)
    channel.send(action)
