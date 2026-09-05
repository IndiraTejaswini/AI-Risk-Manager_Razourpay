"""The proposal object emitted by gates and templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from responder.gates.types import GateResult
from responder.templates.registry import Tier
from responder.trust import Tagged


class MessageClass:
    SERVICE = "SERVICE"
    PROMOTIONAL = "PROMOTIONAL"


@dataclass(frozen=True)
class CandidateAction:
    decision_id: str
    tier: Tier
    template_id: str
    template_version: str
    message_class: str
    rendered: str
    fields: Mapping[str, Tagged]
    gate_trace: tuple[GateResult, ...]
