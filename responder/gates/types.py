"""Shared typed inputs and outputs for responder gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Candidate:
    decision_id: str
    tier: str
    calibrated_p: float
    reason_class: str = "unknown"
    reasons: tuple[str, ...] = ()
    c_rto: float = 0.0
    impression_cost: float = 0.0
    effectiveness: float = 0.0
    message_class: str = "service"
    rendered_text: str = ""
    address: str = ""
    send_attempts: int = 0
    escalations: int = 0


@dataclass(frozen=True)
class Context:
    kill_switch: bool = False
    terminal: bool = False
    tier_actionable: bool = True
    dnd_scrubbed: bool = False
    consent_timestamp: str | None = None
    opt_out_keyword: bool = False
    recontact_allowed: bool = True
    send_window_open: bool = True
    merchant_daily_budget_available: bool = True
    customer_fatigued: bool = False
    consent_store: object | None = None
    reason_vocabulary: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    gate: str
    version: str
    passed: bool
    action: str | None = None
    reason: str | None = None


Gate = Callable[[Candidate, Context], GateResult]


def pass_result(name: str, version: str) -> GateResult:
    return GateResult(name, version, True)


def block_result(name: str, version: str, reason: str, action: str = "SUPPRESSED") -> GateResult:
    return GateResult(name, version, False, action, reason)
