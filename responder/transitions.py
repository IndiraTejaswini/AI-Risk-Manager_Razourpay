"""The sole writer for decision-log state transitions."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import json

from responder.templates.registry import reason_class
from responder.states import (
    MAX_ESCALATIONS,
    MAX_SEND_ATTEMPTS,
    TERMINAL,
    TRANSITIONS,
    State,
)


class TransitionError(ValueError):
    """A requested state change violates the declared machine."""


def start_decision(connection, *, decision_id, order_id, account_id, response, state_reason):
    reasons = list(response.reasons)
    top_reason_class = reason_class(reasons[0]).value if reasons else "order_structure"
    values = (
        decision_id, order_id, account_id, response.model_version, "policy-v1",
        "cost-v1", "effectiveness-v1", "gates-v1", None, response.risk,
        response.tier, response.threshold_used, 0.0, 0.0, response.cost_breakdown.expected_rto_loss,
        top_reason_class, json.dumps(reasons), response.features_missing, State.SCORED.value, state_reason,
        "scorer", datetime.now(timezone.utc).isoformat(),
    )
    with connection:
        connection.execute(
            "INSERT INTO decision_log("
            "decision_id, order_id, account_id, model_version, policy_version, "
            "cost_constants_id, effectiveness_prior_id, gate_set_version, template_version, "
            "calibrated_p, tier, threshold_used, c_fp_impression, c_fp_triggered, c_fn, "
            "top_reason_class, reasons_json, features_missing, state, state_reason, actor, "
            "occurred_at) VALUES (" + ",".join("?" for _ in values) + ")",
            values,
        )


def transition(connection, decision_id: str, to_state: State, actor: str, reason: str | None):
    row = connection.execute(
        "SELECT * FROM decision_log WHERE decision_id = ? ORDER BY seq DESC LIMIT 1",
        (decision_id,),
    ).fetchone()
    if row is None:
        raise TransitionError(f"decision {decision_id!r} has no current state")
    columns = [item[1] for item in connection.execute("PRAGMA table_info(decision_log)")]
    current = dict(zip(columns, row))
    from_state = State(current["state"])
    to_state = State(to_state)
    if from_state in TERMINAL:
        raise TransitionError(f"{from_state.value} is terminal")
    if to_state not in TRANSITIONS[from_state]:
        raise TransitionError(f"{from_state.value} -> {to_state.value} is not allowed")
    history = connection.execute(
        "SELECT state FROM decision_log WHERE decision_id = ?", (decision_id,)
    ).fetchall()
    if to_state is State.ESCALATED and sum(state == State.ESCALATED.value for (state,) in history) >= MAX_ESCALATIONS:
        raise TransitionError("escalation limit reached")
    send_attempts = sum(state == State.SENT.value for (state,) in history)
    if to_state is State.SENT and send_attempts >= MAX_SEND_ATTEMPTS:
        to_state = State.ABANDONED
        reason = "send attempts exhausted"

    now = datetime.now(timezone.utc).isoformat()
    values = [
        current["decision_id"], current["order_id"], current["account_id"],
        current["model_version"], current["policy_version"], current["cost_constants_id"],
        current["effectiveness_prior_id"], current["gate_set_version"],
        current["template_version"], current["calibrated_p"], current["tier"],
        current["threshold_used"], current["c_fp_impression"], current["c_fp_triggered"],
        current["c_fn"], current["top_reason_class"], current["reasons_json"],
        current["features_missing"], to_state.value, reason, actor, now,
    ]
    placeholders = ", ".join("?" for _ in values)
    with connection:
        connection.execute(
            f"INSERT INTO decision_log("
            "decision_id, order_id, account_id, model_version, policy_version, "
            "cost_constants_id, effectiveness_prior_id, gate_set_version, "
            "template_version, calibrated_p, tier, threshold_used, c_fp_impression, "
            "c_fp_triggered, c_fn, top_reason_class, reasons_json, features_missing, "
            f"state, state_reason, actor, occurred_at) VALUES ({placeholders})",
            values,
        )
    return to_state
