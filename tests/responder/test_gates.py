"""Gate-chain invariants and regulatory/economic guardrails."""

from __future__ import annotations

import hashlib

import pytest

from responder.gates.registry import GATE_NAMES, GATE_REGISTRY, run
from responder.gates.types import Candidate, Context
from responder.gates.gate_10_exploration import gate as exploration
from responder.gates.gate_11_effectiveness import gate as effectiveness
from responder.gates.gate_12_no_disclosure import gate as disclosure

EXPECTED_ORDER = (
    "kill_switch", "tier_is_actionable", "already_terminal",
    "message_class_matches_tier", "dnd_scrub", "consent_on_record",
    "opt_out_and_recontact_spacing", "send_window", "merchant_daily_budget",
    "customer_fatigue", "exploration_slice",
    "effectiveness_below_impression_cost", "no_risk_disclosure",
    "assert_no_block_tier",
)


def candidate(**updates) -> Candidate:
    values = {"decision_id": "decision-1", "tier": "confirm", "calibrated_p": 0.2}
    values.update(updates)
    return Candidate(**values)


@pytest.mark.parametrize("index", range(14))
def test_each_gate_has_pass_and_block(index):
    name, gate = GATE_REGISTRY[index]
    passing = candidate(message_class="service")
    pass_context = Context()
    failing = passing
    fail_context = Context()
    if index == 0:
        fail_context = Context(kill_switch=True)
    elif index == 1:
        passing = candidate(tier="confirm")
        failing = candidate(tier="allow")
    elif index == 2:
        fail_context = Context(terminal=True)
    elif index == 3:
        failing = candidate(message_class="promotional")
    elif index == 4:
        passing = candidate(message_class="promotional")
        pass_context = Context(dnd_scrubbed=True)
        failing = passing
    elif index == 5:
        passing = candidate(message_class="promotional")
        pass_context = Context(dnd_scrubbed=True, consent_timestamp="now")
        failing = passing
    elif index == 6:
        passing = candidate(message_class="promotional")
        pass_context = Context(dnd_scrubbed=True, consent_timestamp="now", opt_out_keyword=True)
        failing = passing
    elif index == 7:
        passing = candidate(message_class="promotional")
        pass_context = Context(dnd_scrubbed=True, consent_timestamp="now", opt_out_keyword=True, send_window_open=True)
        failing = passing
        fail_context = Context(send_window_open=False)
    elif index == 8:
        fail_context = Context(merchant_daily_budget_available=False)
    elif index == 9:
        fail_context = Context(customer_fatigued=True)
    elif index == 10:
        passing = candidate(decision_id="not-in-slice")
        failing = candidate(decision_id=next(
            f"slice-{i}" for i in range(10000)
            if int(hashlib.sha256(f"slice-{i}".encode()).hexdigest(), 16) % 10000 < 200
        ))
    elif index == 11:
        passing = candidate(c_rto=100, effectiveness=0.5, impression_cost=1)
        failing = candidate(c_rto=1, effectiveness=0.1, impression_cost=1)
    elif index == 12:
        passing = candidate(rendered_text="Please verify your delivery address.")
        failing = candidate(rendered_text="Your order was flagged due to address.")
        fail_context = Context(reason_vocabulary=("address",))
    elif index == 13:
        passing = candidate(tier="confirm")
        failing = candidate(tier="block")
    assert gate(passing, pass_context).passed
    assert not gate(failing, fail_context).passed


def test_gate_order_stable():
    assert GATE_NAMES == EXPECTED_ORDER


def test_first_block_short_circuits():
    results = run(candidate(), Context(kill_switch=True))
    assert len(results) == 1
    assert results[0].gate == "kill_switch"


def test_exploration_slice_deterministic():
    value = candidate(decision_id="stable")
    assert exploration(value, Context()) == exploration(value, Context())


def test_gate_11_suppresses_never_downgrades():
    result = effectiveness(candidate(tier="prepaid_only", c_rto=10, effectiveness=0.1,
                                     impression_cost=2), Context())
    assert not result.passed
    assert result.action == "SUPPRESSED"


def test_promotional_gates_skip_service_messages():
    service = candidate(tier="confirm", message_class="service")
    results = run(service, Context())
    assert all(result.passed for result in results[3:7])


def test_no_reason_vocabulary_in_rendered_text():
    context = Context(reason_vocabulary=("elevated failure rate", "incomplete address"))
    result = disclosure(candidate(rendered_text="Please verify your delivery address."), context)
    assert result.passed
    for token in context.reason_vocabulary:
        assert token not in "Please confirm your delivery address.".lower()


@pytest.mark.parametrize("text", ["risk 0.2", "prepaid_only", "high risk due to pincode"])
def test_no_score_or_tier_in_rendered_text(text):
    assert not disclosure(candidate(rendered_text=text), Context()).passed


def test_address_permitted_in_template():
    result = disclosure(
        candidate(rendered_text="Please verify delivery to 12 Main Street.",
                  address="12 Main Street"),
        Context(),
    )
    assert result.passed
