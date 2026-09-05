"""Ordered responder gate registry and short-circuiting chain."""

from __future__ import annotations

from .gate_00_kill_switch import gate
from .gate_01_tier_actionable import gate as tier_is_actionable
from .gate_02_already_terminal import gate as already_terminal
from .gate_03_message_class import gate as message_class_matches_tier
from .gate_04_dnd_scrub import gate as dnd_scrub
from .gate_05_consent import gate as consent_on_record
from .gate_06_opt_out import gate as opt_out_and_recontact_spacing
from .gate_07_send_window import gate as send_window
from .gate_08_daily_budget import gate as merchant_daily_budget
from .gate_09_fatigue import gate as customer_fatigue
from .gate_10_exploration import gate as exploration_slice
from .gate_11_effectiveness import gate as effectiveness_below_impression_cost
from .gate_12_no_disclosure import gate as no_risk_disclosure
from .gate_13_no_block import gate as assert_no_block_tier
from .types import Candidate, Context, Gate, GateResult

GATE_REGISTRY: tuple[tuple[str, Gate], ...] = (
    ("kill_switch", gate),
    ("tier_is_actionable", tier_is_actionable),
    ("already_terminal", already_terminal),
    ("message_class_matches_tier", message_class_matches_tier),
    ("dnd_scrub", dnd_scrub),
    ("consent_on_record", consent_on_record),
    ("opt_out_and_recontact_spacing", opt_out_and_recontact_spacing),
    ("send_window", send_window),
    ("merchant_daily_budget", merchant_daily_budget),
    ("customer_fatigue", customer_fatigue),
    ("exploration_slice", exploration_slice),
    ("effectiveness_below_impression_cost", effectiveness_below_impression_cost),
    ("no_risk_disclosure", no_risk_disclosure),
    ("assert_no_block_tier", assert_no_block_tier),
)
GATE_NAMES = tuple(name for name, _ in GATE_REGISTRY)


def run(candidate: Candidate, context: Context) -> tuple[GateResult, ...]:
    results: list[GateResult] = []
    for _, gate_fn in GATE_REGISTRY:
        result = gate_fn(candidate, context)
        results.append(result)
        if not result.passed:
            break
    return tuple(results)
