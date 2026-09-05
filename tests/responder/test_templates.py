"""Template registry completeness, determinism, and disclosure tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.explain import NO_RISK_REASON, REASON_TEMPLATES
from responder.templates.registry import (
    REASON_CLASS_BY_REASON,
    TEMPLATES,
    ReasonClass,
    Tier,
    reason_class,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rendered"
FIELDS = {"address": "12 Main Street", "order_id": "order_123"}


def test_reason_mapping_is_total():
    assert set(REASON_CLASS_BY_REASON) == set(REASON_TEMPLATES.values()) | {NO_RISK_REASON}


def test_no_unmapped_reason_defaults():
    with pytest.raises(ValueError, match="unmapped"):
        reason_class("a reason the detector never declared")


@pytest.mark.parametrize("cell", list(TEMPLATES))
def test_rendered_matches_golden(cell):
    reason, tier = cell
    actual = TEMPLATES[cell].render(FIELDS)
    expected = (FIXTURES / f"{reason.value}__{tier.value}.txt").read_text(encoding="utf-8")
    assert actual == expected


def test_message_class_tagged():
    assert {template.message_class for template in TEMPLATES.values()} == {
        "SERVICE", "PROMOTIONAL"
    }


def test_no_risk_disclosure_in_any_template():
    forbidden = {"risk", "allow", "confirm", "prepaid_only", "defer", "block"}
    forbidden.update(value.lower() for value in REASON_TEMPLATES.values())
    for template in TEMPLATES.values():
        text = template.render(FIELDS).lower()
        assert not any(token in text for token in forbidden)
        assert not any(token in template.field_names for token in forbidden)
