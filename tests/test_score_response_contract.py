"""Compatibility checks for the public ``/score`` response schema."""

from __future__ import annotations

from typing import Literal

from api.contract import ScoreResponse


# Captured from the current response contract. New response fields are compatible;
# removing or renaming one is not.
BASELINE_FIELDS = {
    "risk",
    "tier",
    "reasons",
    "model_version",
    "threshold_used",
    "features_missing",
}

BASELINE_TYPES = {
    "risk": float,
    "tier": Literal["allow", "confirm", "prepaid_only", "defer"],
    "reasons": list[str],
    "model_version": str,
    "threshold_used": float,
    "features_missing": int,
}


def test_score_response_schema_is_additive_only():
    current_fields = ScoreResponse.model_fields

    assert BASELINE_FIELDS <= current_fields.keys(), (
        "ScoreResponse removed or renamed baseline field(s): "
        + ", ".join(sorted(BASELINE_FIELDS - current_fields.keys()))
    )


def test_score_response_baseline_types_are_not_narrowed():
    current_fields = ScoreResponse.model_fields

    for field_name, baseline_type in BASELINE_TYPES.items():
        assert current_fields[field_name].annotation == baseline_type, (
            f"ScoreResponse.{field_name} narrowed or changed from "
            f"{baseline_type!r} to {current_fields[field_name].annotation!r}"
        )
