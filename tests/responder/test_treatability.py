"""Treatability explanations remain explicitly assumption-labelled."""

from __future__ import annotations

import pytest

from responder.explain.treatability import treatability
from responder.templates.registry import ReasonClass, Tier
from responder.trust import UngroundedClaim, render_measured
from api.main import app


def test_treatability_is_assumed_tier():
    result = treatability(ReasonClass.ADDRESS_QUALITY, Tier.CONFIRM)
    with pytest.raises(UngroundedClaim):
        render_measured(result.effectiveness)


def test_treatability_mapping_total():
    for reason in ReasonClass:
        for tier in Tier:
            result = treatability(reason, tier)
            assert result.best_tier in Tier
    with pytest.raises(ValueError, match="unmapped"):
        treatability("not-a-reason", Tier.CONFIRM)
    with pytest.raises(ValueError, match="unmapped"):
        treatability(ReasonClass.ADDRESS_QUALITY, "not-a-tier")


def test_explain_endpoint_not_in_score_path():
    routes = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/explain/{order_id}" in routes
    assert "/score" in routes
    assert not any(
        route.path == "/score" and "explain" in getattr(route, "name", "").lower()
        for route in app.routes
        if hasattr(route, "path")
    )
