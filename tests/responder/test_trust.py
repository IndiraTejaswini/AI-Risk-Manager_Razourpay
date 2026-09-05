from __future__ import annotations

import pytest

from responder.trust import Tagged, Tier, UngroundedClaim, render_measured


def test_assumed_propagates():
    assumed = Tagged(2.0, Tier.ASSUMED, "effectiveness_prior")
    measured = Tagged(3.0, Tier.MEASURED, "observed")
    assert (assumed * measured).tier is Tier.ASSUMED
    assert (measured + assumed).tier is Tier.ASSUMED


def test_render_measured_rejects_assumed():
    with pytest.raises(UngroundedClaim):
        render_measured(Tagged(10.0, Tier.ASSUMED, "cost_constants"))


def test_savings_figure_is_unrenderable():
    treated = Tagged(10.0, Tier.MEASURED, "treated_set")
    failure_rate = Tagged(0.2, Tier.MEASURED, "realized_failure_rate")
    c_rto = Tagged(100.0, Tier.MEASURED, "observed")
    effectiveness = Tagged(0.5, Tier.ASSUMED, "effectiveness_prior")
    with pytest.raises(UngroundedClaim):
        render_measured(treated * failure_rate * c_rto * effectiveness)
