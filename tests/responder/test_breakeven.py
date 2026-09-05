"""Break-even effectiveness remains measured and prior-free."""

import pytest

from responder.breakeven import compute_break_even
from responder.trust import Tagged, Tier, render_measured


def test_breakeven_uses_no_prior():
    result = compute_break_even(
        Tagged(100.0, Tier.MEASURED, "treated_failures"),
        Tagged(10.0, Tier.MEASURED, "observed_cost"),
        Tagged(5.0, Tier.MEASURED, "observed_cost"),
        Tagged(100.0, Tier.MEASURED, "c_rto_sweep"),
    )
    assert result.tier is Tier.MEASURED
    assert "ASSUMED" not in result.source
    assert "effectiveness_prior" not in result.source


def test_breakeven_survives_render_measured():
    result = compute_break_even(100.0, 10.0, 5.0, 100.0)
    assert render_measured(result) == "0.0015"


def test_savings_figure_still_raises():
    effectiveness = Tagged(0.5, Tier.ASSUMED, "effectiveness_prior")
    savings = Tagged(100.0, Tier.MEASURED, "treated_set") * effectiveness
    with pytest.raises(Exception):
        render_measured(savings)


def test_breakeven_is_swept():
    from responder.breakeven import sweep_break_even
    values = sweep_break_even()
    assert values["low"].value > values["base"].value > values["high"].value
