"""Semi-synthetic assignment and oracle invariants."""

import numpy as np

from simulation.generate import (
    DEFAULT_XI,
    TARGET_TREATED_FRACTION,
    generate,
    held_out_index,
)


def test_treated_fraction_constant_across_xi():
    worlds = generate()
    expected = [world.expected_treated_fraction for world in worlds]
    assert max(expected) - min(expected) < 1e-9
    assert all(abs(value - TARGET_TREATED_FRACTION) < 1e-9 for value in expected)


def test_xi_zero_is_random_assignment():
    world = generate((0.0,))[0]
    assert abs(np.corrcoef(world.treatment.astype(float), world.score)[0, 1]) < 0.08


def test_monotone_response():
    for world in generate():
        assert np.all(world.y1 <= world.y0)


def test_ground_truth_recoverable_from_oracle():
    world = generate((0.0,))[0]
    assert world.oracle_prevented_failures() == int((world.y0 - world.y1).sum())
    assert world.oracle_policy_value() == float(world.y1.sum())


def test_score_not_used_as_selection_signal():
    world = generate((1.0,))[0]
    assert "model" not in held_out_index.__code__.co_names
    assert "score" in world.__dataclass_fields__
