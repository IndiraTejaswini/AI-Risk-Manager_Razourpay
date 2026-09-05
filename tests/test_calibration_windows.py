"""
Calibration window candidates.

ARCHITECTURE.md 6 fits a 2-parameter Platt map on the temporal validation window and
selects the window length from {30, 60, 90} days by Brier score.  Before any of that can
run, each candidate has to be checked for whether it can support the fit at all.

A candidate is admissible only if its positive count can stably fit two parameters.  The
rule used is events-per-parameter: Peduzzi's 10-EPP floor for logistic regression, so 20
positives for Platt's two.  Anything below that is dropped and the exclusion is stated,
rather than fitted badly and reported as a number.
"""

from __future__ import annotations

from features.calibration import (
    MIN_EVENTS_PER_PARAM,
    PLATT_N_PARAMS,
    calibration_window_candidates,
)

CANDIDATE_DAYS = (30, 60, 90)


def test_all_candidate_lengths_are_reported(loader):
    cands = calibration_window_candidates(loader)
    assert [c.days for c in cands] == list(CANDIDATE_DAYS)


def test_candidates_are_nested_and_monotone(loader):
    """A longer window is a superset, so orders and positives may only increase."""
    cands = calibration_window_candidates(loader)
    for a, b in zip(cands, cands[1:]):
        assert b.n_orders >= a.n_orders
        assert b.n_positives >= a.n_positives


def test_candidates_lie_entirely_inside_validation(loader):
    """
    The calibration map is fit on validation.  A candidate window that reached past the
    boundary would be fitting on test.
    """
    cands = calibration_window_candidates(loader)
    for c in cands:
        assert c.end <= loader.split_boundary
        assert c.start >= loader.split_boundary - __import__(
            "pandas"
        ).Timedelta(days=max(CANDIDATE_DAYS))


def test_admissibility_rule_is_events_per_parameter(loader):
    cands = calibration_window_candidates(loader)
    floor = MIN_EVENTS_PER_PARAM * PLATT_N_PARAMS
    for c in cands:
        assert c.admissible == (c.n_positives >= floor)
        assert c.events_per_param == c.n_positives / PLATT_N_PARAMS


def test_every_excluded_candidate_states_a_reason(loader):
    for c in calibration_window_candidates(loader):
        if not c.admissible:
            assert c.exclusion_reason, f"{c.days}d excluded without a stated reason"
        else:
            assert c.exclusion_reason is None


def test_prevalence_gap_against_test_is_reported(loader):
    """
    The candidates drift in prevalence, and a calibration map fit at the wrong base rate
    is miscalibrated by construction.  The gap has to be surfaced, not just the counts.
    """
    for c in calibration_window_candidates(loader):
        assert c.prevalence > 0
        assert c.test_prevalence > 0
        assert c.prevalence_ratio == c.prevalence / c.test_prevalence


def test_at_least_one_candidate_survives(loader):
    """If none did, calibration could not be fit at all and 6 would need rewriting."""
    assert any(c.admissible for c in calibration_window_candidates(loader))
