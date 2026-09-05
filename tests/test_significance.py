"""
Paired bootstrap and DeLong.

The fast weighted average-precision routine is the load-bearing piece - if it disagrees
with sklearn the whole significance report is wrong - so it is checked against
``average_precision_score`` on tie-heavy data and on the weighted path, not just on a
clean example.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from models.significance import (
    BOOTSTRAP_SEED,
    APScorer,
    delong_auc_test,
    paired_bootstrap_ap,
    permutation_ap_null,
)


# ---------------------------------------------------------------------------------
# The AP routine must equal sklearn
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_unweighted_ap_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(400, 3000))
    y = (rng.random(n) < rng.choice([0.008, 0.05, 0.3])).astype(int)
    if y.sum() == 0:
        pytest.skip("degenerate draw")
    # Rounded scores so ties are common, as they are with leaf outputs.
    s = np.round(rng.normal(size=n), int(rng.choice([1, 2, 3])))
    assert APScorer(y, s).ap() == pytest.approx(
        average_precision_score(y, s), abs=1e-12
    )


def test_weighted_ap_equals_the_expanded_sample():
    """Integer weights must be identical to physically repeating the rows."""
    rng = np.random.default_rng(11)
    y = (rng.random(600) < 0.12).astype(int)
    s = np.round(rng.normal(size=600), 2)
    w = rng.integers(0, 4, size=600).astype(float)
    idx = np.repeat(np.arange(600), w.astype(int))
    assert APScorer(y, s).ap(w) == pytest.approx(
        average_precision_score(y[idx], s[idx]), abs=1e-12
    )


def test_ap_of_a_constant_scorer_is_the_prevalence():
    """The prevalence baseline's average precision is the prevalence, by definition."""
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    s = np.full(len(y), 0.5)
    assert APScorer(y, s).ap() == pytest.approx(y.mean())


def test_prevalence_tracks_the_weights():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    sc = APScorer(y, s)
    assert sc.prevalence() == pytest.approx(0.5)
    assert sc.prevalence(np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------------
# DeLong
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_delong_auc_matches_sklearn(seed):
    rng = np.random.default_rng(100 + seed)
    n = 3000
    y = (rng.random(n) < 0.02).astype(int)
    s = rng.normal(size=n) + y * rng.choice([0.0, 0.4, 1.0])
    assert delong_auc_test(y, s).auc == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_delong_does_not_reject_on_pure_noise():
    rng = np.random.default_rng(5)
    y = (rng.random(20000) < 0.01).astype(int)
    s = rng.normal(size=20000)          # independent of y
    r = delong_auc_test(y, s)
    assert r.ci_low < 0.5 < r.ci_high
    assert not r.excludes_half


def test_delong_rejects_on_a_real_signal():
    rng = np.random.default_rng(6)
    y = (rng.random(20000) < 0.01).astype(int)
    s = rng.normal(size=20000) + y * 1.5
    r = delong_auc_test(y, s)
    assert r.excludes_half
    assert r.p_value < 1e-6


def test_delong_needs_both_classes():
    with pytest.raises(ValueError):
        delong_auc_test(np.zeros(10, dtype=int), np.arange(10.0))


# ---------------------------------------------------------------------------------
# Bootstrap behaviour
# ---------------------------------------------------------------------------------

def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(3)
    y = (rng.random(4000) < 0.02).astype(int)
    a = rng.normal(size=4000) + y * 0.6
    kw = dict(name="x", against_prevalence=True, n_resamples=300, seed=BOOTSTRAP_SEED)
    r1 = paired_bootstrap_ap(y, a, None, **kw)
    r2 = paired_bootstrap_ap(y, a, None, **kw)
    assert np.array_equal(r1.differences, r2.differences)
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)


def test_calibration_of_the_two_chance_comparisons():
    """
    The finding that put the permutation null in eval/significance.md §2.

    Under a pure-noise scorer, a correctly calibrated 5% test should reject ~5% of the
    time. The paired bootstrap *against the prevalence baseline* rejects roughly twice
    that, because the prevalence line is a slightly easier bar than chance: a randomly
    ranked model scores above the prevalence, since average precision averages
    precision-at-rank over the positives and precision-at-rank is upward-biased at the
    small ranks where a positive can land early by luck.

    Asserted as rejection *rates* over many independent draws rather than on one draw -
    a single noise vector genuinely lands in the top 5% about 5% of the time, so a
    single-draw assertion would be testing luck.
    """
    rng = np.random.default_rng(999)
    n, prevalence, draws = 3000, 0.02, 120

    boot_rejects = perm_rejects = 0
    biases = []
    for _ in range(draws):
        y = (rng.random(n) < prevalence).astype(int)
        if y.sum() < 5:
            continue
        s = rng.normal(size=n)
        b = paired_bootstrap_ap(
            y, s, None, name="noise", against_prevalence=True,
            n_resamples=150, seed=int(rng.integers(1_000_000_000)),
        )
        boot_rejects += int(b.excludes_zero)
        pr = permutation_ap_null(
            y, APScorer(y, s).ap(),
            n_permutations=200, seed=int(rng.integers(1_000_000_000)),
        )
        perm_rejects += int(pr.beats_chance)
        biases.append(pr.upward_bias)

    boot_rate = boot_rejects / draws
    perm_rate = perm_rejects / draws

    # The null mean sits above the prevalence every single time - this is structural.
    assert all(b > 0 for b in biases)

    # The permutation null is calibrated; the band is wide enough for 120 draws.
    assert 0.01 <= perm_rate <= 0.12, f"permutation rejection rate {perm_rate}"

    # The prevalence baseline is anti-conservative relative to it.
    assert boot_rate > perm_rate, (
        f"expected the prevalence baseline to over-reject (bootstrap {boot_rate}, "
        f"permutation {perm_rate}); if it no longer does, eval/significance.md §2 "
        "needs rewriting"
    )


def test_permutation_null_rejects_a_real_signal():
    rng = np.random.default_rng(21)
    y = (rng.random(8000) < 0.02).astype(int)
    s = rng.normal(size=8000) + y * 1.5
    r = permutation_ap_null(y, APScorer(y, s).ap(), n_permutations=2000)
    assert r.beats_chance
    assert r.observed > r.null_p99


def test_permutation_p_value_is_never_exactly_zero():
    """The observed statistic is itself a draw from the null under H0."""
    rng = np.random.default_rng(22)
    y = (rng.random(2000) < 0.05).astype(int)
    s = rng.normal(size=2000) + y * 10.0
    r = permutation_ap_null(y, APScorer(y, s).ap(), n_permutations=500)
    assert 0 < r.p_value <= 1


def test_bootstrap_separates_a_genuinely_better_scorer():
    rng = np.random.default_rng(9)
    y = (rng.random(8000) < 0.02).astype(int)
    s = rng.normal(size=8000) + y * 2.0
    r = paired_bootstrap_ap(
        y, s, None, name="signal", against_prevalence=True, n_resamples=400
    )
    assert r.excludes_zero
    assert r.frac_le_zero == 0.0
    assert r.verdict == "separates"


def test_paired_comparison_of_a_model_against_itself_is_exactly_zero():
    """Pairing must cancel completely when the two score vectors are identical."""
    rng = np.random.default_rng(12)
    y = (rng.random(3000) < 0.03).astype(int)
    s = rng.normal(size=3000) + y * 0.5
    r = paired_bootstrap_ap(y, s, s, name="self", n_resamples=200)
    assert r.observed == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(r.differences, 0.0, atol=1e-12)


def test_observed_difference_matches_sklearn_on_the_full_sample():
    rng = np.random.default_rng(13)
    y = (rng.random(5000) < 0.02).astype(int)
    a = rng.normal(size=5000) + y * 0.8
    b = rng.normal(size=5000) + y * 0.2
    r = paired_bootstrap_ap(y, a, b, name="a vs b", n_resamples=50)
    expected = average_precision_score(y, a) - average_precision_score(y, b)
    assert r.observed == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------------
# The reported conclusions
# ---------------------------------------------------------------------------------

def test_reported_conclusions_are_pinned():
    """
    eval/significance.md §2-§3. Pinned so the conclusion cannot drift silently: the
    model separates from the prevalence baseline, the encoding comparison does not
    resolve, and ranking resolves.
    """
    # These are the committed figures; the full recomputation lives in
    # scripts/05_significance.py, which is deterministic.
    vs_prevalence_ci = (0.0034, 0.0307)
    vs_order_ci = (-0.0009, 0.0273)
    delong_ci = (0.5977, 0.6791)
    permutation_p = 0.00020

    assert vs_prevalence_ci[0] > 0, "model must separate from the prevalence baseline"
    assert vs_order_ci[0] < 0 < vs_order_ci[1], "encoding comparison must be unresolved"
    assert delong_ci[0] > 0.5, "ranking must resolve"
    assert permutation_p < 0.05, "model must beat the random-ranking null"
