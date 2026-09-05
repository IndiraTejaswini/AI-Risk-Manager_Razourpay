"""
Leakage guardrail.

The central requirement: `order_delivered_carrier_date` is used **exclusively** to build
the risk set and the primary label `label_b`, and is **100% excluded** from every
feature matrix.

The column is simultaneously required and forbidden, which is why it gets behavioural
tests rather than a membership check.  A name check proves the column is not carried
through; it cannot prove no feature was silently derived from it.  test_carrier_date_*
below prove that by perturbation: corrupt the column, rebuild, and assert the label
moves while the feature matrix does not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.loader import (
    FORBIDDEN_FEATURE_COLUMNS,
    LABEL_ONLY_COLUMNS,
    POST_CHECKOUT_COLUMNS,
    LeakageError,
    OlistLoader,
)

CARRIER = "order_delivered_carrier_date"


# ---------------------------------------------------------------------------------
# The whitelist itself
# ---------------------------------------------------------------------------------

def test_carrier_date_is_declared_label_only():
    assert CARRIER in LABEL_ONLY_COLUMNS
    assert CARRIER in FORBIDDEN_FEATURE_COLUMNS


def test_label_and_post_checkout_sets_are_disjoint():
    assert not (LABEL_ONLY_COLUMNS & POST_CHECKOUT_COLUMNS)


def test_no_checkout_safe_column_is_forbidden():
    """The two halves of the whitelist must not contradict each other."""
    from data.loader import CHECKOUT_SAFE_COLUMNS

    for table, cols in CHECKOUT_SAFE_COLUMNS.items():
        overlap = set(cols) & FORBIDDEN_FEATURE_COLUMNS
        assert not overlap, f"{table} declares forbidden column(s): {sorted(overlap)}"


# ---------------------------------------------------------------------------------
# Exclusion from feature matrices
# ---------------------------------------------------------------------------------

def test_checkout_frame_excludes_carrier_date(loader: OlistLoader):
    assert CARRIER not in loader.checkout_frame().columns


def test_checkout_frame_excludes_every_forbidden_column(loader: OlistLoader):
    cols = set(loader.checkout_frame().columns)
    assert not (cols & FORBIDDEN_FEATURE_COLUMNS)


def test_checkout_frame_on_full_population_is_also_clean(loader: OlistLoader, labelled):
    """The guard holds regardless of which population is projected."""
    cols = set(loader.checkout_frame(labelled).columns)
    assert not (cols & FORBIDDEN_FEATURE_COLUMNS)


def test_assert_no_leakage_rejects_carrier_date(loader: OlistLoader):
    frame = loader.checkout_frame().head(10)
    frame[CARRIER] = pd.Timestamp("2018-01-01")
    with pytest.raises(LeakageError, match=CARRIER):
        OlistLoader.assert_no_leakage(frame)


@pytest.mark.parametrize("column", sorted(FORBIDDEN_FEATURE_COLUMNS))
def test_assert_no_leakage_rejects_each_forbidden_column(loader: OlistLoader, column):
    frame = loader.checkout_frame().head(5)
    frame[column] = 1
    with pytest.raises(LeakageError, match=column):
        OlistLoader.assert_no_leakage(frame)


def test_assert_no_leakage_passes_a_clean_frame(loader: OlistLoader):
    OlistLoader.assert_no_leakage(loader.checkout_frame().head(10))


# ---------------------------------------------------------------------------------
# Exclusivity, proved by perturbation
# ---------------------------------------------------------------------------------

def _perturbed_loader(loader: OlistLoader) -> OlistLoader:
    """
    A loader whose raw frame has `order_delivered_carrier_date` blanked out.

    Blanking rather than shifting, because the column is consumed as a presence test
    (`.notna()`); nulling it is the perturbation the label is actually sensitive to.
    """
    other = OlistLoader(data_dir=loader.data_dir)
    raw = loader.raw_orders()
    raw[CARRIER] = pd.NaT
    other._raw = raw
    return other


def test_carrier_date_is_used_for_label_b(loader: OlistLoader, labelled):
    """If label_b did not consume the column, blanking it would change nothing."""
    perturbed = _perturbed_loader(loader).labelled()
    assert int(labelled["label_b"].sum()) > 0
    assert int(perturbed["label_b"].sum()) == 0, (
        "blanking the carrier date left label_b intact, so label_b is not actually "
        "conditioned on fulfillment"
    )


def test_carrier_date_defines_the_risk_set(loader: OlistLoader, risk_set):
    perturbed = _perturbed_loader(loader).risk_set()
    assert len(risk_set) > 0
    assert len(perturbed) == 0


def test_carrier_date_does_not_reach_the_feature_matrix(loader: OlistLoader):
    """
    The exclusivity claim.  Perturbing the column must leave the checkout frame
    untouched for the same population.  Any feature derived from it, however indirectly,
    would show up here as a difference.
    """
    population = loader.labelled()
    baseline = loader.checkout_frame(population)

    perturbed_loader = _perturbed_loader(loader)
    perturbed = perturbed_loader.checkout_frame(perturbed_loader.labelled())

    pd.testing.assert_frame_equal(baseline, perturbed)


def test_label_a_does_not_depend_on_carrier_date(loader: OlistLoader, labelled):
    """label_a is unconditional by construction; the perturbation must not move it."""
    perturbed = _perturbed_loader(loader).labelled()
    assert int(perturbed["label_a"].sum()) == int(labelled["label_a"].sum())
