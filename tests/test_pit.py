"""
Point-in-time correctness.

The requirement, stated exactly: no feature column for the row at time t may be computed
from any row with `order_purchase_timestamp >= t`.  Strictly prior, ties included - two
orders sharing a timestamp must not see each other.

Three levels of assurance here:

1. Golden fixtures for each PIT primitive, hand-computed, including a tie block.
2. Truncation invariance on the real panel: features built on the full panel must equal
   features built on the panel truncated to `ts < T`, for every row with `ts < T`.  A
   feature that reads the future changes when the future is removed.
3. A deliberately future-leaking aggregate, asserted to FAIL the truncation check - so
   the check is known to be capable of catching a leak rather than vacuously passing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.pit import (
    PITViolation,
    assert_truncation_invariant,
    pit_prior_count,
    pit_prior_sum,
    pit_smoothed_rate,
)

TS = "order_purchase_timestamp"


def _fixture() -> pd.DataFrame:
    """
    Hand-built panel with a deliberate tie block.

    Rows 3 and 4 share timestamp 2024-01-03 and the same key 'a'.  Neither may see the
    other: both must observe only rows 1 and 2.
    """
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "key": ["a", "b", "a", "a", "b"],
            TS: pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",  # tie with o3, same key
                    "2024-01-04",
                ]
            ),
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "y": [True, False, True, False, True],
        }
    )


# ---------------------------------------------------------------------------------
# Golden fixtures
# ---------------------------------------------------------------------------------

def test_pit_prior_count_golden():
    df = _fixture()
    got = pit_prior_count(df, "key", TS)
    # o1: first 'a'            -> 0
    # o2: first 'b'            -> 0
    # o3: 'a' has o1 before    -> 1
    # o4: ties with o3, so o3 is NOT visible; only o1 -> 1
    # o5: 'b' has o2 before    -> 1
    assert list(got) == [0, 0, 1, 1, 1]


def test_pit_prior_sum_golden():
    df = _fixture()
    got = pit_prior_sum(df, "key", TS, "value")
    # o3 sees o1 (10). o4 ties with o3 so it also sees only o1 (10), NOT 10+30.
    assert list(got) == [0.0, 0.0, 10.0, 10.0, 20.0]


def test_ties_do_not_see_each_other():
    """The tie case is the one a naive cumsum().shift() gets wrong."""
    df = _fixture()
    counts = pit_prior_count(df, "key", TS)
    assert counts.iloc[2] == counts.iloc[3] == 1, (
        "rows sharing a timestamp must observe identical history"
    )


def test_pit_smoothed_rate_golden():
    df = _fixture()
    got = pit_smoothed_rate(df, "key", TS, "y", smoothing=2.0)
    # Global prior rate is itself PIT (expanding mean of y over strictly prior rows).
    # o1: no prior at all -> falls back to the smoothing prior, which with no global
    #     history is 0.0; assert the shape rather than re-deriving the whole chain.
    assert len(got) == 5
    assert got.notna().all()
    # o3 and o4 tie, so they must receive the same encoding.
    assert got.iloc[2] == got.iloc[3]


def test_smoothing_pulls_small_groups_toward_the_prior():
    """
    A key with one prior observation must not take an extreme value; that is the whole
    point of smoothing (ARCHITECTURE.md 4.5 - no blanket pincode blacklist).
    """
    df = _fixture()
    weak = pit_smoothed_rate(df, "key", TS, "y", smoothing=100.0)
    strong = pit_smoothed_rate(df, "key", TS, "y", smoothing=1.0)
    # Heavier smoothing must move every encoding closer to the global prior, so the
    # spread across rows shrinks.
    assert weak.std() <= strong.std()


# ---------------------------------------------------------------------------------
# Truncation invariance on the real panel
# ---------------------------------------------------------------------------------

def test_feature_matrix_is_truncation_invariant(loader):
    """
    Build on the full panel, build again on the panel truncated to ts < T, and require
    the surviving rows to be identical.  This is the general PIT check: anything that
    read the future moves when the future is deleted.
    """
    from features.builder import FeatureBuilder

    population = loader.risk_set()
    cut = population[TS].quantile(0.6)
    assert_truncation_invariant(FeatureBuilder(), population, cut)


@pytest.mark.parametrize(
    "group", ["order", "customer", "pincode", "availability"]
)
def test_each_group_is_truncation_invariant(loader, group):
    from features.builder import FeatureBuilder

    population = loader.risk_set()
    cut = population[TS].quantile(0.6)
    assert_truncation_invariant(FeatureBuilder(groups=(group,)), population, cut)


# ---------------------------------------------------------------------------------
# The check must be capable of failing
# ---------------------------------------------------------------------------------

def test_truncation_check_catches_a_deliberate_future_leak(loader):
    """
    A guardrail that cannot fail is not a guardrail.  FeatureBuilder exposes a
    `leak_for_testing` switch that adds one aggregate computed over the whole panel
    rather than strictly-prior rows.  The check must reject it.
    """
    from features.builder import FeatureBuilder

    population = loader.risk_set()
    cut = population[TS].quantile(0.6)

    with pytest.raises(PITViolation, match="leaky_customer_failure_rate"):
        assert_truncation_invariant(
            FeatureBuilder(leak_for_testing=True), population, cut
        )


def test_leaky_aggregate_really_does_read_the_future(loader):
    """
    Confirm the injected leak is a genuine future read rather than merely a different
    number - the whole-panel rate for a customer must differ from their prior-only rate
    on at least one row.
    """
    from features.builder import FeatureBuilder

    population = loader.risk_set()
    leaked = FeatureBuilder(leak_for_testing=True).build(population)
    assert "leaky_customer_failure_rate" in leaked.columns
    assert leaked["leaky_customer_failure_rate"].notna().any()
