"""
Regression tests for two defects that fixed-stride sampling let through.

Both were live in shipped code. Neither was found by a test - the first was caught by
the startup store self-check only once a feature group with large history groups was
added, and the second only once that check was widened from a 40-order stride to every
row. They are pinned here as unit fixtures so neither can return, and so that a reader
can see the failure mode without running the pipeline.

--------------------------------------------------------------------------------------
DEFECT 1 - the compensated-cumsum subtraction
--------------------------------------------------------------------------------------

The point-in-time "sum over strictly prior rows" was ``inclusive_cumsum - own_value``.
pandas' groupby cumsum uses Kahan compensation, so subtracting a row's own value back
off the running total does not recover the previous entry exactly. On the real panel
that moved ``cust_prior_avg_value`` on 1,169 of 97,658 rows by up to 2.3e-13 - enough to
flip LightGBM split thresholds - and it made the feature unreproducible from a
prior-window index, which can only ever return the previous cumsum entry.

--------------------------------------------------------------------------------------
DEFECT 2 - transform("first") skips NaN
--------------------------------------------------------------------------------------

Tie handling collapses a value to the start of its ``(group, timestamp)`` block, via
``transform("first")``. ``GroupBy.first`` **skips nulls by default**. For the count and
sum primitives the collapsed values are never null and the mechanism worked. For
``pit_prior_last`` the value at a block start is exactly NaT when nothing precedes it -
so pandas skipped it and returned the *tied* row's own timestamp instead.

``cust_days_since_prior_order`` therefore read a same-instant order as "the previous
order" for 534 of 97,658 rows, reporting 0.0 days where the answer is "no prior order".
That is a point-in-time violation: a feature reading data at its own timestamp.

**Truncation-invariance cannot catch this.** It deletes rows at or after a cut and
requires the survivors to be unchanged. Two orders sharing a timestamp are deleted
*together*, so a same-instant read leaves no trace when the future is removed. The
invariant tests the future; this defect lives in the present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.pit import pit_prior_last, pit_prior_sum

TS = "order_purchase_timestamp"


# ---------------------------------------------------------------------------------
# Defect 1: the exclusive prior sum must equal the previous cumsum entry exactly
# ---------------------------------------------------------------------------------

def _large_group(n: int = 4000, seed: int = 1) -> pd.DataFrame:
    """One group big enough for Kahan compensation to diverge from a subtraction."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "key": ["k"] * n,
        TS: pd.date_range("2017-01-01", periods=n, freq="h"),
        "value": rng.uniform(1.0, 500.0, n),
    })


def test_exclusive_prior_sum_is_the_previous_cumsum_entry_bit_for_bit():
    """
    The property that makes the feature reproducible from an index.

    An index stores a per-group cumulative sum and answers a query by reading entry
    j-1. If the row-scan computes anything other than exactly that value, the two
    evaluation strategies disagree and no amount of sampling makes them agree.
    """
    df = _large_group()
    got = pit_prior_sum(df, "key", TS, "value").to_numpy()

    vals = df["value"].to_numpy()
    inclusive = pd.Series(vals).groupby(np.zeros(len(vals), dtype=int)).cumsum()
    expected = np.concatenate(([0.0], inclusive.to_numpy()[:-1]))

    assert np.array_equal(got, expected), (
        "prior sum is not the previous cumulative entry; the serving index cannot "
        "reproduce it"
    )


def test_the_old_subtraction_form_would_fail_that_property():
    """
    The defect, reproduced explicitly.

    This asserts the OLD implementation is wrong rather than that the new one is right,
    so the fixture keeps documenting why the change was necessary even if the primitive
    is rewritten again.
    """
    df = _large_group()
    vals = df["value"].astype("float64")
    inclusive = vals.groupby(np.zeros(len(vals), dtype=int)).cumsum()

    old = (inclusive - vals).to_numpy()                      # cumsum - own value
    new = np.concatenate(([0.0], inclusive.to_numpy()[:-1]))  # the shifted entry

    differing = int((old != new).sum())
    assert differing > 0, (
        "the fixture no longer reproduces the compensated-cumsum discrepancy, so it is "
        "no longer testing anything - widen the group or the value range"
    )
    # Tiny, and that is the point: it was large enough to move a model.
    assert np.abs(old - new).max() < 1e-8

    # And the shipped implementation must be on the correct side of that difference.
    assert np.array_equal(pit_prior_sum(df, "key", TS, "value").to_numpy(), new)


# ---------------------------------------------------------------------------------
# Defect 2: a tie block must not report its own members as prior
# ---------------------------------------------------------------------------------

def _tied_pair() -> pd.DataFrame:
    """
    Two orders from one customer at the identical instant, and nothing before them.

    Real: customer d01cf8c6c7c836c5dd9320585928f42b placed both of their orders at
    2018-04-15 19:42:06.
    """
    return pd.DataFrame({
        "key": ["c1", "c1"],
        TS: pd.to_datetime(["2018-04-15 19:42:06", "2018-04-15 19:42:06"]),
    })


def test_tied_first_orders_have_no_prior_order():
    """
    Neither of two same-instant orders is prior to the other.

    The old code returned the *other* row's timestamp here, which made
    `cust_days_since_prior_order` read 0.0 days - a feature observing data at its own
    timestamp.
    """
    got = pit_prior_last(_tied_pair(), "key", TS, TS)
    assert got.isna().all(), (
        f"a tied first order reported a prior order at {got.tolist()}; that is a "
        "same-instant read"
    )


def test_days_since_prior_order_is_null_not_zero_for_a_tied_first_order():
    """
    The feature-level consequence, stated in the units a merchant would see.

    0.0 days means "ordered again immediately"; null means "no previous order". The
    difference is not cosmetic - the booster branches on null natively and 0.0 puts the
    row at the extreme end of a real distribution.
    """
    df = _tied_pair()
    days = (df[TS] - pit_prior_last(df, "key", TS, TS)).dt.total_seconds() / 86400.0
    assert days.isna().all()
    assert not (days == 0.0).any()


def test_a_later_order_still_sees_the_tied_block():
    """The fix must not overshoot: a genuinely later order has a prior one."""
    df = pd.DataFrame({
        "key": ["c1", "c1", "c1"],
        TS: pd.to_datetime([
            "2018-04-15 19:42:06", "2018-04-15 19:42:06", "2018-04-20 08:00:00",
        ]),
    })
    got = pit_prior_last(df, "key", TS, TS)
    assert got.isna().iloc[0] and got.isna().iloc[1]
    assert got.iloc[2] == pd.Timestamp("2018-04-15 19:42:06")


def test_tie_block_collapse_does_not_skip_nulls():
    """
    The mechanism, isolated from any feature.

    `GroupBy.first` skips nulls; the tie-block collapse must take the value at the first
    *position* of the block whether or not it is null. This is the one-line difference
    between the two implementations.
    """
    from features.pit import _block_first

    keys = pd.DataFrame({"g": ["a", "a", "b"], "t": [1, 1, 2]})
    values = pd.Series([np.nan, 5.0, 7.0])
    got = _block_first(values, keys)

    assert pd.isna(got.iloc[0]) and pd.isna(got.iloc[1]), (
        "the collapse skipped a null at the block start and promoted a tied row's value"
    )
    assert got.iloc[2] == 7.0
