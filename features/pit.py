"""
Point-in-time primitives.

The rule these enforce: for the row at time *t*, every history-derived value is computed
from rows with ``order_purchase_timestamp < t``.  Strictly less than - two orders sharing
a timestamp must not see each other.

Ties are the case a naive ``cumsum().shift(1)`` gets wrong.  Shifting by one row makes
the second order in a tie block observe the first, which is a same-instant read, not a
prior one.  Every primitive here resolves the value as of the *start of the tie block*
instead, via ``transform("first")`` over ``(group, timestamp)`` on a stably sorted frame.

ARCHITECTURE.md 4.3: PIT construction supersedes the embargo gap, and it is what makes
the 472 train/test straddling customers harmless (3.7).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

import pandas as pd

__all__ = [
    "PITViolation",
    "pit_prior_count",
    "pit_prior_count_asof",
    "pit_prior_sum",
    "pit_prior_first",
    "pit_expanding_mean",
    "pit_smoothed_rate",
    "assert_truncation_invariant",
]


class PITViolation(AssertionError):
    """Raised when a feature's value depends on data at or after its own timestamp."""


class _Builder(Protocol):
    def build(self, population: pd.DataFrame) -> pd.DataFrame: ...


def _block_first(values: pd.Series, keys: pd.DataFrame) -> pd.Series:
    """
    Collapse a per-row running value to the value as of the start of its tie block.

    ``values`` must already be exclusive of the row itself and computed on a frame
    sorted ascending by timestamp.

    Implemented by gathering the value at the first *position* of each block, not by
    ``transform("first")``.

    ``GroupBy.first`` skips nulls.  For the count and sum primitives that is invisible -
    their collapsed values are never null - but for :func:`pit_prior_last` the value at a
    block start is NaT exactly when nothing precedes it, and pandas skipped it and
    returned the *tied* row's own timestamp instead.  That made
    ``cust_days_since_prior_order`` report 0.0 days for 534 of 97,658 orders whose true
    answer is "no prior order": a feature reading data at its own timestamp.

    Truncation-invariance cannot catch that class of error - two rows sharing a timestamp
    are deleted together, so a same-instant read leaves no trace when the future is
    removed.  tests/test_pit_exactness.py pins it directly instead.
    """
    positions = keys.copy()
    positions["_pos"] = np.arange(len(positions))
    first_pos = positions.groupby(
        list(keys.columns), sort=False, dropna=False
    )["_pos"].transform("min")
    gathered = values.to_numpy()[first_pos.to_numpy()]
    return pd.Series(gathered, index=values.index)


def pit_prior_count(
    df: pd.DataFrame, group_col: str, ts_col: str, store=None
) -> pd.Series:
    """
    Number of strictly-prior rows sharing ``group_col``.

    With ``store``, the same window is read from a prior-index instead of scanned.
    See features/store.py: one definition, two evaluation strategies, asserted equal.
    """
    if store is not None:
        counts, _, _ = store.prior(df, group_col)
        return pd.Series(counts, index=df.index, dtype="int64")
    s = df.sort_values(ts_col, kind="mergesort")
    exclusive = s.groupby(group_col, sort=False, dropna=False).cumcount()
    out = _block_first(exclusive, s[[group_col, ts_col]])
    return out.reindex(df.index).astype("int64")


def pit_prior_count_asof(
    df: pd.DataFrame,
    group_col: str,
    ts_col: str,
    offset: pd.Timedelta,
    store=None,
) -> pd.Series:
    """
    Number of prior rows sharing ``group_col`` with ``ts < t - offset``.

    Subtracting this from :func:`pit_prior_count` gives a rolling window count over
    ``[t - offset, t)`` - half-open at both ends for the same reason the plain prior
    count is: a row at exactly *t* is not prior to itself, and a row at exactly
    ``t - offset`` is inside the window.

    The window is expressed as the difference of two prior-counts rather than as its own
    aggregation, so it inherits the tie rule from the primitive instead of restating it.

    Batch evaluation does not sort-and-scan per group.  It concatenates the history rows
    with the query cut-times and sorts once, placing the query row **before** any history
    row sharing its timestamp - which is what makes the count strict.  The store path
    binary-searches with ``side="left"``, which is the same inequality.
    """
    if store is not None:
        q = df[[group_col, ts_col]].copy()
        q[ts_col] = q[ts_col] - offset
        counts, _, _ = store.prior(q, group_col)
        return pd.Series(counts, index=df.index, dtype="int64")

    n = len(df)
    both = pd.DataFrame({
        "_g": np.concatenate([df[group_col].to_numpy(), df[group_col].to_numpy()]),
        "_t": np.concatenate([
            df[ts_col].to_numpy(), (df[ts_col] - offset).to_numpy()
        ]),
        "_h": np.concatenate([np.ones(n, dtype="int64"), np.zeros(n, dtype="int64")]),
        "_pos": np.concatenate([np.full(n, -1), np.arange(n)]),
    })
    # ascending time; on an exact tie the query row (_h == 0) sorts first, so a history
    # row at t == cut is not yet counted.  Strictly less than, by ordering.
    both = both.sort_values(["_t", "_h"], kind="mergesort")
    both["_c"] = both.groupby("_g", sort=False, dropna=False)["_h"].cumsum()

    q = both[both["_pos"] >= 0].sort_values("_pos", kind="mergesort")
    return pd.Series(q["_c"].to_numpy(), index=df.index, dtype="int64")


def pit_prior_sum(
    df: pd.DataFrame, group_col: str, ts_col: str, value_col: str, store=None
) -> pd.Series:
    """Sum of ``value_col`` over strictly-prior rows sharing ``group_col``."""
    if store is not None:
        _, sums, _ = store.prior(df, group_col)
        return pd.Series(sums[value_col], index=df.index, dtype="float64")
    s = df.sort_values(ts_col, kind="mergesort")
    vals = s[value_col].astype("float64").fillna(0.0)
    grouped = vals.groupby(s[group_col], sort=False, dropna=False)
    # Exclusive-of-self by SHIFTING the inclusive cumsum, not by subtracting the row's
    # own value back off it.  The subtraction is a cancellation: on a group with
    # thousands of prior rows it loses the low bits of a large running total, and the
    # result is not bit-equal to the previous cumsum entry that the store's precomputed
    # index returns.  Shifting is exact, and it is what makes the two evaluation
    # strategies agree rather than nearly agree.
    exclusive = grouped.cumsum().groupby(
        s[group_col], sort=False, dropna=False
    ).shift(1).fillna(0.0)
    out = _block_first(exclusive, s[[group_col, ts_col]])
    return out.reindex(df.index).astype("float64")


def pit_prior_last(
    df: pd.DataFrame, group_col: str, ts_col: str, value_col: str, store=None
) -> pd.Series:
    """
    Value of ``value_col`` on the most recent strictly-prior row in the group.

    The store path serves ``value_col == ts_col`` only, which is the single use in
    the feature set (days since the customer's previous order); anything else falls
    back to the scan so a future caller cannot get a silently wrong answer.
    """
    if store is not None and value_col == ts_col:
        _, _, last = store.prior(df, group_col)
        return pd.Series(last, index=df.index)
    s = df.sort_values(ts_col, kind="mergesort")
    shifted = s.groupby(group_col, sort=False, dropna=False)[value_col].shift(1)
    out = _block_first(shifted, s[[group_col, ts_col]])
    return out.reindex(df.index)


def pit_prior_first(
    df: pd.DataFrame, group_col: str, ts_col: str, store=None
) -> pd.Series:
    """
    Timestamp of the **earliest** strictly-prior row in the group; NaT if there is none.

    The mirror of :func:`pit_prior_last`, and the basis for tenure: how long the group
    has existed as of *t*.  Restricted to the timestamp column because that is the only
    use, and because "the value on the earliest prior row" for an arbitrary column is a
    different question that should not share a name with this one.

    Note the NaT is meaningful and is not filled: a seller's first order has no prior
    history, and a zero would assert tenure the data does not support.
    """
    if store is not None:
        return pd.Series(store.prior_first(df, group_col), index=df.index)
    s = df.sort_values(ts_col, kind="mergesort")
    earliest = s.groupby(group_col, sort=False, dropna=False)[ts_col].transform("first")
    n_prior = pit_prior_count(df, group_col, ts_col)
    return earliest.reindex(df.index).where(n_prior > 0)


def pit_expanding_mean(
    df: pd.DataFrame, ts_col: str, value_col: str, store=None
) -> pd.Series:
    """
    Expanding mean of ``value_col`` over all strictly-prior rows, ungrouped.

    This is the global prior the target encodings smooth toward.  It is itself PIT: a
    row never contributes to its own prior, and tie blocks share one value.
    """
    if store is not None:
        counts, sums = store.prior_global(df)
        n = counts.astype("float64")
        mean = np.divide(sums[value_col], n, out=np.zeros_like(n), where=n > 0)
        return pd.Series(mean, index=df.index, dtype="float64")
    s = df.sort_values(ts_col, kind="mergesort")
    vals = s[value_col].astype("float64").fillna(0.0)
    n_prior = pd.Series(range(len(s)), index=s.index, dtype="float64")
    sum_prior = vals.cumsum().shift(1).fillna(0.0)   # exact, not cancelled; see above
    keys = s[[ts_col]]
    n_prior = _block_first(n_prior, keys)
    sum_prior = _block_first(sum_prior, keys)
    mean = (sum_prior / n_prior.where(n_prior > 0)).fillna(0.0)
    return mean.reindex(df.index).astype("float64")


def pit_smoothed_rate(
    df: pd.DataFrame,
    group_col: str,
    ts_col: str,
    target_col: str,
    smoothing: float,
    store=None,
    global_prior: pd.Series | None = None,
) -> pd.Series:
    """
    Point-in-time target encoding, smoothed toward the running global mean.

        (prior_positives + m * global_prior) / (prior_count + m)

    Smoothing is what stops the encoding degenerating into a blanket blacklist: a group
    with a handful of prior observations cannot take an extreme value (ARCHITECTURE.md
    4.5).  ``m`` is the number of pseudo-observations at the global rate.

    ``global_prior`` overrides the prior the encoding smooths toward.  It exists for one
    case: the seller encoding is computed on an **order-seller edge frame**, where the
    expanding mean over rows would count a two-seller order twice and so would not be the
    global *order* failure rate.  The caller passes the order-level prior instead.  The
    store's global index is built from orders, so without this the batch and serving
    paths would smooth toward different numbers - a parity break the store self-check
    would catch, but only after the fact.
    """
    prior_n = pit_prior_count(df, group_col, ts_col, store).astype("float64")
    prior_pos = pit_prior_sum(df, group_col, ts_col, target_col, store)
    if global_prior is None:
        global_prior = pit_expanding_mean(df, ts_col, target_col, store)
    return (prior_pos + smoothing * global_prior) / (prior_n + smoothing)


def assert_truncation_invariant(
    builder: _Builder,
    population: pd.DataFrame,
    cut: pd.Timestamp,
    ts_col: str = "order_purchase_timestamp",
) -> None:
    """
    Assert every feature is point-in-time, by deleting the future and re-checking.

    Build on the full population, build again on the population truncated to
    ``ts < cut``, and require the surviving rows to be identical.  A feature that read
    data at or after its own timestamp changes when that data is removed; a PIT feature
    cannot.

    This is a general check - it needs no knowledge of how any individual feature is
    computed, so it stays valid as feature groups are added.
    """
    full = builder.build(population)
    truncated_pop = population[population[ts_col] < cut]
    if truncated_pop.empty:
        raise ValueError("truncation cut removes every row; choose a later cut")
    truncated = builder.build(truncated_pop)

    before = full.loc[truncated.index]

    offenders: list[str] = []
    for col in truncated.columns:
        a, b = before[col], truncated[col]
        if a.dtype.kind == "f" and b.dtype.kind == "f":
            same = ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
        else:
            same = (a == b) | (a.isna() & b.isna())
        if not bool(same.all()):
            offenders.append(col)

    if offenders:
        raise PITViolation(
            "feature(s) changed when data at or after the cut was removed, so they are "
            "not point-in-time: " + ", ".join(sorted(offenders))
        )
