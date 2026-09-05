"""
History store: the same answers as the row-scan, by a different route.

The store exists to remove ~3s of per-request feature construction. What makes that
safe rather than merely fast is that it is *not* a second definition of any feature -
only a second way of evaluating the prior window. These tests are what hold that line.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

from data.loader import OlistLoader  # noqa: E402
from features.builder import FeatureBuilder  # noqa: E402
from features.pit import (  # noqa: E402
    pit_expanding_mean,
    pit_prior_count,
    pit_prior_last,
    pit_prior_sum,
)
from features.store import HistoryStore  # noqa: E402

TS = "order_purchase_timestamp"


def _fixture() -> pd.DataFrame:
    """Hand-built panel with a tie block, matching tests/test_pit.py's."""
    return pd.DataFrame({
        "key": ["a", "b", "a", "a", "b"],
        TS: pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                            "2024-01-03", "2024-01-04"]),
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
        "y": [1.0, 0.0, 1.0, 0.0, 1.0],
    })


@pytest.fixture(scope="module")
def real(loader: OlistLoader):
    risk = loader.risk_set()
    builder = FeatureBuilder(loader=loader)
    src = builder._sources(risk)
    store = builder.history_store(src)
    return {"risk": risk, "builder": builder, "src": src, "store": store,
            "batch": builder.build(risk)}


# ---------------------------------------------------------------------------------
# Primitive equivalence
# ---------------------------------------------------------------------------------

def test_store_reproduces_prior_count_on_a_tie_fixture():
    df = _fixture()
    store = HistoryStore(df, ("key",), ("value", "y"), ts_col=TS)
    scan = pit_prior_count(df, "key", TS)
    indexed = pit_prior_count(df, "key", TS, store)
    assert list(indexed) == list(scan) == [0, 0, 1, 1, 1]


def test_store_excludes_ties_exactly_as_the_scan_does():
    """searchsorted 'left' is the strict inequality; the block collapse is the same rule."""
    df = _fixture()
    store = HistoryStore(df, ("key",), ("value", "y"), ts_col=TS)
    indexed = pit_prior_sum(df, "key", TS, "value", store)
    scan = pit_prior_sum(df, "key", TS, "value")
    assert list(indexed) == list(scan) == [0.0, 0.0, 10.0, 10.0, 20.0]


def test_store_reproduces_last_prior_timestamp():
    df = _fixture()
    store = HistoryStore(df, ("key",), ("value", "y"), ts_col=TS)
    a = pit_prior_last(df, "key", TS, TS, store)
    b = pit_prior_last(df, "key", TS, TS)
    assert all((x == y) or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b))


def test_store_reproduces_the_global_expanding_mean():
    df = _fixture()
    store = HistoryStore(df, ("key",), ("value", "y"), ts_col=TS)
    a = pit_expanding_mean(df, TS, "y", store).to_numpy()
    b = pit_expanding_mean(df, TS, "y").to_numpy()
    assert np.array_equal(a, b)


def test_an_unknown_group_has_no_prior():
    df = _fixture()
    store = HistoryStore(df, ("key",), ("value", "y"), ts_col=TS)
    query = pd.DataFrame({"key": ["zzz"], TS: [pd.Timestamp("2024-06-01")]})
    counts, sums, last = store.prior(query, "key")
    assert counts[0] == 0
    assert sums["value"][0] == 0.0
    assert pd.isna(last[0])


# ---------------------------------------------------------------------------------
# The assertion the optimisation rests on
# ---------------------------------------------------------------------------------

def test_store_features_are_bit_identical_to_the_batch_matrix(real):
    """
    The check that stops the optimisation silently changing a feature value.

    Not approximate: a difference in the last bits would still be a different number
    fed to a model whose splits are exact thresholds.

    **Every row, not a stride.**  This test walked a 75-order stride and passed against
    two genuinely defective implementations, because neither defect's rows intersected
    the stride (tests/test_pit_exactness.py has both).  A fixed stride and a
    data-determined defect are independent patterns, so 0.08% coverage finds a 1% defect
    essentially never - and says PASS when it does not.  The full comparison is ~12s.
    """
    risk, batch = real["risk"], real["batch"]
    via_store = real["builder"].build(risk, store=real["store"])

    for col in batch.columns:
        a, b = via_store[col].to_numpy(), batch[col].to_numpy()
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            bad = ~((a.astype(float) == b.astype(float)) | (pd.isna(a) & pd.isna(b)))
        else:
            bad = np.array([
                not ((x == y) or (pd.isna(x) and pd.isna(y))) for x, y in zip(a, b)
            ])
        assert not bad.any(), (
            f"{col}: {int(bad.sum()):,} of {len(a):,} rows disagree; first at "
            f"{int(np.flatnonzero(bad)[0])}"
        )


def test_store_is_strictly_point_in_time_on_real_data(real):
    """
    The store holds the whole history including rows after the order being scored.
    Queried by timestamp, it must not see any of them - which is why a "state as of the
    end of history" shortcut was rejected.
    """
    risk, src, store = real["risk"], real["src"], real["store"]
    ordered = src.sort_values(TS, kind="mergesort")
    mid = ordered.iloc[[len(ordered) // 2]]

    counts, _, _ = store.prior(mid, "customer_zip_code_prefix")
    zip_prefix = mid["customer_zip_code_prefix"].iloc[0]
    t = mid[TS].iloc[0]
    truth = int(
        ((src["customer_zip_code_prefix"] == zip_prefix) & (src[TS] < t)).sum()
    )
    assert counts[0] == truth


def test_store_answers_do_not_depend_on_query_order(real):
    src, store = real["src"], real["store"]
    q = src.iloc[[10, 5000, 20000, 90000]]
    a, _, _ = store.prior(q, "customer_unique_id")
    b, _, _ = store.prior(q.iloc[::-1], "customer_unique_id")
    assert list(a) == list(b)[::-1]


def test_store_build_is_deterministic(real):
    src = real["src"]
    s1 = HistoryStore(src, ("customer_unique_id",), ("label_a",))
    s2 = HistoryStore(src, ("customer_unique_id",), ("label_a",))
    q = src.iloc[[1, 100, 10000]]
    a1, v1, _ = s1.prior(q, "customer_unique_id")
    a2, v2, _ = s2.prior(q, "customer_unique_id")
    assert list(a1) == list(a2)
    assert np.array_equal(v1["label_a"], v2["label_a"])
