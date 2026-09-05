"""
Golden fixtures, one per extractor.

Each group is exercised on a hand-built source frame with values chosen so the expected
output can be written down by hand rather than recorded from a run.  These double as
documentation of the edge cases: a tie block, a repeat customer, a shared pincode, a
zero-value order, and a row with missing source joins.

Testing the group methods directly rather than through `build()` keeps these fixtures
independent of the real CSVs, so a failure points at the extractor and not at the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.builder import (
    CUSTOMER_SMOOTHING,
    HISTORY_TARGET,
    PINCODE_SMOOTHING,
    FeatureBuilder,
)

TS = "order_purchase_timestamp"


@pytest.fixture
def src() -> pd.DataFrame:
    """
    Six orders.

    c1 places o1, o3, o4 - and o3/o4 share a timestamp, so neither may see the other.
    c2 places o2 and o5.  c3 places o6, its only order.
    Zip 11111 covers o1, o3, o4; zip 22222 covers o2, o5, o6.
    o6 has no item or payment join, standing in for a degraded source path.
    """
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5", "o6"],
            "customer_id": ["a1", "a2", "a3", "a4", "a5", "a6"],
            "customer_unique_id": ["c1", "c2", "c1", "c1", "c2", "c3"],
            "customer_zip_code_prefix": [11111, 22222, 11111, 11111, 22222, 22222],
            TS: pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-02 11:00",
                    "2024-01-05 09:00",
                    "2024-01-05 09:00",  # tie with o3, same customer and zip
                    "2024-01-08 20:00",
                    "2024-01-09 23:00",
                ]
            ),
            HISTORY_TARGET: [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "order_value": [100.0, 50.0, 200.0, 0.0, 75.0, np.nan],
            "order_freight": [10.0, 5.0, 40.0, 0.0, 25.0, np.nan],
            "n_items": [1.0, 2.0, 4.0, 1.0, 3.0, np.nan],
            "n_sellers": [1.0, 1.0, 2.0, 1.0, 1.0, np.nan],
            "n_products": [1.0, 2.0, 3.0, 1.0, 3.0, np.nan],
            "payment_value": [110.0, 55.0, 240.0, 0.0, 100.0, np.nan],
            "payment_installments": [1.0, 3.0, 6.0, 1.0, 2.0, np.nan],
            "n_payment_types": [1.0, 1.0, 2.0, 1.0, 1.0, np.nan],
            "is_boleto": [True, False, True, False, False, False],
            "product_weight_g": [500.0, 250.0, 1000.0, 100.0, 750.0, np.nan],
            "product_length_cm": [10.0, 5.0, 20.0, 2.0, 15.0, np.nan],
            "product_height_cm": [10.0, 5.0, 20.0, 2.0, 15.0, np.nan],
            "product_width_cm": [10.0, 5.0, 20.0, 2.0, 15.0, np.nan],
            "product_photos_qty": [1.0, 2.0, 3.0, 1.0, 4.0, np.nan],
            "product_category_name": ["a", "b", "a", "c", "b", None],
        }
    )


@pytest.fixture
def builder() -> FeatureBuilder:
    return FeatureBuilder.__new__(FeatureBuilder)  # no loader needed for group methods


# ---------------------------------------------------------------------------------
# order
# ---------------------------------------------------------------------------------

def test_order_extractor_golden(builder, src):
    builder.leak_for_testing = False
    f = builder._order_features(src)

    assert list(f["order_value"]) [:5] == [100.0, 50.0, 200.0, 0.0, 75.0]
    # freight / (value + freight)
    assert f["freight_ratio"].iloc[0] == pytest.approx(10.0 / 110.0)
    assert f["freight_ratio"].iloc[2] == pytest.approx(40.0 / 240.0)
    # o4 has value 0 and freight 0 -> denominator 0, must be NaN not a divide-by-zero
    assert pd.isna(f["freight_ratio"].iloc[3])
    assert f["log_order_value"].iloc[0] == pytest.approx(np.log1p(100.0))
    assert f["avg_item_price"].iloc[2] == pytest.approx(200.0 / 4.0)
    assert f["product_volume_cm3"].iloc[0] == pytest.approx(1000.0)
    assert list(f["is_boleto"])[:3] == [1.0, 0.0, 1.0]
    assert list(f["product_category"])[:3] == ["a", "b", "a"]


def test_order_extractor_timing_golden(builder, src):
    builder.leak_for_testing = False
    f = builder._order_features(src)
    assert list(f["purchase_hour"]) == [10, 11, 9, 9, 20, 23]
    # 2024-01-01 is a Monday
    assert list(f["purchase_dow"]) == [0, 1, 4, 4, 0, 1]
    assert list(f["purchase_is_weekend"]) == [0, 0, 0, 0, 0, 0]
    assert set(f["purchase_month"]) == {1}


def test_order_extractor_carries_missing_joins_through(builder, src):
    builder.leak_for_testing = False
    f = builder._order_features(src)
    assert pd.isna(f["order_value"].iloc[5])
    assert pd.isna(f["product_volume_cm3"].iloc[5])


# ---------------------------------------------------------------------------------
# customer
# ---------------------------------------------------------------------------------

def test_customer_extractor_golden(builder, src):
    builder.leak_for_testing = False
    f = builder._customer_features(src)

    # o1 is c1's first order; o3 and o4 tie, so both see only o1.
    assert list(f["cust_prior_orders"]) == [0, 0, 1, 1, 1, 0]
    # c1's prior failures at o3/o4 = label_a of o1 = 1
    assert list(f["cust_prior_failures"]) == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    # c1's prior average value at o3/o4 = o1's value = 100
    assert f["cust_prior_avg_value"].iloc[2] == pytest.approx(100.0)
    assert f["cust_prior_avg_value"].iloc[3] == pytest.approx(100.0)
    assert pd.isna(f["cust_prior_avg_value"].iloc[0])
    # days since prior: o3/o4 on 01-05 09:00 vs o1 on 01-01 10:00
    expected_days = (
        pd.Timestamp("2024-01-05 09:00") - pd.Timestamp("2024-01-01 10:00")
    ).total_seconds() / 86400.0
    assert f["cust_days_since_prior_order"].iloc[2] == pytest.approx(expected_days)
    assert pd.isna(f["cust_days_since_prior_order"].iloc[0])
    # o1 was boleto, so c1's prior boleto ratio at o3 is 1.0
    assert f["cust_prior_boleto_ratio"].iloc[2] == pytest.approx(1.0)


def test_customer_tie_block_sees_identical_history(builder, src):
    builder.leak_for_testing = False
    f = builder._customer_features(src)
    for col in f.columns:
        a, b = f[col].iloc[2], f[col].iloc[3]
        assert (a == b) or (pd.isna(a) and pd.isna(b)), col


def test_customer_smoothed_rate_is_bounded_by_the_prior(builder, src):
    """
    c1 has one prior order and it failed.  Without smoothing the encoding would read
    1.0; with CUSTOMER_SMOOTHING pseudo-observations it must stay far below that.
    """
    builder.leak_for_testing = False
    f = builder._customer_features(src)
    assert f["cust_prior_failure_rate"].iloc[2] < 1.0 / CUSTOMER_SMOOTHING + 0.5


def test_leak_switch_adds_a_non_pit_column(builder, src):
    builder.leak_for_testing = True
    f = builder._customer_features(src)
    assert "leaky_customer_failure_rate" in f.columns
    # c1's whole-panel mean over o1, o3, o4 = (1 + 1 + 0) / 3
    assert f["leaky_customer_failure_rate"].iloc[0] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------------
# pincode
# ---------------------------------------------------------------------------------

def test_pincode_extractor_golden(builder, src):
    f = builder._pincode_features(src)

    # zip 11111: o1, then o3/o4 tied. zip 22222: o2, o5, o6.
    assert list(f["pincode_prior_orders"]) == [0, 0, 1, 1, 1, 2]
    assert list(f["pincode_prior_failures"]) == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    # Global prior at o6 = mean label_a over o1..o5 = (1+0+1+0+0)/5
    assert f["global_prior_failure_rate"].iloc[5] == pytest.approx(0.4)
    assert f["global_prior_failure_rate"].iloc[0] == pytest.approx(0.0)


def test_pincode_smoothing_dominates_a_thin_group(builder, src):
    """
    ARCHITECTURE.md 4.5 - smoothing is what stops the encoding degenerating into a
    blanket pincode blacklist.  One prior failure out of one observation must not
    produce an extreme encoding.
    """
    f = builder._pincode_features(src)
    encoded = f["pincode_failure_rate_smoothed"].iloc[2]
    prior = f["global_prior_failure_rate"].iloc[2]
    # With 1 observation against PINCODE_SMOOTHING pseudo-counts, the encoding must sit
    # within 1/(1+PINCODE_SMOOTHING) of the global prior.
    assert abs(encoded - prior) <= 1.0 / (1.0 + PINCODE_SMOOTHING)


# ---------------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------------

def test_availability_extractor_golden(builder, src):
    f = builder._availability_features(src, pd.DataFrame(index=src.index))
    assert list(f["has_item_rows"]) == [1, 1, 1, 1, 1, 0]
    assert list(f["has_payment_row"]) == [1, 1, 1, 1, 1, 0]
    assert list(f["has_product_metadata"]) == [1, 1, 1, 1, 1, 0]
    assert list(f["has_zip_prefix"]) == [1, 1, 1, 1, 1, 1]
    # No other columns supplied, so nothing can be missing.
    assert list(f["n_missing_features"]) == [0, 0, 0, 0, 0, 0]


def test_n_missing_features_counts_supplied_columns(builder, src):
    built = pd.DataFrame(
        {"x": [1.0, np.nan, 1.0, np.nan, 1.0, np.nan],
         "y": [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0]},
        index=src.index,
    )
    f = builder._availability_features(src, built)
    assert list(f["n_missing_features"]) == [1, 2, 0, 1, 0, 1]
