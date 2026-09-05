"""
Maturation filter, temporal split, and the group-leakage measurement.

Counts asserted here are committed in eval/positive_counts.md.
"""

from __future__ import annotations

import pandas as pd

from data.loader import CHECKOUT_SAFE_COLUMNS, OlistLoader

RAW_ORDERS = 99_441
MATURE_ORDERS = 99_433
DROPPED_IMMATURE = 8
TEST_ROWS = 19_887
TRAIN_ROWS = 79_546
SNAPSHOT = pd.Timestamp("2018-10-17 17:30:18")
SPLIT_BOUNDARY = pd.Timestamp("2018-05-24 16:58:49")

# Measured, not aspirational.  ARCHITECTURE.md 3.7 asserts this should be zero; on this
# panel it is not.  See data/COLUMN_WHITELIST.md.
STRADDLING_CUSTOMERS_ALL = 472
STRADDLING_CUSTOMERS_RISK_SET = 461


# ---------------------------------------------------------------------------------
# Maturation
# ---------------------------------------------------------------------------------

def test_snapshot_is_derived_from_observed_events(loader: OlistLoader):
    assert loader.snapshot() == SNAPSHOT


def test_snapshot_ignores_the_forward_looking_estimate(loader: OlistLoader):
    """
    order_estimated_delivery_date runs past the snapshot; if it were included the
    maturation filter would silently weaken.
    """
    raw = loader.raw_orders()
    estimated = pd.to_datetime(raw["order_estimated_delivery_date"])
    assert estimated.max() > loader.snapshot()


def test_maturation_drops_the_expected_rows(loader: OlistLoader, labelled):
    assert len(loader.raw_orders()) == RAW_ORDERS
    assert len(labelled) == MATURE_ORDERS
    assert RAW_ORDERS - MATURE_ORDERS == DROPPED_IMMATURE


def test_every_matured_order_had_time_to_resolve(loader: OlistLoader, labelled):
    latest_admissible = loader.snapshot() - pd.Timedelta(days=loader.maturation_days)
    assert bool((labelled["order_purchase_timestamp"] <= latest_admissible).all())


def test_maturation_does_not_touch_the_risk_set(loader: OlistLoader):
    """
    All 8 dropped orders are `canceled` and none shipped, so conditioning on shipment
    and filtering for maturity do not interact.
    """
    raw = loader.raw_orders()
    dropped = raw[~raw["order_id"].isin(set(loader.labelled()["order_id"]))]
    assert len(dropped) == DROPPED_IMMATURE
    assert int(dropped["order_delivered_carrier_date"].notna().sum()) == 0


# ---------------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------------

def test_split_boundary_is_an_observed_timestamp(loader: OlistLoader, labelled):
    assert loader.split_boundary == SPLIT_BOUNDARY
    assert bool((labelled["order_purchase_timestamp"] == SPLIT_BOUNDARY).any())


def test_split_sizes(labelled):
    assert int(labelled["is_test"].sum()) == TEST_ROWS
    assert int((~labelled["is_test"]).sum()) == TRAIN_ROWS


def test_split_is_chronological(labelled):
    """No train row may be purchased after any test row."""
    train_max = labelled.loc[~labelled["is_test"], "order_purchase_timestamp"].max()
    test_min = labelled.loc[labelled["is_test"], "order_purchase_timestamp"].min()
    assert train_max < test_min


def test_same_boundary_applies_to_the_risk_set(loader: OlistLoader, risk_set):
    train_max = risk_set.loc[~risk_set["is_test"], "order_purchase_timestamp"].max()
    test_min = risk_set.loc[risk_set["is_test"], "order_purchase_timestamp"].min()
    assert train_max < test_min
    assert test_min >= loader.split_boundary


# ---------------------------------------------------------------------------------
# Group leakage
# ---------------------------------------------------------------------------------

def _straddle_count(loader: OlistLoader, frame) -> int:
    customers = pd.read_csv(
        loader.data_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_unique_id"],
    )
    merged = frame.merge(customers, on="customer_id", how="left", validate="m:1")
    assert int(merged["customer_unique_id"].isna().sum()) == 0
    spans = merged.groupby("customer_unique_id")["is_test"].agg(["min", "max"])
    return int((spans["min"] != spans["max"]).sum())


def test_customer_straddle_count_is_the_documented_value(loader: OlistLoader, labelled):
    """
    ARCHITECTURE.md 3.7 asks us to assert that no customer_unique_id straddles train and
    test.  That assertion is false on this panel, so the measured count is pinned
    instead - it cannot drift unnoticed, and the discrepancy is documented in
    data/COLUMN_WHITELIST.md rather than hidden behind a test that was quietly deleted.

    This is not feature leakage under point-in-time construction (ARCHITECTURE.md 4.3):
    a repeat customer's test-period order sees only their own strictly-prior history.
    It would be leakage under k-fold out-of-fold target encoding, which 4.3 rejects.
    """
    assert _straddle_count(loader, labelled) == STRADDLING_CUSTOMERS_ALL


def test_customer_straddle_count_within_the_risk_set(loader: OlistLoader, risk_set):
    assert _straddle_count(loader, risk_set) == STRADDLING_CUSTOMERS_RISK_SET


# ---------------------------------------------------------------------------------
# Checkout frame shape
# ---------------------------------------------------------------------------------

def test_checkout_frame_is_row_aligned_with_its_population(loader: OlistLoader, risk_set):
    assert len(loader.checkout_frame(risk_set)) == len(risk_set)


def test_checkout_frame_columns_are_all_declared(loader: OlistLoader):
    declared = set(CHECKOUT_SAFE_COLUMNS["orders"]) | set(
        CHECKOUT_SAFE_COLUMNS["customers"]
    )
    assert set(loader.checkout_frame().columns) <= declared


def test_checkout_frame_does_not_silently_duplicate_rows(loader: OlistLoader, risk_set):
    frame = loader.checkout_frame(risk_set)
    assert int(frame["order_id"].duplicated().sum()) == 0
