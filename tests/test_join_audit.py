"""
Pins the join-cardinality audit (eval/join_cardinality_audit.md) and the corrected
secondary average precision.

These are characterisation tests. They do not assert the leaks are fixed - two of them
are properties of the dataset and cannot be. They assert the leaks are still exactly the
size the reports say, so a change in the data, the join paths, or the population surfaces
as a failure rather than as a quietly different number in a markdown file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import PRIMARY_LABEL, SECONDARY_LABEL, OlistLoader
from features.builder import FeatureBuilder
from models.evaluate import evaluate
from models.train import predict, prepare_matrix, train

# --- eval/join_cardinality_audit.md §1 ------------------------------------------------
MATURED = 99_433
RISK_SET = 97_658

ITEMS_ZERO_MATURED = 767
ITEMS_ZERO_RISK = 1
REVIEWS_ZERO_MATURED = 768
REVIEWS_ZERO_RISK = 731
PAYMENTS_ZERO_MATURED = 1
CUSTOMERS_ZERO_MATURED = 0
GEO_ZERO_MATURED = 278            # customer side
GEO_SELLER_ZERO_MATURED = 983     # seller side, of which 767 are the items artifact
GEO_SELLER_ZERO_RISK = 214
GEO_SELLER_DUE_TO_ITEMS = 767

REVIEWS_RISK_RATIO = 9.61   # label_b rate within the no-review group / risk-set panel
FLAG_RATIO = 3.0

# --- eval/model_report.md §5 ----------------------------------------------------------
CORRECTED_SECONDARY_AP = 0.0190


def _ids(loader: OlistLoader, filename: str, col: str = "order_id") -> set:
    return set(pd.read_csv(loader.data_dir / filename, usecols=[col])[col])


# ---------------------------------------------------------------------------------
# Join cardinality
# ---------------------------------------------------------------------------------

def test_item_join_gap_is_unchanged(loader: OlistLoader, labelled, risk_set):
    ids = _ids(loader, "olist_order_items_dataset.csv")
    assert int((~labelled["order_id"].isin(ids)).sum()) == ITEMS_ZERO_MATURED
    assert int((~risk_set["order_id"].isin(ids)).sum()) == ITEMS_ZERO_RISK


def test_item_join_gap_is_a_perfect_predictor_on_the_matured_set(loader, labelled):
    ids = _ids(loader, "olist_order_items_dataset.csv")
    missing = ~labelled["order_id"].isin(ids)
    assert float(labelled.loc[missing, SECONDARY_LABEL].mean()) == 1.0


def test_review_join_gap_reaches_the_primary_target(loader: OlistLoader, risk_set):
    """
    The finding the audit exists for: unlike the items artifact, the review gap bites on
    the risk set. A customer who never receives an order does not review it.
    """
    ids = _ids(loader, "olist_order_reviews_dataset.csv")
    missing = ~risk_set["order_id"].isin(ids)
    assert int(missing.sum()) == REVIEWS_ZERO_RISK

    rate = float(risk_set.loc[missing, PRIMARY_LABEL].mean())
    panel = float(risk_set[PRIMARY_LABEL].mean())
    assert rate / panel == pytest.approx(REVIEWS_RISK_RATIO, abs=0.05)
    assert rate / panel > FLAG_RATIO


def test_review_gap_is_independent_of_the_item_gap(loader: OlistLoader, labelled):
    """If it were the same rows, it would not be a separate finding."""
    review_ids = _ids(loader, "olist_order_reviews_dataset.csv")
    item_ids = _ids(loader, "olist_order_items_dataset.csv")
    no_review = ~labelled["order_id"].isin(review_ids)
    no_items = ~labelled["order_id"].isin(item_ids)
    assert int(no_review.sum()) == REVIEWS_ZERO_MATURED
    overlap = int((no_review & no_items).sum())
    assert overlap < 0.05 * REVIEWS_ZERO_MATURED


def test_reviews_are_never_joined_by_the_feature_builder():
    """
    The review leak is a landmine, not a live defect. This asserts it stays that way.
    """
    import inspect

    from features import builder

    source = inspect.getsource(builder)
    assert "review" not in source.lower()


def test_payments_and_customers_have_no_usable_cardinality_signal(
    loader: OlistLoader, labelled
):
    pay = _ids(loader, "olist_order_payments_dataset.csv")
    assert int((~labelled["order_id"].isin(pay)).sum()) == PAYMENTS_ZERO_MATURED

    cust = set(
        pd.read_csv(loader.data_dir / "olist_customers_dataset.csv",
                    usecols=["customer_id"])["customer_id"]
    )
    assert int((~labelled["customer_id"].isin(cust)).sum()) == CUSTOMERS_ZERO_MATURED


def test_products_and_sellers_inherit_the_item_gap_rather_than_adding_one(
    loader: OlistLoader, labelled
):
    items = pd.read_csv(
        loader.data_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "seller_id"],
    )
    products = _ids(loader, "olist_products_dataset.csv", "product_id")
    sellers = _ids(loader, "olist_sellers_dataset.csv", "seller_id")

    with_product = set(items.loc[items["product_id"].isin(products), "order_id"])
    with_seller = set(items.loc[items["seller_id"].isin(sellers), "order_id"])
    no_items = ~labelled["order_id"].isin(set(items["order_id"]))

    for reached in (with_product, with_seller):
        missing = ~labelled["order_id"].isin(reached)
        assert int(missing.sum()) == ITEMS_ZERO_MATURED
        # Identical rows, not merely an identical count.
        assert bool((missing == no_items).all())


def test_geolocation_gap_stays_below_the_flag_threshold(loader: OlistLoader, labelled):
    customers = pd.read_csv(
        loader.data_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_zip_code_prefix"],
    )
    geo = pd.read_csv(
        loader.data_dir / "olist_geolocation_dataset.csv",
        usecols=["geolocation_zip_code_prefix"],
    )
    ok = set(
        customers.loc[
            customers["customer_zip_code_prefix"].isin(
                set(geo["geolocation_zip_code_prefix"].unique())
            ),
            "customer_id",
        ]
    )
    missing = ~labelled["customer_id"].isin(ok)
    assert int(missing.sum()) == GEO_ZERO_MATURED
    ratio = float(labelled.loc[missing, SECONDARY_LABEL].mean()) / float(
        labelled[SECONDARY_LABEL].mean()
    )
    assert ratio < FLAG_RATIO


def test_seller_side_geolocation_gap_is_the_item_artifact_not_a_new_leak(
    loader: OlistLoader, labelled, risk_set
):
    """
    `route_distance_km` needs BOTH endpoints geocoded, so the seller side of the
    geolocation join needs its own entry.

    It trips the flag on the matured set - 26x the panel label_a rate - and the pin
    asserts *why*: the join runs through `order_items`, so an order with no item rows
    reaches no seller to geocode.  What is left after removing those is a plain
    reference-table miss, and on the risk set the group's failure rate is BELOW the
    panel, not above it.
    """
    items = pd.read_csv(
        loader.data_dir / "olist_order_items_dataset.csv",
        usecols=["order_id", "seller_id"],
    )
    sellers = pd.read_csv(
        loader.data_dir / "olist_sellers_dataset.csv",
        usecols=["seller_id", "seller_zip_code_prefix"],
    )
    geo_zips = set(
        pd.read_csv(
            loader.data_dir / "olist_geolocation_dataset.csv",
            usecols=["geolocation_zip_code_prefix"],
        )["geolocation_zip_code_prefix"].unique()
    )
    ok_sellers = set(
        sellers.loc[sellers["seller_zip_code_prefix"].isin(geo_zips), "seller_id"]
    )
    reached = set(items.loc[items["seller_id"].isin(ok_sellers), "order_id"])
    has_items = set(items["order_id"])

    missing = ~labelled["order_id"].isin(reached)
    assert int(missing.sum()) == GEO_SELLER_ZERO_MATURED

    # It trips the flag, and the reason is the items artifact, not the reference table.
    ratio = float(labelled.loc[missing, SECONDARY_LABEL].mean()) / float(
        labelled[SECONDARY_LABEL].mean()
    )
    assert ratio > FLAG_RATIO
    due_to_items = int((missing & ~labelled["order_id"].isin(has_items)).sum())
    assert due_to_items == GEO_SELLER_DUE_TO_ITEMS

    # On the population the model is trained and scored on, the group is inert.
    r_missing = ~risk_set["order_id"].isin(reached)
    assert int(r_missing.sum()) == GEO_SELLER_ZERO_RISK
    r_ratio = float(risk_set.loc[r_missing, PRIMARY_LABEL].mean()) / float(
        risk_set[PRIMARY_LABEL].mean()
    )
    assert r_ratio < 1.0, "seller-side geocoding gap must not be enriched for failures"


def test_geocoding_failure_is_not_itself_a_feature():
    """
    The audit admits reading a joined *value*; it does not admit manufacturing a feature
    out of the join's *failure*.  ARCHITECTURE.md 4.5 lists geocoding confidence as a
    candidate feature and it is deliberately not built.
    """
    from features.builder import FEATURE_GROUPS

    cols = set(
        FeatureBuilder(groups=FEATURE_GROUPS).build(
            OlistLoader().risk_set().head(500)
        ).columns
    )
    banned = {"has_geo_centroid", "has_geolocation", "geocode_confidence",
              "has_route_distance", "zip_resolved"}
    assert not (cols & banned), sorted(cols & banned)


# ---------------------------------------------------------------------------------
# Availability flags as proxies
# ---------------------------------------------------------------------------------

def test_has_item_rows_is_the_artifact_restated(loader: OlistLoader, labelled):
    matrix = FeatureBuilder(groups=("availability",)).build(labelled)
    v = matrix["has_item_rows"].astype(float).to_numpy()
    y_a = labelled[SECONDARY_LABEL].astype(float).to_numpy()
    y_b = labelled[PRIMARY_LABEL].astype(float).to_numpy()

    # Severe against the secondary target, inert against the primary.
    assert abs(float(np.corrcoef(v, y_a)[0, 1])) > 0.4
    assert abs(float(np.corrcoef(v, y_b)[0, 1])) < 0.05


def test_n_missing_features_inherits_the_same_artifact(loader: OlistLoader, labelled):
    matrix = FeatureBuilder().build(labelled)
    v = matrix["n_missing_features"].astype(float).to_numpy()
    y_a = labelled[SECONDARY_LABEL].astype(float).to_numpy()
    assert abs(float(np.corrcoef(v, y_a)[0, 1])) > 0.4


def test_availability_flags_are_inert_on_the_risk_set(loader: OlistLoader, risk_set):
    """
    The conditionality that matters: these features are admissible on the population the
    primary target uses, and inadmissible off it.
    """
    matrix = FeatureBuilder().build(risk_set)
    y = risk_set[PRIMARY_LABEL].astype(float).to_numpy()
    for col in [c for c in matrix.columns if c.startswith("has_")] + [
        "n_missing_features"
    ]:
        v = matrix[col].astype(float).to_numpy()
        if float(np.nanstd(v)) == 0.0:
            continue
        assert abs(float(np.corrcoef(v, y)[0, 1])) < 0.05, col


# ---------------------------------------------------------------------------------
# Task H - the correction cannot silently revert
# ---------------------------------------------------------------------------------

def test_corrected_secondary_ap_recomputed_from_raw(loader: OlistLoader):
    """
    Recomputes the corrected secondary average precision end to end from the raw data -
    population, features, fit, score - and pins it.

    Without this, a reload path that quietly restored the 767 no-item orders to the
    evaluation population would put AP back near 0.295 and nothing would object; the
    correction lives in a markdown file otherwise.
    """
    matured = loader.labelled().join(loader.split_labelled()["split"], how="left")
    item_ids = set(
        pd.read_csv(loader.data_dir / "olist_order_items_dataset.csv",
                    usecols=["order_id"])["order_id"]
    )
    clean = matured[matured["order_id"].isin(item_ids)].copy()
    assert len(clean) == MATURED - ITEMS_ZERO_MATURED

    matrix = FeatureBuilder().build(clean)
    split = clean["split"].to_numpy()
    y = clean[SECONDARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=SECONDARY_LABEL, population="matured, item-joined only",
        category_levels=levels,
    )
    metrics = evaluate(y[te], predict(bundle, X.loc[te]))

    assert round(metrics.average_precision, 4) == CORRECTED_SECONDARY_AP, (
        f"corrected secondary AP is {metrics.average_precision:.4f}, expected "
        f"{CORRECTED_SECONDARY_AP}. If the leaked population crept back in this will "
        "read near 0.295."
    )
