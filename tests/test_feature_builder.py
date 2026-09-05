"""
Feature builder: whitelist enforcement at load, group coverage, determinism, and the
carrier-date perturbation extended to every feature group.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.loader import FORBIDDEN_FEATURE_COLUMNS, LeakageError, OlistLoader
from data.whitelist import parse_column_whitelist
from features.builder import FEATURE_GROUPS

CARRIER = "order_delivered_carrier_date"
#: Imported, not retyped: a new feature group must inherit the perturbation gate below
#: automatically rather than by someone remembering to add it here.
GROUPS = FEATURE_GROUPS


# ---------------------------------------------------------------------------------
# Whitelist enforced at load, against the document
# ---------------------------------------------------------------------------------

def test_whitelist_document_parses():
    wl = parse_column_whitelist()
    assert set(wl) >= {"orders", "customers", "payments", "items", "products",
                       "sellers", "geolocation"}
    assert "order_purchase_timestamp" in wl["orders"]


def test_code_whitelist_matches_the_document():
    """
    data/COLUMN_WHITELIST.md is the stated rule; CHECKOUT_SAFE_COLUMNS is the enforced
    one.  If they drift, the document is decoration.
    """
    from data.loader import CHECKOUT_SAFE_COLUMNS

    doc = parse_column_whitelist()
    code = {k: set(v) for k, v in CHECKOUT_SAFE_COLUMNS.items()}
    assert code == {k: set(v) for k, v in doc.items()}


def test_load_table_rejects_a_column_absent_from_the_document(loader: OlistLoader):
    with pytest.raises(LeakageError, match="not on the whitelist"):
        loader.load_table("orders", columns=["order_id", CARRIER])


def test_load_table_rejects_an_invented_column(loader: OlistLoader):
    with pytest.raises(LeakageError, match="not on the whitelist"):
        loader.load_table("orders", columns=["order_id", "totally_made_up"])


def test_load_table_returns_only_whitelisted_columns(loader: OlistLoader):
    doc = parse_column_whitelist()
    for table in doc:
        got = set(loader.load_table(table).columns)
        assert got <= doc[table], f"{table} leaked {sorted(got - doc[table])}"


def test_unknown_table_is_rejected(loader: OlistLoader):
    with pytest.raises(KeyError):
        loader.load_table("reviews")


# ---------------------------------------------------------------------------------
# The matrix itself
# ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matrix(loader: OlistLoader):
    from features.builder import FeatureBuilder

    return FeatureBuilder().build(loader.risk_set())


def test_matrix_is_row_aligned(loader: OlistLoader, matrix):
    assert len(matrix) == len(loader.risk_set())


def test_matrix_carries_no_forbidden_column(matrix):
    assert not (set(matrix.columns) & FORBIDDEN_FEATURE_COLUMNS)


def test_matrix_carries_no_label(matrix):
    for col in ("label_a", "label_b", "entered_fulfillment", "order_status"):
        assert col not in matrix.columns


def test_every_group_contributes_columns(loader: OlistLoader):
    from features.builder import FeatureBuilder

    population = loader.risk_set().head(2000)
    for group in GROUPS:
        cols = FeatureBuilder(groups=(group,)).build(population).columns
        assert len(cols) > 0, f"group {group} produced no features"


def test_address_and_ring_groups_are_absent(matrix):
    """ARCHITECTURE.md 4.2 - excluded from the trained model until a v2 retrain."""
    banned = ("addr_", "address_", "ring_", "component_", "device_", "phone_")
    offenders = [c for c in matrix.columns if c.startswith(banned)]
    assert not offenders, f"unevaluable feature groups present: {offenders}"


def test_n_missing_features_matches_actual_nulls(matrix):
    feature_cols = [c for c in matrix.columns if c != "n_missing_features"]
    expected = matrix[feature_cols].isna().sum(axis=1)
    pd.testing.assert_series_equal(
        matrix["n_missing_features"], expected, check_names=False, check_dtype=False
    )


def test_build_is_deterministic(loader: OlistLoader):
    from features.builder import FeatureBuilder

    # Random sample rather than head(5000): a nondeterminism triggered by particular
    # data (a hash-ordering dependence, say) would not be in the earliest rows by
    # construction.
    risk = loader.risk_set()
    population = risk[risk["order_id"].isin(
        set(risk["order_id"].sample(n=20_000, random_state=20260903))
    )]
    pd.testing.assert_frame_equal(
        FeatureBuilder().build(population), FeatureBuilder().build(population)
    )


# ---------------------------------------------------------------------------------
# Carrier-date perturbation, extended to every feature group
# ---------------------------------------------------------------------------------

def _blanked(loader: OlistLoader) -> OlistLoader:
    other = OlistLoader(data_dir=loader.data_dir)
    raw = loader.raw_orders()
    raw[CARRIER] = pd.NaT
    other._raw = raw
    return other


@pytest.mark.parametrize("group", GROUPS)
def test_group_is_unaffected_by_carrier_date(loader: OlistLoader, group):
    """
    Blanking the carrier date must not move any feature in any group.  The risk set is
    defined by that column, so the populations are compared on their shared orders.
    """
    from features.builder import FeatureBuilder

    population = loader.labelled()
    baseline = FeatureBuilder(groups=(group,)).build(population)

    other = _blanked(loader)
    perturbed = FeatureBuilder(groups=(group,)).build(other.labelled())

    pd.testing.assert_frame_equal(baseline, perturbed)


def test_full_matrix_is_unaffected_by_carrier_date(loader: OlistLoader):
    from features.builder import FeatureBuilder

    baseline = FeatureBuilder().build(loader.labelled())
    perturbed = FeatureBuilder().build(_blanked(loader).labelled())
    pd.testing.assert_frame_equal(baseline, perturbed)


def test_every_buildable_group_together_is_unaffected_by_carrier_date(
    loader: OlistLoader,
):
    """
    The default matrix is not every group.  The unshipped expansion groups are covered
    individually above; this covers them assembled, which is the configuration the
    ablation actually trained on.
    """
    from features.builder import FEATURE_GROUPS, FeatureBuilder

    # A RANDOM 20% sample, not head(20_000).  `head` takes the earliest rows in frame
    # order, which correlates with purchase time - so it systematically excludes the
    # late orders where a history-derived defect would be largest.  That is a worse
    # sample than a stride, not a cheaper one.  The full population would be two
    # 54-column builds (~250s); the per-group tests above each cover it in full, and
    # this one exists for the cross-group coupling (`n_missing_features`) that they
    # cannot see.
    ids = loader.labelled()["order_id"].sample(n=20_000, random_state=20260903)
    population = loader.labelled()
    population = population[population["order_id"].isin(set(ids))]

    baseline = FeatureBuilder(groups=FEATURE_GROUPS).build(population)
    other = _blanked(loader)
    other_pop = other.labelled()
    other_pop = other_pop[other_pop["order_id"].isin(set(ids))]
    perturbed = FeatureBuilder(groups=FEATURE_GROUPS).build(other_pop)
    pd.testing.assert_frame_equal(baseline, perturbed)


# ---------------------------------------------------------------------------------
# What the default builder ships
# ---------------------------------------------------------------------------------

def test_default_groups_exclude_the_unresolved_expansion():
    """
    eval/feature_expansion.md: the paired bootstrap on AP(expanded) - AP(current) has a
    95% CI that includes zero, so the pre-fixed decision rule keeps the current model.

    This asserts the code honours that verdict.  Flipping it is a deliberate act that
    fails here first, rather than a default that quietly widened.
    """
    from features.builder import DEFAULT_GROUPS, EXPANSION_GROUPS, FEATURE_GROUPS

    assert set(DEFAULT_GROUPS) & set(EXPANSION_GROUPS) == set()
    assert set(DEFAULT_GROUPS) | set(EXPANSION_GROUPS) == set(FEATURE_GROUPS)
    assert DEFAULT_GROUPS == ("order", "customer", "pincode", "availability")


def test_shipped_matrix_width_is_unchanged_by_the_expansion(loader: OlistLoader):
    """The expansion added code, not columns to the shipped model."""
    from features.builder import FEATURE_GROUPS, FeatureBuilder

    population = loader.risk_set().head(3000)
    assert FeatureBuilder().build(population).shape[1] == 35
    assert FeatureBuilder(groups=FEATURE_GROUPS).build(population).shape[1] == 54
