"""
Step 3 model assertions.

Covers the constraint contract, the column-order contract, the training-boundary
leakage gate, determinism, and the join-cardinality artifact found while evaluating the
secondary target.

The fits here run on a subsample so the suite stays usable; the full-data determinism
check runs in scripts/04_train_model.py, which exits non-zero if it fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import PRIMARY_LABEL, SECONDARY_LABEL, LeakageError, OlistLoader
from features.builder import FeatureBuilder
from models.constraints import (
    INVERTED_FROM_SPEC,
    MONOTONE_CONSTRAINTS,
    OMITTED_FROM_SPEC,
    ConstraintError,
    build_constraint_vector,
)
from models.train import (
    DROPPED_CONSTANT_COLUMNS,
    ORDER_ONLY_ABSENT_CONSTRAINTS,
    PARAMS,
    FeatureContractError,
    predict,
    prepare_matrix,
    train,
)


# ---------------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------------

def test_sign_map_is_a_named_module_constant():
    """Not inlined at a call site: the map is importable, named, and non-empty."""
    assert isinstance(MONOTONE_CONSTRAINTS, dict)
    assert MONOTONE_CONSTRAINTS
    assert all(v in (-1, 0, 1) for v in MONOTONE_CONSTRAINTS.values())


def test_spec_constraints_are_all_accounted_for():
    """Every §5.3 constraint is either mapped or explicitly omitted with a reason."""
    assert set(MONOTONE_CONSTRAINTS) == {
        "pincode_failure_rate_smoothed",
        "cust_prior_failures",
        "cust_prior_boleto_ratio",
    }
    assert set(OMITTED_FROM_SPEC) == {"component_size"}
    assert OMITTED_FROM_SPEC["component_size"]


def test_pincode_and_prior_failures_increase_risk():
    assert MONOTONE_CONSTRAINTS["pincode_failure_rate_smoothed"] == +1
    assert MONOTONE_CONSTRAINTS["cust_prior_failures"] == +1


def test_boleto_ratio_sign_is_inverted_from_the_spec_deliberately():
    """
    §5.3 says prepaid_ratio ↑ → risk ↓, which is -1.  This matrix carries the boleto
    ratio, the complement, so the correct constraint is +1.  Pinned so nobody reading
    §5.3 alone "corrects" it to -1 and silently forces a backwards relationship.
    """
    assert MONOTONE_CONSTRAINTS["cust_prior_boleto_ratio"] == +1
    assert "cust_prior_boleto_ratio" in INVERTED_FROM_SPEC
    assert INVERTED_FROM_SPEC["cust_prior_boleto_ratio"]


def test_constraint_vector_length_matches_feature_count():
    names = ["a", "pincode_failure_rate_smoothed", "b", "cust_prior_failures",
             "cust_prior_boleto_ratio", "c"]
    vec = build_constraint_vector(names)
    assert len(vec) == len(names)
    assert vec == [0, 1, 0, 1, 1, 0]


def test_constraint_vector_rejects_a_silently_dropped_constraint():
    with pytest.raises(ConstraintError, match="pincode_failure_rate_smoothed"):
        build_constraint_vector(["cust_prior_failures", "cust_prior_boleto_ratio"])


def test_allow_missing_is_not_a_blanket_switch():
    """A reduced matrix must name what it omits; a typo in that list still raises."""
    with pytest.raises(ConstraintError, match="carry no constraint"):
        build_constraint_vector(["a"], allow_missing=frozenset({"not_a_feature"}))


def test_allow_missing_still_catches_an_unlisted_constraint():
    partial = frozenset({"cust_prior_failures", "cust_prior_boleto_ratio"})
    with pytest.raises(ConstraintError, match="pincode_failure_rate_smoothed"):
        build_constraint_vector(["order_value"], allow_missing=partial)


def test_order_only_absent_set_covers_exactly_the_constrained_features():
    assert ORDER_ONLY_ABSENT_CONSTRAINTS == frozenset(MONOTONE_CONSTRAINTS)


# ---------------------------------------------------------------------------------
# Hyperparameters (§5.2)
# ---------------------------------------------------------------------------------

def test_params_sit_inside_the_specified_bands():
    assert 15 <= PARAMS["num_leaves"] <= 31
    assert 4 <= PARAMS["max_depth"] <= 6
    assert 50 <= PARAMS["min_child_samples"] <= 100
    assert PARAMS["lambda_l2"] > 0
    assert 0.6 <= PARAMS["feature_fraction"] <= 0.8


def test_early_stopping_metric_is_average_precision():
    """Not AUC, not logloss."""
    assert PARAMS["metric"] == "average_precision"


def test_natural_class_distribution():
    assert "scale_pos_weight" not in PARAMS
    assert "is_unbalance" not in PARAMS
    assert PARAMS["bagging_fraction"] == 1.0


def test_determinism_flags_are_set():
    assert PARAMS["deterministic"] is True
    assert PARAMS["force_row_wise"] is True
    assert PARAMS["seed"] == 42
    # LightGBM's guarantee holds for the same data AND thread count.
    assert isinstance(PARAMS["num_threads"], int) and PARAMS["num_threads"] > 0


# ---------------------------------------------------------------------------------
# Fitted-model fixtures
# ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted(loader: OlistLoader):
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    risk = risk[risk["split"].isin(["train", "validation"])]
    # Subsample for suite speed; stable slice, no randomness.
    risk = risk.iloc[::4]
    matrix = FeatureBuilder().build(risk)

    split = risk["split"].to_numpy()
    y = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va = split == "train", split == "validation"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)

    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=PRIMARY_LABEL, population="test-subsample", category_levels=levels,
    )
    return bundle, X.loc[va]


def test_constant_column_is_dropped_from_the_trained_matrix(fitted):
    bundle, _ = fitted
    for col in DROPPED_CONSTANT_COLUMNS:
        assert col not in bundle.feature_names


def test_constraint_vector_matches_the_trained_feature_count(fitted):
    bundle, _ = fitted
    assert len(bundle.monotone_constraints) == len(bundle.feature_names)


def test_constraints_land_on_the_right_features(fitted):
    bundle, _ = fitted
    for name, expected in MONOTONE_CONSTRAINTS.items():
        i = bundle.feature_names.index(name)
        assert bundle.monotone_constraints[i] == expected


# ---------------------------------------------------------------------------------
# Column-order contract
# ---------------------------------------------------------------------------------

def test_predict_accepts_the_exact_contract(fitted):
    bundle, X = fitted
    assert len(predict(bundle, X)) == len(X)


def test_predict_rejects_reordered_columns_rather_than_reordering(fitted):
    """The assertion this contract exists for."""
    bundle, X = fitted
    cols = list(X.columns)
    swapped = X[[cols[1], cols[0]] + cols[2:]]
    with pytest.raises(FeatureContractError, match="wrong ORDER"):
        predict(bundle, swapped)


def test_predict_rejects_a_missing_column(fitted):
    bundle, X = fitted
    with pytest.raises(FeatureContractError, match="missing"):
        predict(bundle, X.drop(columns=[X.columns[0]]))


def test_predict_rejects_an_extra_column(fitted):
    bundle, X = fitted
    extra = X.copy()
    extra["surprise"] = 1.0
    with pytest.raises(FeatureContractError, match="unexpected"):
        predict(bundle, extra)


def test_reordering_would_have_changed_the_predictions(fitted):
    """
    Proves the contract is load-bearing rather than pedantic: scoring the permuted frame
    without the guard produces different numbers, silently.
    """
    bundle, X = fitted
    cols = list(X.columns)
    swapped = X[[cols[1], cols[0]] + cols[2:]]
    straight = bundle.booster.predict(X, num_iteration=bundle.best_iteration)
    permuted = bundle.booster.predict(swapped, num_iteration=bundle.best_iteration)
    assert not np.allclose(straight, permuted)


def test_contract_is_recorded_in_the_saved_artifact(fitted, tmp_path):
    import json

    bundle, _ = fitted
    bundle.save(tmp_path)
    contract = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    assert contract["feature_names"] == list(bundle.feature_names)
    assert contract["monotone_constraints"] == list(bundle.monotone_constraints)
    assert contract["model_sha256"] == bundle.model_sha256


# ---------------------------------------------------------------------------------
# Leakage gate at the training boundary
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "column", ["order_delivered_carrier_date", "order_status", "review_score"]
)
def test_train_rejects_a_forbidden_column(fitted, column):
    bundle, X = fitted
    bad = X.copy()
    bad[column] = 1
    with pytest.raises((LeakageError, FeatureContractError)):
        train(bad, pd.Series(np.zeros(len(bad), dtype=int)),
              bad, pd.Series(np.zeros(len(bad), dtype=int)),
              target="x", population="x", category_levels={})


@pytest.mark.parametrize("column", ["label_a", "label_b", "split", "entered_fulfillment"])
def test_train_rejects_a_label_column(fitted, column):
    bundle, X = fitted
    bad = X.copy()
    bad[column] = 0
    with pytest.raises(FeatureContractError, match="label or bookkeeping"):
        train(bad, pd.Series(np.zeros(len(bad), dtype=int)),
              bad, pd.Series(np.zeros(len(bad), dtype=int)),
              target="x", population="x", category_levels={})


@pytest.mark.parametrize("column", ["order_id", "customer_id", "customer_unique_id"])
def test_train_rejects_an_identifier(fitted, column):
    bundle, X = fitted
    bad = X.copy()
    bad[column] = "x"
    with pytest.raises(FeatureContractError, match="identifier"):
        train(bad, pd.Series(np.zeros(len(bad), dtype=int)),
              bad, pd.Series(np.zeros(len(bad), dtype=int)),
              target="x", population="x", category_levels={})


# ---------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------

def test_two_fits_produce_identical_artifacts_and_predictions(loader: OlistLoader):
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    risk = risk[risk["split"].isin(["train", "validation"])].iloc[::8]
    matrix = FeatureBuilder().build(risk)

    split = risk["split"].to_numpy()
    y = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va = split == "train", split == "validation"
    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)

    def fit():
        return train(
            X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
            target=PRIMARY_LABEL, population="determinism", category_levels=levels,
        )

    a, b = fit(), fit()
    assert a.model_sha256 == b.model_sha256
    assert np.array_equal(predict(a, X.loc[va]), predict(b, X.loc[va]))


# ---------------------------------------------------------------------------------
# The join-cardinality artifact
# ---------------------------------------------------------------------------------

MATURED_WITHOUT_ITEMS = 767
TEST_WITHOUT_ITEMS = 98
RISK_SET_WITHOUT_ITEMS = 1


def test_item_join_absence_perfectly_predicts_the_secondary_label(loader: OlistLoader):
    """
    Pins the leak documented in data/COLUMN_WHITELIST.md and eval/model_report.md §5.

    An order's absence from order_items is a consequence of its outcome, so any feature
    derived from that join is post-outcome for the secondary target. This test does not
    assert the leak is fixed - it asserts it is still exactly this size, so the
    correction in the report cannot silently go stale.
    """
    matured = loader.labelled()
    item_ids = set(loader.load_table("items", ["order_id"])["order_id"])
    missing = ~matured["order_id"].isin(item_ids)

    assert int(missing.sum()) == MATURED_WITHOUT_ITEMS
    # Every single one is a positive - that is what makes it a perfect predictor.
    assert int(matured.loc[missing, SECONDARY_LABEL].sum()) == MATURED_WITHOUT_ITEMS


def test_the_primary_population_is_immune_to_that_artifact(loader: OlistLoader):
    risk = loader.risk_set()
    item_ids = set(loader.load_table("items", ["order_id"])["order_id"])
    assert int((~risk["order_id"].isin(item_ids)).sum()) == RISK_SET_WITHOUT_ITEMS
