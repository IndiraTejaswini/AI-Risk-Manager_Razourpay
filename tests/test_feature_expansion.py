"""
Pins the seller/route/structure/density ablation (eval/feature_expansion.md).

The features were built, passed every gate, and **did not ship**: the paired bootstrap
on AP(expanded) - AP(current) has a 95% interval that includes zero, and the decision
rule was fixed before the run.

These tests hold three separate lines:

  * the *verdict* is honoured in code - the default builder does not quietly widen;
  * the *numbers in the report* are reproducible from raw data;
  * the *features themselves* still satisfy the gates, so the ablation stays rerunnable
    rather than rotting behind a flag nobody exercises.

The third matters most. Code that is built but not shipped is exactly the code that
stops being correct without anyone noticing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.loader import FORBIDDEN_FEATURE_COLUMNS, PRIMARY_LABEL, OlistLoader
from features.builder import (
    DEFAULT_GROUPS,
    EXPANSION_GROUPS,
    FEATURE_GROUPS,
    MULTI_SELLER_RULE,
    ROUTE_SMOOTHING,
    SELLER_SMOOTHING,
    FeatureBuilder,
)
from features.pit import (
    assert_truncation_invariant,
    pit_prior_count,
    pit_prior_count_asof,
    pit_prior_first,
)
from models.constraints import (
    MONOTONE_CONSTRAINTS,
    UNSHIPPED_CONSTRAINTS,
    build_constraint_vector,
)
from models.evaluate import evaluate
from models.significance import paired_bootstrap_ap
from models.train import predict, prepare_matrix, train

TS = "order_purchase_timestamp"

# --- eval/feature_expansion.md ---------------------------------------------------------
N_NEW_FEATURES = 19
CURRENT_WIDTH = 35                 # before prepare_matrix drops has_zip_prefix
EXPANDED_WIDTH = 54
MULTI_SELLER_ORDERS = 1_277

CURRENT_AP = 0.0171
EXPANDED_AP = 0.0143
DELTA_AP = -0.0028
DELTA_CI = (-0.0180, +0.0027)

#: The two new features that carry a monotone constraint, and why only these two.
NEW_CONSTRAINED = ("seller_failure_rate_smoothed", "route_pair_failure_rate_smoothed")

#: The order-only baseline is the matrix that legitimately lacks required constraints.
ORDER_ONLY_ABSENT = frozenset({
    "pincode_failure_rate_smoothed",
    "cust_prior_failures",
    "cust_prior_boleto_ratio",
})


@pytest.fixture(scope="module")
def matrices(loader: OlistLoader):
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    return {
        "risk": risk,
        "current": FeatureBuilder(groups=DEFAULT_GROUPS, loader=loader).build(risk),
        "expanded": FeatureBuilder(groups=FEATURE_GROUPS, loader=loader).build(risk),
    }


# ---------------------------------------------------------------------------------
# The verdict, honoured in code
# ---------------------------------------------------------------------------------

def test_the_expansion_is_not_in_the_shipped_matrix(matrices):
    assert matrices["current"].shape[1] == CURRENT_WIDTH
    assert matrices["expanded"].shape[1] == EXPANDED_WIDTH
    assert EXPANDED_WIDTH - CURRENT_WIDTH == N_NEW_FEATURES
    assert set(DEFAULT_GROUPS).isdisjoint(EXPANSION_GROUPS)


def test_new_features_carry_no_forbidden_column(matrices):
    assert not (set(matrices["expanded"].columns) & FORBIDDEN_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------------
# The gates, still passing
# ---------------------------------------------------------------------------------

def test_expansion_groups_are_point_in_time(loader: OlistLoader):
    """
    Deleting the future must not move a single new feature.

    A general check: it needs no knowledge of how any individual feature is computed,
    so it stays valid as the group definitions change.
    """
    risk = loader.risk_set()
    sample = risk.sample(n=12_000, random_state=42).sort_values(TS)
    assert_truncation_invariant(
        FeatureBuilder(groups=EXPANSION_GROUPS + ("order",), loader=loader),
        sample,
        loader.split_boundary,
    )


def test_expansion_groups_do_not_read_the_carrier_date(loader: OlistLoader):
    """
    The seller dispatch feature is the *contractual* window precisely so this holds.
    A realised-dispatch-delay feature would need `order_delivered_carrier_date` and
    would fail here.
    """
    other = OlistLoader(data_dir=loader.data_dir)
    raw = loader.raw_orders()
    raw["order_delivered_carrier_date"] = pd.NaT
    other._raw = raw

    # Random sample, not head(): see tests/test_feature_builder.py for why `head` is
    # the worst available choice here.
    lab = loader.labelled()
    ids = set(lab["order_id"].sample(n=20_000, random_state=20260903))
    population = lab[lab["order_id"].isin(ids)]
    other_pop = other.labelled()
    other_pop = other_pop[other_pop["order_id"].isin(ids)]

    before = FeatureBuilder(groups=EXPANSION_GROUPS, loader=loader).build(population)
    after = FeatureBuilder(groups=EXPANSION_GROUPS, loader=other).build(other_pop)
    pd.testing.assert_frame_equal(before, after)


def test_constraint_vector_covers_the_expanded_matrix(matrices):
    names = list(matrices["expanded"].columns)
    vector = build_constraint_vector(names)
    assert len(vector) == len(names)
    assert set(UNSHIPPED_CONSTRAINTS) == set(NEW_CONSTRAINED)
    for feature in NEW_CONSTRAINED:
        assert UNSHIPPED_CONSTRAINTS[feature] == +1
        # index() raises if the feature was renamed without renaming the constraint -
        # which is the protection that moved here when the map was split in two.
        assert vector[names.index(feature)] == +1


def test_the_shipped_matrix_needs_no_exception(matrices):
    """
    The unshipped constraints must not make the *normal* case declare an exception.

    They were briefly in MONOTONE_CONSTRAINTS, which required every ordinary caller -
    including the serving path - to name two features it does not carry.  A guard that
    fires on the normal case is a guard people learn to silence, so the map was split:
    required constraints in one, applied-where-present in the other.
    """
    build_constraint_vector(list(matrices["current"].columns))
    build_constraint_vector(list(matrices["expanded"].columns))


def test_a_reduced_matrix_must_still_name_what_it_drops(matrices):
    """The guard itself, unchanged, on the matrix it was written for."""
    from models.constraints import ConstraintError

    order_only = FeatureBuilder(groups=("order",)).build(
        matrices["risk"].head(2000)
    )
    names = list(order_only.columns)
    with pytest.raises(ConstraintError, match="absent from the matrix"):
        build_constraint_vector(names)
    build_constraint_vector(names, ORDER_ONLY_ABSENT)          # named: allowed


def test_store_reproduces_the_expanded_matrix_exactly(loader: OlistLoader, matrices):
    """
    Serving parity for the unshipped groups too.

    This is the check that caught the pairwise-vs-sequential float summation gap: the
    seller index sums groups thousands of rows long, where the customer index never
    exceeded a handful.

    **Every row**, for the reason given in tests/test_store.py: this one used a
    60-order stride and the defect it is named for was found by the startup check, not
    by this test.  ~42s over the 54-column matrix.
    """
    builder = FeatureBuilder(groups=FEATURE_GROUPS, loader=loader)
    risk = matrices["risk"]
    src = builder._sources(risk)
    store = builder.history_store(src)
    via_store = builder.build(risk, store=store)
    expected = matrices["expanded"]

    for col in expected.columns:
        a, b = via_store[col].to_numpy(), expected[col].to_numpy()
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


def test_every_new_feature_has_a_group_and_a_reason_string(matrices):
    """
    The explainer's maps must cover the unshipped features too.

    tests/test_explain.py checks coverage of the *shipped* matrix, which is a subset -
    so without this, a new feature could sit in the builder for months with no group and
    no merchant-facing sentence, and only fail on the day it ships.
    """
    from models.explain import FEATURE_GROUP, REASON_TEMPLATES
    from models.train import DROPPED_CONSTANT_COLUMNS

    # has_zip_prefix is constant on this panel and dropped before the model sees it, so
    # it has no attribution to group and nothing to say to a merchant.
    for col in matrices["expanded"].columns.drop(list(DROPPED_CONSTANT_COLUMNS)):
        assert col in FEATURE_GROUP, f"{col} has no SHAP group"
        assert col in REASON_TEMPLATES, f"{col} has no reason template"
        sentence = REASON_TEMPLATES[col]
        # Same house style the shipped reasons are held to: a sentence a merchant reads,
        # not a feature name with a value stapled to it.
        assert "_" not in sentence and "=" not in sentence, sentence
        assert sentence[0].isupper() and sentence.rstrip().endswith("."), sentence


# ---------------------------------------------------------------------------------
# The rolling-window primitive
# ---------------------------------------------------------------------------------

def _window_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "key": ["a", "a", "a", "a", "b"],
        TS: pd.to_datetime([
            "2024-01-01", "2024-01-05", "2024-01-09", "2024-01-09", "2024-01-02",
        ]),
    })


def test_asof_count_with_zero_offset_is_the_plain_prior_count():
    df = _window_fixture()
    plain = pit_prior_count(df, "key", TS)
    asof = pit_prior_count_asof(df, "key", TS, pd.Timedelta(0))
    pd.testing.assert_series_equal(plain, asof, check_names=False)


def test_rolling_window_is_half_open_and_tie_safe():
    """
    [t - w, t): a row at exactly *t* is not prior to itself, a row at exactly t - w is
    inside the window, and two rows sharing a timestamp do not see each other.
    """
    df = _window_fixture()
    total = pit_prior_count(df, "key", TS)
    before = pit_prior_count_asof(df, "key", TS, pd.Timedelta(days=7))
    window = (total - before).tolist()

    # a: 01-01 sees nothing; 01-05 sees 01-01; both 01-09 rows see 01-05 only
    # (01-01 is 8 days back, outside the 7-day window; the tied row is not prior).
    assert window == [0, 1, 1, 1, 0]


def test_prior_first_is_nat_for_a_groups_own_first_row():
    df = _window_fixture()
    first = pit_prior_first(df, "key", TS)
    assert pd.isna(first.iloc[0])          # a's earliest row
    assert pd.isna(first.iloc[4])          # b's only row
    assert first.iloc[1] == pd.Timestamp("2024-01-01")
    assert first.iloc[2] == pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------------------------
# Multi-seller aggregation - the rule, not whatever a groupby kept
# ---------------------------------------------------------------------------------

def test_multi_seller_order_count(loader: OlistLoader, risk_set):
    items = loader.load_table("items", ["order_id", "seller_id"])
    n = (
        items[items["order_id"].isin(set(risk_set["order_id"]))]
        .groupby("order_id")["seller_id"]
        .nunique()
    )
    assert int((n > 1).sum()) == MULTI_SELLER_ORDERS


def test_every_aggregated_feature_has_a_stated_rule(matrices):
    seller_route = [
        c for c in matrices["expanded"].columns
        if c.startswith("seller_") and c != "seller_state"
    ]
    for col in seller_route:
        assert col in MULTI_SELLER_RULE, f"{col} aggregates across sellers with no rule"


def test_seller_risk_takes_the_max_across_an_orders_sellers(loader: OlistLoader):
    """
    A parcel fails if any of its sellers fails, so the order inherits the worst one.
    Asserted on real multi-seller orders rather than on the docstring.
    """
    risk = loader.risk_set()
    builder = FeatureBuilder(groups=("seller",), loader=loader)
    items = loader.load_table("items", ["order_id", "seller_id"])
    counts = items.groupby("order_id")["seller_id"].nunique()
    multi = set(counts[counts > 1].index)

    population = risk[risk["order_id"].isin(multi)]      # all 1,277, not head(400)
    src = builder._sources(population)
    edges = builder.edge_frame(src)
    built = builder.build(population)

    # Rebuild the per-edge rate the same way the group does, then check the order-level
    # value is the maximum over that order's edges.
    from features.pit import pit_expanding_mean, pit_smoothed_rate

    gp = pit_expanding_mean(src, TS, "label_a")
    e = edges.copy()
    e["_gp"] = e["order_id"].map(
        pd.Series(gp.to_numpy(), index=src["order_id"].to_numpy())
    )
    per_edge = pit_smoothed_rate(
        e, "seller_id", TS, "label_a", SELLER_SMOOTHING, None, global_prior=e["_gp"]
    )
    expected = (
        pd.DataFrame({"order_id": e["order_id"].to_numpy(), "r": per_edge.to_numpy()})
        .groupby("order_id")["r"].max()
        .reindex(population["order_id"].to_numpy())
        .to_numpy()
    )
    got = built["seller_failure_rate_smoothed"].to_numpy()
    assert np.allclose(got, expected, equal_nan=True)

    # And it is genuinely a choice: on these orders the max differs from the mean.
    mean = (
        pd.DataFrame({"order_id": e["order_id"].to_numpy(), "r": per_edge.to_numpy()})
        .groupby("order_id")["r"].mean()
        .reindex(population["order_id"].to_numpy())
        .to_numpy()
    )
    assert not np.allclose(got, mean, equal_nan=True)


def test_smoothing_constants_match_the_pincode_encoding():
    """m = 50 for seller and route, the same pseudo-count the pincode encoding uses."""
    from features.builder import PINCODE_SMOOTHING

    assert SELLER_SMOOTHING == ROUTE_SMOOTHING == PINCODE_SMOOTHING == 50.0


# ---------------------------------------------------------------------------------
# The headline: the ablation does not resolve
# ---------------------------------------------------------------------------------

def _fit_and_score(risk, matrix, allow_missing=frozenset()):
    split = risk["split"].to_numpy()
    y = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"
    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=PRIMARY_LABEL, population="risk_set", category_levels=levels,
        allow_missing_constraints=allow_missing,
    )
    return predict(bundle, X.loc[te]), y[te]


@pytest.fixture(scope="module")
def scores(matrices):
    """Both fits, once.  Each is ~30s; the tests below share them."""
    risk = matrices["risk"]
    cur, y_test = _fit_and_score(risk, matrices["current"])
    exp, _ = _fit_and_score(risk, matrices["expanded"])
    return {"current": cur, "expanded": exp, "y": y_test}


def test_the_expansion_does_not_resolve(scores):
    """
    The number the decision rests on, recomputed from raw rather than read back from
    the markdown file it is reported in.
    """
    y = scores["y"]
    assert evaluate(y, scores["current"]).average_precision == pytest.approx(
        CURRENT_AP, abs=5e-5
    )
    assert evaluate(y, scores["expanded"]).average_precision == pytest.approx(
        EXPANDED_AP, abs=5e-5
    )

    result = paired_bootstrap_ap(
        y, scores["expanded"], scores["current"], name="expanded - current"
    )
    assert result.observed == pytest.approx(DELTA_AP, abs=5e-5)
    assert (result.ci_low, result.ci_high) == pytest.approx(DELTA_CI, abs=5e-5)

    # The verdict itself: the interval includes zero, so the features do not ship.
    assert not result.excludes_zero
    assert result.ci_low < 0.0 < result.ci_high


def test_the_current_model_is_the_one_step_three_reported(scores):
    """
    The expansion changed how features/pit.py takes an exclusive-of-self prior sum -
    a shift instead of a subtraction, for float exactness against the serving index.
    This asserts the change moved no committed number: the shipped model is still the
    one eval/model_report.md describes.
    """
    m = evaluate(scores["y"], scores["current"])
    assert m.average_precision == pytest.approx(0.0171, abs=5e-5)
    assert m.roc_auc == pytest.approx(0.6450, abs=5e-5)
