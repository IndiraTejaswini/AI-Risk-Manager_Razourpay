"""
Step 6 SHAP assertions.

The named ones: additivity, monotone constraints hold in the attribution, every order
produces a non-empty reason, determinism.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

from data.loader import PRIMARY_LABEL, OlistLoader  # noqa: E402
from features.builder import FeatureBuilder  # noqa: E402
from models.constraints import INVERTED_FROM_SPEC, MONOTONE_CONSTRAINTS  # noqa: E402
from models.explain import (  # noqa: E402
    FEATURE_GROUP,
    REASON_TEMPLATES,
    AdditivityError,
    ReasonExplainer,
)
from models.train import predict, prepare_matrix, train  # noqa: E402


@pytest.fixture(scope="module")
def explained(loader: OlistLoader):
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    matrix = FeatureBuilder().build(risk)
    split = risk["split"].to_numpy()
    y = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=PRIMARY_LABEL, population="risk_set", category_levels=levels,
    )
    X_te = X.loc[te]
    ex = ReasonExplainer(bundle)
    return {
        "bundle": bundle, "explainer": ex, "X": X_te,
        "expl": ex.explain(X_te),
        "margin": predict(bundle, X_te, raw_score=True),
        "y": y[te],
    }


# ---------------------------------------------------------------------------------
# Additivity - the wiring check
# ---------------------------------------------------------------------------------

def test_shap_sums_to_margin_minus_base(explained):
    """
    A failure here means the explainer is misconfigured - wrong output space, wrong
    feature order - and every reason string downstream would be silently wrong.
    """
    err = explained["explainer"].assert_additive(
        explained["expl"], explained["margin"], tol=1e-6
    )
    assert err < 1e-9


def test_additivity_check_can_fail(explained):
    """A guardrail that cannot fail is not a guardrail."""
    corrupted = explained["margin"] + 1.0
    with pytest.raises(AdditivityError, match="do not reconstruct"):
        explained["explainer"].assert_additive(explained["expl"], corrupted, tol=1e-6)


def test_agrees_with_lightgbm_pred_contrib(explained):
    """
    Independent confirmation from a separate implementation of the same algorithm.
    """
    b = explained["bundle"]
    contrib = b.booster.predict(
        explained["X"], num_iteration=b.best_iteration, pred_contrib=True
    )
    assert np.abs(contrib[:, :-1] - explained["expl"].values).max() == 0.0
    assert contrib[0, -1] == pytest.approx(explained["expl"].base_value)


def test_explain_rejects_a_frame_that_does_not_match_the_contract(explained):
    X = explained["X"]
    cols = list(X.columns)
    with pytest.raises(ValueError, match="does not match the trained contract"):
        explained["explainer"].explain(X[[cols[1], cols[0]] + cols[2:]])


# ---------------------------------------------------------------------------------
# Monotone constraints in the attribution - the section 5.3 payoff
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("feature", sorted(MONOTONE_CONSTRAINTS))
def test_constrained_feature_attribution_moves_the_right_way(explained, feature):
    """
    Probe: sweep the feature, hold everything else fixed, require the attribution to
    move in the constrained direction. This is what stops a merchant-facing string
    saying "this pincode's elevated failure rate reduced risk".
    """
    constraint = MONOTONE_CONSTRAINTS[feature]
    _, attribution = explained["explainer"].probe_monotone(explained["X"], feature)
    d = np.diff(attribution)
    if constraint > 0:
        assert np.all(d >= -1e-9), f"{feature} attribution decreases under a +1 constraint"
    else:
        assert np.all(d <= 1e-9), f"{feature} attribution increases under a -1 constraint"


def test_the_inverted_constraint_is_checked_at_the_applied_sign():
    """
    cust_prior_boleto_ratio is +1 in the model, inverted from section 5.3's
    prepaid_ratio. The monotonicity test must check the sign actually applied - checking
    against the spec's sign would test the wrong thing.
    """
    assert "cust_prior_boleto_ratio" in INVERTED_FROM_SPEC
    assert MONOTONE_CONSTRAINTS["cust_prior_boleto_ratio"] == +1


def test_pincode_failure_rate_actually_contributes(explained):
    """
    The one constrained feature that is not inert. If this went to zero the
    monotonicity checks would all pass vacuously and mean nothing.
    """
    j = explained["explainer"].feature_names.index("pincode_failure_rate_smoothed")
    col = explained["expl"].values[:, j]
    assert np.abs(col).max() > 0.01


# ---------------------------------------------------------------------------------
# Reasons
# ---------------------------------------------------------------------------------

def test_every_order_produces_at_least_one_reason(explained):
    reasons = explained["explainer"].top_reasons(explained["expl"])
    assert len(reasons) == len(explained["X"])
    empty = [i for i, r in enumerate(reasons) if not r]
    assert not empty, f"{len(empty)} orders produced no reason"


def test_reasons_are_capped_and_unique(explained):
    for r in explained["explainer"].top_reasons(explained["expl"], k=3):
        assert len(r) <= 3
        assert len(set(r)) == len(r)


def test_reasons_are_sentences_not_feature_names(explained):
    """
    Merchant-facing phrasing: no snake_case, no bare numbers.

    Every order, not the first 500.  A template is selected per order by attribution, so
    a badly-worded template reaches only the orders where its feature happens to lead -
    which the first 500 rows need not include.
    """
    for r in explained["explainer"].top_reasons(explained["expl"]):
        for s in r:
            assert "_" not in s, s
            assert s[0].isupper() and s.rstrip().endswith(".")
            assert "=" not in s


def test_every_feature_has_a_group_and_a_template(explained):
    for f in explained["explainer"].feature_names:
        assert f in FEATURE_GROUP, f"{f} has no group"
        assert f in REASON_TEMPLATES, f"{f} has no reason template"


def test_reasons_are_ordered_by_attribution(explained):
    """The first reason must correspond to the largest positive attribution."""
    ex, expl = explained["explainer"], explained["expl"]
    reasons = ex.top_reasons(expl)
    # Every order.  This walked a 977-stride, which is the pattern that let two
    # exactness defects through elsewhere in this suite; ordering is cheap to check in
    # full and there is no reason to sample it.
    for i in range(len(reasons)):
        if reasons[i][0].startswith("No elevated"):
            continue
        top_j = int(np.argmax(expl.values[i]))
        assert reasons[i][0] == REASON_TEMPLATES[ex.feature_names[top_j]], i


def test_orders_with_no_positive_attribution_get_the_explicit_sentence(explained):
    ex, expl = explained["explainer"], explained["expl"]
    reasons = ex.top_reasons(expl)
    none_positive = np.flatnonzero((expl.values > 0).sum(axis=1) == 0)
    for i in none_positive:            # all of them, not the first 20
        assert reasons[i] == ["No elevated risk factors - this order scores below the "
                              "base rate."]


# ---------------------------------------------------------------------------------
# Buckets and determinism
# ---------------------------------------------------------------------------------

def test_reason_buckets_cover_every_order(explained):
    buckets = explained["explainer"].reason_buckets(explained["expl"])
    assert len(buckets) == len(explained["X"])
    assert set(buckets) <= set(FEATURE_GROUP.values())


def test_group_attributions_sum_to_the_total(explained):
    groups = explained["explainer"].group_attributions(explained["expl"])
    assert np.allclose(
        groups.sum(axis=1).to_numpy(), explained["expl"].values.sum(axis=1), atol=1e-9
    )


def test_explanations_are_deterministic(explained):
    again = explained["explainer"].explain(explained["X"])
    assert np.array_equal(again.values, explained["expl"].values)
    assert again.base_value == explained["expl"].base_value


def test_a_fresh_explainer_gives_identical_values(explained):
    fresh = ReasonExplainer(explained["bundle"]).explain(explained["X"])
    assert np.array_equal(fresh.values, explained["expl"].values)
