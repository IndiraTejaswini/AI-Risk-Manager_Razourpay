"""
Step 5 policy assertions.

The named ones: determinism, the policy never sees labels when selecting thresholds,
every assumed constant carries a tag, and the cost model runs on the risk set row-for-row
with Step 4's calibrated output.
"""

from __future__ import annotations

import inspect

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

from data.loader import PRIMARY_LABEL, OlistLoader
from features.builder import FeatureBuilder
from models.calibration import PlattCalibrator
from models.train import predict, prepare_matrix, train
from policy import costs as costs_mod
from policy import elkan as elkan_mod
from policy.constants import ASSUMPTIONS, CURRENCY
from policy.costs import ACTION_TIERS, TIERS, c_rto, ltv_multiplier, tier_costs
from models.explain import ReasonExplainer
from policy.effectiveness import EFFECTIVENESS, REASONS
from policy.elkan import apply_policy

CALIBRATION_WINDOW_DAYS = 30

# Committed in eval/policy.md.  The policy buckets reasons by SHAP attribution as of
# step 6; the pre-SHAP stand-in treated 24 (eval/reasons.md section 7).
TREATED_ORDERS = 34
FLAT_MODEL_TREATED = 0
TEST_ROWS = 19_662


# ---------------------------------------------------------------------------------
# Assumption registry
# ---------------------------------------------------------------------------------

def test_every_constant_has_a_valid_basis_tag():
    valid = {"OBSERVED", "ASSUMPTION", "GUESS"}
    for a in ASSUMPTIONS.values():
        assert a.basis in valid, f"{a.name} has basis {a.basis!r}"
        assert a.rationale.strip(), f"{a.name} has no rationale"
        assert a.unit.strip()


def test_the_fatigue_allowance_is_labelled_a_guess():
    """Section 7.2 requires it be included AND labelled. Both halves matter."""
    assert ASSUMPTIONS["fatigue_allowance"].basis == "GUESS"
    assert ASSUMPTIONS["fatigue_allowance"].value > 0


def test_only_price_and_freight_are_observed():
    observed = {a.name for a in ASSUMPTIONS.values() if a.basis == "OBSERVED"}
    assert observed == {"order_value", "forward_freight"}


def test_currency_is_brl_and_no_conversion_is_applied():
    """
    Relabelling BRL as INR would make every cost figure wrong by the exchange rate.
    The reference rate exists but must not be wired into any cost function.
    """
    assert CURRENCY == "BRL"
    assert "brl_to_inr" in " ".join(ASSUMPTIONS)
    source = inspect.getsource(costs_mod) + inspect.getsource(elkan_mod)
    assert "brl_to_inr" not in source.lower(), "conversion rate must not enter a cost"


# ---------------------------------------------------------------------------------
# Cost model shape
# ---------------------------------------------------------------------------------

def test_c_rto_grows_with_value_and_freight():
    v = np.array([100.0, 100.0, 500.0])
    f = np.array([10.0, 40.0, 10.0])
    c = c_rto(v, f)
    assert c[1] > c[0]      # more freight
    assert c[2] > c[0]      # more blocked inventory


def test_prepaid_only_has_zero_impression_cost():
    """
    Section 7.2's central point: prepaid-only is a checkout configuration, not a send,
    so its entire cost is conditional and its threshold moves accordingly.
    """
    f = pd.DataFrame({"order_value": [200.0], "cust_prior_orders": [0.0]})
    t = tier_costs(f)
    assert t["prepaid_only"]["impression"][0] == 0.0
    assert t["confirm"]["impression"][0] > 0.0
    assert t["defer"]["impression"][0] > 0.0
    assert t["prepaid_only"]["triggered"][0] > 0.0


def test_impression_is_flat_and_triggered_scales_with_value():
    f = pd.DataFrame({"order_value": [100.0, 1000.0], "cust_prior_orders": [0.0, 0.0]})
    t = tier_costs(f)
    assert t["confirm"]["impression"][0] == t["confirm"]["impression"][1]
    assert t["confirm"]["triggered"][1] > t["confirm"]["triggered"][0]


def test_ltv_multiplier_rises_with_history_and_is_capped():
    m = ltv_multiplier(np.array([0.0, 1.0, 3.0, 100.0]))
    assert m[0] == 1.0
    assert m[1] > m[0]
    assert m[3] == ASSUMPTIONS["ltv_multiplier_cap"].value


def test_effectiveness_matrix_is_complete():
    for r in REASONS:
        for t in ACTION_TIERS:
            v = EFFECTIVENESS[r][t]
            assert 0.0 <= v <= 1.0


def test_prepaid_beats_confirm_for_a_customer_history_reason():
    """Section 7.3: a serial refuser is not fixed by asking; it is fixed by removing COD."""
    assert (EFFECTIVENESS["customer_history"]["prepaid_only"]
            > EFFECTIVENESS["customer_history"]["confirm"])


# ---------------------------------------------------------------------------------
# Elkan
# ---------------------------------------------------------------------------------

def test_elkan_threshold_matches_the_closed_form():
    """p* = c_FP / (c_FP + c_FN), verified against the expected-cost crossing point."""
    f = pd.DataFrame({
        "order_value": [300.0], "order_freight": [40.0], "cust_prior_orders": [0.0],
        "freight_ratio": [0.12], "pincode_failure_rate_smoothed": [0.01],
        "cust_prior_failure_rate": [0.01], "n_missing_features": [0.0],
    })
    reasons = np.array(["order_composition"])
    r = apply_policy(np.array([0.5]), f, reasons)
    for tier in ACTION_TIERS:
        star = r.thresholds[tier][0]
        if not np.isfinite(star):
            continue
        expected = r.c_fp[tier][0] / (r.c_fp[tier][0] + r.c_fn[tier][0])
        assert star == pytest.approx(expected)
        # At p*, acting and allowing must cost the same.
        at = apply_policy(np.array([star]), f, reasons)
        assert at.expected_cost[tier][0] == pytest.approx(
            at.expected_cost["allow"][0], rel=1e-9
        )


def test_threshold_functions_take_no_labels():
    """
    Structural, not procedural: there is no argument to pass a label to.
    """
    sig = inspect.signature(apply_policy)
    assert "y" not in sig.parameters
    assert "label" not in sig.parameters
    src = inspect.getsource(apply_policy)
    assert "label_b" not in src and "y_true" not in src


def test_the_policy_stands_down_when_the_intervention_never_works():
    """
    The pessimistic end of section 7.3's sweep. At effectiveness 0 every c_FN is
    -impression, so acting costs more than the failure it prevents at any risk. Every
    threshold must go to infinity and nothing may fire - including for an order the
    model is certain about.
    """
    f = pd.DataFrame({
        "order_value": [400.0], "order_freight": [60.0], "cust_prior_orders": [0.0],
    })
    r = apply_policy(np.array([0.99]), f, np.array(["order_composition"]),
                     effectiveness_scale=0.0)
    for tier in ACTION_TIERS:
        assert np.isinf(r.thresholds[tier][0]), tier
        assert not r.fires[tier][0], tier
    assert r.tier[0] == "allow"
    assert not r.any_fires


def test_a_tier_whose_cost_exceeds_the_loss_it_prevents_never_fires():
    """c_FN <= 0 must produce an infinite threshold rather than a nonsensical ratio."""
    f = pd.DataFrame({
        "order_value": [400.0], "order_freight": [60.0], "cust_prior_orders": [0.0],
    })
    # Scale effectiveness low enough that defer's manual-review cost dominates.
    r = apply_policy(np.array([0.99]), f, np.array(["order_composition"]),
                     effectiveness_scale=0.01)
    assert np.isinf(r.thresholds["defer"][0])
    assert not r.fires["defer"][0]


def test_nobody_is_ever_blocked():
    """Section 7.5 - the worst outcome for a false positive is being asked to prepay."""
    assert "block" not in TIERS and "reject" not in TIERS
    assert set(TIERS) == {"allow", "confirm", "prepaid_only", "defer"}


def test_higher_risk_never_reduces_the_chosen_tier_severity():
    f = pd.DataFrame({
        "order_value": [400.0] * 3, "order_freight": [60.0] * 3,
        "cust_prior_orders": [0.0] * 3,
    })
    reasons = np.array(["pincode"] * 3)
    rank = {t: i for i, t in enumerate(TIERS)}
    r = apply_policy(np.array([0.001, 0.05, 0.5]), f, reasons)
    sev = [rank[t] for t in r.tier]
    assert sev == sorted(sev)


# ---------------------------------------------------------------------------------
# The real pipeline
# ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored(loader: OlistLoader):
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    matrix = FeatureBuilder().build(risk)
    split = risk["split"].to_numpy()
    y = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    b = train(X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
              target=PRIMARY_LABEL, population="risk_set", category_levels=levels)
    s_va = predict(b, X.loc[va], raw_score=True)
    s_te = predict(b, X.loc[te], raw_score=True)
    ts = risk.loc[va, "order_purchase_timestamp"]
    fit = (ts >= loader.split_boundary - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)).to_numpy()
    p = PlattCalibrator().fit(s_va[fit], y[va][fit]).predict(s_te)
    explainer = ReasonExplainer(b)
    reasons = explainer.reason_buckets(explainer.explain(X.loc[te]))
    return {
        "risk": risk, "loader": loader, "p": p, "feats": matrix.loc[te],
        "y": y[te], "reasons": reasons, "test_rows": risk[risk["split"] == "test"],
    }


def test_population_is_the_risk_set_not_the_matured_set(scored):
    """
    The 767-order item-join artifact lives in the matured set. It must not reach the
    cost table.
    """
    loader = scored["loader"]
    matured_only = set(loader.labelled()["order_id"]) - set(loader.risk_set()["order_id"])
    assert set(scored["test_rows"]["order_id"]).isdisjoint(matured_only)

    item_ids = set(loader.load_table("items", ["order_id"])["order_id"])
    n_missing = int((~scored["test_rows"]["order_id"].isin(item_ids)).sum())
    assert n_missing <= 1, "the item-join artifact has reached the policy population"


def test_row_alignment_with_the_calibrated_output(scored):
    assert len(scored["p"]) == len(scored["feats"]) == len(scored["y"]) == TEST_ROWS


def test_policy_is_deterministic(scored):
    a = apply_policy(scored["p"], scored["feats"], scored["reasons"])
    b = apply_policy(scored["p"], scored["feats"], scored["reasons"])
    assert np.array_equal(a.tier, b.tier)
    for tier in ACTION_TIERS:
        assert np.array_equal(a.thresholds[tier], b.thresholds[tier])


def test_treated_count_is_pinned(scored):
    r = apply_policy(scored["p"], scored["feats"], scored["reasons"])
    assert int((r.tier != "allow").sum()) == TREATED_ORDERS


def test_a_flat_cost_model_treats_far_fewer_orders(scored):
    """
    Section 7.1's argument, as a test. A single median-derived threshold cannot see that
    a high-freight, low-margin order has a different economic case from a low-freight,
    high-margin one at the same risk, so it misses most of the orders where intervening
    pays.

    On this build the gap is total rather than partial: the calibrated ceiling fell to
    6.86% after the PIT tie-block fix (see
    test_every_median_threshold_sits_above_the_ceiling), which pushed every tier's flat
    threshold out of reach for every order. The nested model still finds
    TREATED_ORDERS because its threshold is per-order, not per-tier.
    """
    from policy.effectiveness import effectiveness_vector

    feats, p, reasons = scored["feats"], scored["p"], scored["reasons"]
    value = np.nan_to_num(feats["order_value"].to_numpy(dtype=float), nan=0.0)
    freight = np.nan_to_num(feats["order_freight"].to_numpy(dtype=float), nan=0.0)
    rto = c_rto(value, freight)
    costs = tier_costs(feats)

    treated = 0
    for tier in ACTION_TIERS:
        e = effectiveness_vector(reasons, tier)
        fp = float(np.median(costs[tier]["c_fp"]))
        fn = float(np.median(e) * np.median(rto) - np.median(costs[tier]["impression"]))
        star = fp / (fp + fn) if fn > 0 else np.inf
        treated += int((p >= star).sum())
    assert treated == FLAT_MODEL_TREATED
    assert treated < TREATED_ORDERS, (
        "the flat cost model must treat strictly fewer orders than the nested one; "
        "that gap is section 7.1's argument"
    )


def test_every_median_threshold_sits_above_the_ceiling(scored):
    """
    The feasibility finding, and it changed direction from an earlier build.

    Before the tie-block PIT fix (features/pit.py; see eval/feature_expansion.md
    section 3.2 and tests/test_pit_exactness.py), the calibrated ceiling sat at 9.43%
    and `confirm`'s median threshold (7.40%) cleared it - one tier was reachable for
    the median order.  The fix corrected 534 orders' `cust_days_since_prior_order` from
    a same-instant-order artifact (0.0 days) to the true NaN, which moved the model
    enough that the calibrated ceiling fell to 6.86% - *below* `confirm`'s median
    threshold, which itself barely moved.  All three tiers are now out of reach for the
    median order, not two of three.

    Pinned as an inequality on the ceiling, not as fixed threshold values, so the test
    keeps making the right claim if the model is retrained again rather than silently
    describing a feasibility story that no longer holds.
    """
    r = apply_policy(scored["p"], scored["feats"], scored["reasons"])
    ceiling = float(scored["p"].max())
    medians = {
        tier: float(np.median(r.thresholds[tier][np.isfinite(r.thresholds[tier])]))
        for tier in ACTION_TIERS
    }
    for tier in ACTION_TIERS:
        assert medians[tier] > ceiling, (
            f"{tier}'s median threshold ({medians[tier]:.4f}) no longer exceeds the "
            f"calibrated ceiling ({ceiling:.4f}) - the feasibility finding has changed "
            "again and eval/policy.md needs to be re-read, not this assertion patched"
        )
    # defer stays the most expensive tier by a wide margin - order, not magnitude, is
    # the part of this claim durable across a retrain.
    assert medians["defer"] > medians["prepaid_only"] > medians["confirm"] > ceiling
