"""
Step 8 fairness assertions.

Pins every number in eval/fairness.md. These are characterisation tests: the findings
are properties of this panel and this policy, and if any of them moves it is either a
bug or a decision that has to be reflected in the report.

The two that matter most are the direction of the intervention-rate disparity and the
size of the pincode ablation. Both are stated in MODEL_CARD.md, and both would be
comfortable to get wrong quietly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from data.loader import PRIMARY_LABEL, OlistLoader  # noqa: E402
from features.builder import PINCODE_SMOOTHING, FeatureBuilder  # noqa: E402
from models.calibration import PlattCalibrator  # noqa: E402
from models.explain import ReasonExplainer  # noqa: E402
from models.train import predict, prepare_matrix, train  # noqa: E402
from policy.costs import ACTION_TIERS  # noqa: E402
from policy.elkan import apply_policy  # noqa: E402

CALIBRATION_WINDOW_DAYS = 30
DENSITY_CUTS = (10, 50)
THIN_ZIP_ORDERS = 10

# Committed in eval/fairness.md.
PANEL_INTERVENTION_RATE = 0.00173
SPARSE_N, SPARSE_POS, SPARSE_TREATED = 13_437, 107, 17
MEDIUM_N, MEDIUM_POS, MEDIUM_TREATED = 5_996, 47, 16
DENSE_N, DENSE_TREATED = 229, 1
SPARSE_SHARE_OF_POSITIVES = 0.695
SPARSE_SHARE_OF_TREATED = 0.500
ABLATION_DELTA = 0.0072
ABLATION_FRACTION_OF_MODEL = 0.42
STARVED_STATES = {"DF"}
TOP_GEOGRAPHY_FEATURE = "order_freight"


@pytest.fixture(scope="module")
def scored(loader: OlistLoader):
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    matrix = FeatureBuilder(loader=loader).build(risk)
    split = risk["split"].to_numpy()
    y_all = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y_all[tr]), X.loc[va], pd.Series(y_all[va]),
        target=PRIMARY_LABEL, population="risk_set", category_levels=levels,
    )
    s_va = predict(bundle, X.loc[va], raw_score=True)
    s_te = predict(bundle, X.loc[te], raw_score=True)
    ts = risk.loc[va, "order_purchase_timestamp"]
    fit = (ts >= loader.split_boundary
           - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)).to_numpy()
    p = PlattCalibrator().fit(s_va[fit], y_all[va][fit]).predict(s_te)

    ex = ReasonExplainer(bundle)
    res = apply_policy(p, matrix.loc[te], ex.reason_buckets(ex.explain(X.loc[te])))

    customers = loader.load_table(
        "customers", ["customer_id", "customer_zip_code_prefix", "customer_state"]
    )
    rows = risk.merge(customers, on="customer_id", how="left")
    train_counts = rows.loc[tr, "customer_zip_code_prefix"].value_counts()
    dens = pd.Series(rows.loc[te, "customer_zip_code_prefix"].to_numpy()).map(
        train_counts
    ).fillna(0).to_numpy()
    lo, hi = DENSITY_CUTS
    density = np.where(dens < lo, "sparse", np.where(dens < hi, "medium", "dense"))

    return {
        "y": y_all[te], "p": p, "tier": res.tier, "treated": res.tier != "allow",
        "density": density, "state": rows.loc[te, "customer_state"].to_numpy(),
        "feats": matrix.loc[te], "X": X, "matrix": matrix, "risk": risk,
        "splits": (tr, va, te), "y_all": y_all, "loader": loader, "levels": levels,
    }


# ---------------------------------------------------------------------------------
# 1. Buckets and the intervention-rate disparity
# ---------------------------------------------------------------------------------

def test_density_buckets_are_pinned(scored):
    d, y = scored["density"], scored["y"]
    assert int((d == "sparse").sum()) == SPARSE_N
    assert int((d == "medium").sum()) == MEDIUM_N
    assert int((d == "dense").sum()) == DENSE_N
    assert int(y[d == "sparse"].sum()) == SPARSE_POS
    assert int(y[d == "medium"].sum()) == MEDIUM_POS


def test_density_buckets_use_training_counts_only(scored):
    """
    A bucket boundary drawn with knowledge of the test window is not a proxy for
    anything. Density must come from the training split.
    """
    import inspect
    from pathlib import Path

    src = Path(inspect.getfile(inspect.currentframe())).parents[1] / "scripts" / "11_fairness.py"
    text = src.read_text(encoding="utf-8")
    assert 'rows.loc[tr, "customer_zip_code_prefix"].value_counts()' in text


def test_sparse_pincodes_are_treated_least(scored):
    """
    The finding, and it runs opposite to the concern §10 is written against. If this
    ever flips, the report's central fairness claim is wrong.
    """
    d, t = scored["density"], scored["treated"]
    sparse = float(t[d == "sparse"].mean())
    medium = float(t[d == "medium"].mean())
    assert sparse < medium, "sparse pincodes are no longer the least-treated bucket"
    assert int(t[d == "sparse"].sum()) == SPARSE_TREATED
    assert int(t[d == "medium"].sum()) == MEDIUM_TREATED
    assert int(t[d == "dense"].sum()) == DENSE_TREATED


def test_sparse_carries_more_failures_than_interventions(scored):
    """Under-protection, stated as the ratio the report quotes."""
    d, y, t = scored["density"], scored["y"], scored["treated"]
    share_pos = y[d == "sparse"].sum() / y.sum()
    share_treated = t[d == "sparse"].sum() / t.sum()
    assert share_pos == pytest.approx(SPARSE_SHARE_OF_POSITIVES, abs=0.01)
    assert share_treated == pytest.approx(SPARSE_SHARE_OF_TREATED, abs=0.01)
    assert share_pos > share_treated


def test_sparse_is_not_pushed_further_up_the_ladder(scored):
    """
    §7.5's claim is about ladder position, not just rate. Sparse must not be escalated
    more than medium.
    """
    d, tier = scored["density"], scored["tier"]

    def escalated(mask):
        sel = tier[mask]
        treated = sel[sel != "allow"]
        if len(treated) == 0:
            return 0.0
        return float(np.isin(treated, ["prepaid_only", "defer"]).mean())

    assert escalated(d == "sparse") <= escalated(d == "medium")


def test_nobody_is_blocked_in_any_bucket(scored):
    assert set(np.unique(scored["tier"])) <= {"allow", *ACTION_TIERS}


def test_states_above_panel_rate_with_zero_intervention(scored):
    """The state-level form of the same disparity."""
    y, t, st = scored["y"], scored["treated"], scored["state"]
    panel = float(y.mean())
    starved = set()
    for s in np.unique(st[pd.notna(st)]):
        m = st == s
        if m.sum() >= 300 and float(y[m].mean()) > panel and t[m].sum() == 0 \
                and y[m].sum() > 0:
            starved.add(s)
    assert starved == STARVED_STATES


def test_panel_intervention_rate_is_pinned(scored):
    assert float(scored["treated"].mean()) == pytest.approx(
        PANEL_INTERVENTION_RATE, abs=1e-4
    )


# ---------------------------------------------------------------------------------
# 2. Pincode ablation
# ---------------------------------------------------------------------------------

def test_pincode_ablation_resolves_and_is_pinned(scored):
    """
    §9 item 13. The pincode features carry roughly half the model, and the paired
    interval excludes zero — so they are not available to drop as a liability
    reduction.
    """
    from sklearn.metrics import average_precision_score

    from models.significance import paired_bootstrap_ap

    risk, loader = scored["risk"], scored["loader"]
    tr, va, te = scored["splits"]
    y_all, y = scored["y_all"], scored["y"]

    m2 = FeatureBuilder(
        loader=loader, groups=("order", "customer", "availability")
    ).build(risk)
    _, lv2 = prepare_matrix(m2.loc[tr])
    X2, _ = prepare_matrix(m2, category_levels=lv2)
    b2 = train(
        X2.loc[tr], pd.Series(y_all[tr]), X2.loc[va], pd.Series(y_all[va]),
        target=PRIMARY_LABEL, population="no pincode", category_levels=lv2,
        allow_missing_constraints=frozenset({"pincode_failure_rate_smoothed"}),
    )
    s2 = predict(b2, X2.loc[te], raw_score=True)

    X = scored["X"]
    bundle_scores = None
    for col in [None]:
        bundle_scores = None
    # Re-derive the full-model margin from the same fixture inputs.
    full_matrix = scored["matrix"]
    _, lv = prepare_matrix(full_matrix.loc[tr])
    Xf, _ = prepare_matrix(full_matrix, category_levels=lv)
    bf = train(
        Xf.loc[tr], pd.Series(y_all[tr]), Xf.loc[va], pd.Series(y_all[va]),
        target=PRIMARY_LABEL, population="full", category_levels=lv,
    )
    s1 = predict(bf, Xf.loc[te], raw_score=True)

    ap_full = average_precision_score(y, s1)
    ap_nopin = average_precision_score(y, s2)
    r = paired_bootstrap_ap(y, s1, s2, name="full - no pincode")

    assert r.observed == pytest.approx(ABLATION_DELTA, abs=5e-4)
    assert r.excludes_zero, "the pincode ablation no longer resolves"
    assert (ap_full - ap_nopin) / ap_full == pytest.approx(
        ABLATION_FRACTION_OF_MODEL, abs=0.03
    )


def test_dropped_pincode_features_are_exactly_the_pincode_group(scored):
    risk, loader = scored["risk"], scored["loader"]
    m2 = FeatureBuilder(
        loader=loader, groups=("order", "customer", "availability")
    ).build(risk)
    dropped = set(scored["matrix"].columns) - set(m2.columns)
    assert dropped == {
        "pincode_failure_rate_smoothed", "pincode_prior_orders",
        "pincode_prior_failures", "global_prior_failure_rate",
    }


# ---------------------------------------------------------------------------------
# 3. Smoothing
# ---------------------------------------------------------------------------------

def test_smoothing_constant_is_the_documented_one():
    assert PINCODE_SMOOTHING == 50.0


def test_thin_pincodes_cannot_reach_an_extreme_value(scored):
    """
    §4.5's mitigation, verified rather than asserted: with k prior orders the encoding
    can move at most k/(k+m) from the global prior. A blanket blacklist is
    arithmetically unreachable.
    """
    f = scored["feats"]
    enc = f["pincode_failure_rate_smoothed"].to_numpy()
    prior = f["global_prior_failure_rate"].to_numpy()
    k = f["pincode_prior_orders"].to_numpy()

    bound = k / (k + PINCODE_SMOOTHING)
    assert np.all(np.abs(enc - prior) <= bound + 1e-9), (
        "an encoded value exceeded the smoothing bound"
    )

    thin = k < THIN_ZIP_ORDERS
    assert thin.sum() > 0
    max_thin_dev = float(np.abs(enc[thin] - prior[thin]).max())
    assert max_thin_dev < THIN_ZIP_ORDERS / (THIN_ZIP_ORDERS + PINCODE_SMOOTHING)


def test_thin_pincodes_sit_near_the_global_prior(scored):
    f = scored["feats"]
    enc = f["pincode_failure_rate_smoothed"].to_numpy()
    prior = f["global_prior_failure_rate"].to_numpy()
    k = f["pincode_prior_orders"].to_numpy()
    thin = k < THIN_ZIP_ORDERS
    assert float(np.abs(enc[thin] - prior[thin]).mean()) < 0.005


# ---------------------------------------------------------------------------------
# 4. Protected attributes
# ---------------------------------------------------------------------------------

def test_no_feature_is_named_for_a_protected_attribute(scored):
    """
    The design claim MODEL_CARD makes, checked against the feature list. Olist carries
    no protected attribute, so this is the strongest test available - and that is
    exactly what the report says about it.
    """
    banned = ("name", "gender", "sex", "religion", "language", "race", "ethnic",
              "age", "birth", "marital")
    for col in scored["matrix"].columns:
        low = col.lower()
        for b in banned:
            assert b not in low, f"{col} looks like a protected attribute"


def test_freight_carries_more_geography_than_the_named_pincode_features(scored):
    """
    The finding: the strongest geography proxy is not called a geography feature, which
    is why dropping the pincode group would not produce a geography-free model.
    """
    from sklearn.metrics import mutual_info_score

    f, st = scored["feats"], scored["state"]
    codes = pd.Categorical(pd.Series(st).fillna("__na__")).codes

    def mi(col):
        v = f[col]
        if v.dtype.kind in "fiu":
            b = pd.qcut(pd.Series(v.to_numpy(dtype=float)), 10,
                        duplicates="drop", labels=False).fillna(-1).astype(int)
        else:
            b = pd.Categorical(v.astype("object").fillna("__na__")).codes
        return mutual_info_score(b, codes)

    freight = mi(TOP_GEOGRAPHY_FEATURE)
    named = max(mi(c) for c in ("pincode_failure_rate_smoothed", "pincode_prior_orders",
                                "pincode_prior_failures", "global_prior_failure_rate"))
    assert freight > named, (
        "order_freight is no longer the strongest geography carrier; "
        "eval/fairness.md §4's central point needs revisiting"
    )
