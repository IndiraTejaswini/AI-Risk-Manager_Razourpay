"""
Step 7 serving assertions.

The named ones: the endpoint's feature construction is the same code path as training
and reproduces batch predictions exactly; assert_no_leakage applies at the request
boundary; column order is enforced; the cold-start path works; determinism.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

from api.contract import SUPPORTED_CURRENCY, ScoreRequest  # noqa: E402
from api.service import LeakyPayloadError, ScoringService  # noqa: E402
from data.loader import PRIMARY_LABEL  # noqa: E402
from models.calibration import PlattCalibrator  # noqa: E402
from models.train import FeatureContractError, predict, prepare_matrix  # noqa: E402

N_PARITY_ORDERS = 5


@pytest.fixture(scope="module")
def svc():
    return ScoringService()


def _payload_for(svc: ScoringService, order_row, *, with_customer=True) -> dict:
    """Build a Razorpay-shaped payload from a real Olist order."""
    items = svc.loader.load_table("items")
    items = items[items["order_id"] == order_row.order_id]
    pays = svc.loader.load_table("payments")
    pays = pays[pays["order_id"] == order_row.order_id]
    cust = svc.loader.load_table("customers")
    cust = cust[cust["customer_id"] == order_row.customer_id].iloc[0]

    ts = int(order_row.order_purchase_timestamp.timestamp())
    amount = int(round(float(pays["payment_value"].sum()) * 100))
    method = "boleto" if (pays["payment_type"] == "boleto").any() else "card"

    customer_block = (
        {"entity": {"entity": "customer",
                    "customer_reference": cust.customer_unique_id}}
        if with_customer else None
    )
    return {
        "entity": "event", "account_id": "acc_TEST", "event": "order.created",
        "contains": ["order", "payment"], "created_at": ts,
        "payload": {
            "order": {"entity": {
                "id": order_row.order_id, "entity": "order", "amount": amount,
                "currency": SUPPORTED_CURRENCY, "created_at": ts,
            }},
            "payment": {"entity": {
                "entity": "payment", "amount": amount,
                "currency": SUPPORTED_CURRENCY, "method": method,
                "emi_installments": int(pays["payment_installments"].max()),
            }},
            "customer": customer_block,
            "shipping_address": {"entity": {
                "zipcode": str(cust.customer_zip_code_prefix)
            }},
            "line_items": [
                {"product_id": r.product_id, "seller_id": r.seller_id,
                 "amount": int(round(r.price * 100)), "quantity": 1,
                 "shipping_amount": int(round(r.freight_value * 100))}
                for r in items.itertuples()
            ],
        },
    }


@pytest.fixture(scope="module")
def batch(svc):
    """Batch predictions over the test split, exactly as the pipeline computes them."""
    risk = svc.loader.risk_set().join(svc.loader.split_labelled()["split"], how="left")
    matrix = svc.builder.build(risk)
    X, _ = prepare_matrix(matrix, category_levels=svc.category_levels)
    te = (risk["split"] == "test").to_numpy()
    margin = predict(svc.bundle, X.loc[te], raw_score=True)
    return {
        "rows": risk[risk["split"] == "test"].reset_index(drop=True),
        "features": matrix.loc[te].reset_index(drop=True),
        "risk": svc.calibrator.predict(margin),
    }


# ---------------------------------------------------------------------------------
# The parity assertion
# ---------------------------------------------------------------------------------

def test_api_reproduces_batch_predictions_exactly(svc, batch):
    """
    The assertion the whole serving design exists to satisfy.

    If the endpoint reimplemented feature construction, this is where the drift would
    show: a plausible-looking score that differs from what the model was evaluated on.
    """
    for i in range(N_PARITY_ORDERS):
        row = batch["rows"].iloc[i]
        payload = _payload_for(svc, row)
        resp, _ = svc.score(ScoreRequest.model_validate(payload), payload)
        assert resp.risk == pytest.approx(float(batch["risk"][i]), abs=1e-12), (
            f"order {row.order_id} scored {resp.risk} through the API and "
            f"{batch['risk'][i]} in batch"
        )


def test_api_features_match_batch_features_exactly(svc, batch):
    """Parity at the feature level, so a mismatch localises."""
    row = batch["rows"].iloc[0]
    payload = _payload_for(svc, row)
    req = ScoreRequest.model_validate(payload)
    order_row, items, payments = svc._order_row(req)
    hist = svc.history[svc.history["order_id"] != order_row["order_id"].iloc[0]]
    population = pd.concat([hist, order_row], ignore_index=True)
    with svc._injected(items, payments, order_row["order_id"]):
        built = svc.builder.build(population).iloc[-1]

    expected = batch["features"].iloc[0]
    for col in expected.index:
        a, b = built[col], expected[col]
        if isinstance(b, float) and np.isnan(b):
            assert pd.isna(a), col
        else:
            assert a == b or a == pytest.approx(b), col


def test_no_second_feature_implementation_exists():
    """
    The service must not define its own feature maths. Anything that looks like a
    recomputation of a named feature is a train/serve skew waiting to happen.
    """
    import inspect

    from api import service as mod

    src = inspect.getsource(mod)
    for name in ("freight_ratio", "pincode_failure_rate_smoothed",
                 "cust_prior_failure_rate", "log_order_value", "product_volume_cm3"):
        assert f'"{name}"' not in src and f"'{name}'" not in src, (
            f"{name} appears in the serving module; features must come from "
            "FeatureBuilder only"
        )


# ---------------------------------------------------------------------------------
# Leakage gate at the request boundary
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field", ["order_status", "order_delivered_carrier_date", "review_score", "label_b"]
)
def test_payload_with_a_post_outcome_field_is_rejected(svc, batch, field):
    payload = _payload_for(svc, batch["rows"].iloc[0])
    payload["payload"]["order"]["entity"][field] = "anything"
    with pytest.raises(LeakyPayloadError, match=field):
        svc.assert_payload_clean(payload)


def test_leakage_gate_finds_nested_fields(svc, batch):
    payload = _payload_for(svc, batch["rows"].iloc[0])
    payload["payload"]["line_items"][0]["delivered_at"] = 123
    with pytest.raises(LeakyPayloadError, match="delivered_at"):
        svc.assert_payload_clean(payload)


def test_a_clean_payload_passes_the_gate(svc, batch):
    svc.assert_payload_clean(_payload_for(svc, batch["rows"].iloc[0]))


def test_non_brl_currency_is_rejected(svc, batch):
    payload = _payload_for(svc, batch["rows"].iloc[0])
    payload["payload"]["order"]["entity"]["currency"] = "INR"
    with pytest.raises(ValueError, match="not supported"):
        ScoreRequest.model_validate(payload)


def test_float_amount_is_rejected(svc, batch):
    """Amounts are integers in the smallest unit; a float is a rounding bug."""
    payload = _payload_for(svc, batch["rows"].iloc[0])
    payload["payload"]["order"]["entity"]["amount"] = 123.45
    with pytest.raises(ValueError):
        ScoreRequest.model_validate(payload)


# ---------------------------------------------------------------------------------
# Column order
# ---------------------------------------------------------------------------------

def test_permuted_feature_order_fails_loudly(svc, batch):
    """Step 3's contract, enforced on the serving path too."""
    row = batch["rows"].iloc[0]
    req = ScoreRequest.model_validate(_payload_for(svc, row))
    order_row, items, payments = svc._order_row(req)
    hist = svc.history[svc.history["order_id"] != order_row["order_id"].iloc[0]]
    population = pd.concat([hist, order_row], ignore_index=True)
    with svc._injected(items, payments, order_row["order_id"]):
        built = svc.builder.build(population).iloc[[-1]]
    X, _ = prepare_matrix(built, category_levels=svc.category_levels)
    X = X[list(svc.bundle.feature_names)]

    cols = list(X.columns)
    with pytest.raises(FeatureContractError, match="wrong ORDER"):
        predict(svc.bundle, X[[cols[1], cols[0]] + cols[2:]], raw_score=True)


# ---------------------------------------------------------------------------------
# Cold start - 97% of traffic
# ---------------------------------------------------------------------------------

def test_cold_start_payload_returns_a_valid_response(svc, batch):
    """
    No customer block at all: a guest checkout. This is the ordinary path on this
    panel, not an edge case.
    """
    payload = _payload_for(svc, batch["rows"].iloc[0], with_customer=False)
    resp, _ = svc.score(ScoreRequest.model_validate(payload), payload)
    assert 0.0 <= resp.risk <= 1.0
    assert resp.tier in ("allow", "confirm", "prepaid_only", "defer")
    assert 1 <= len(resp.reasons) <= 3
    assert resp.model_version
    assert resp.features_missing >= 0


def test_cold_start_with_an_unknown_pincode(svc, batch):
    payload = _payload_for(svc, batch["rows"].iloc[0], with_customer=False)
    payload["payload"]["shipping_address"]["entity"]["zipcode"] = "99999"
    resp, _ = svc.score(ScoreRequest.model_validate(payload), payload)
    assert 0.0 <= resp.risk <= 1.0
    assert resp.reasons


def test_order_with_no_line_items_still_scores(svc, batch):
    payload = _payload_for(svc, batch["rows"].iloc[0], with_customer=False)
    payload["payload"]["line_items"] = []
    resp, _ = svc.score(ScoreRequest.model_validate(payload), payload)
    assert 0.0 <= resp.risk <= 1.0
    assert resp.features_missing > 0


# ---------------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------------

def test_response_carries_a_model_version(svc, batch):
    """Non-negotiable, 11.1."""
    resp, _ = svc.score(
        ScoreRequest.model_validate(_payload_for(svc, batch["rows"].iloc[0]))
    )
    assert resp.model_version.startswith("rto-label_b-")


def test_model_version_tracks_the_artifacts(svc):
    """A retrain that changes the booster or the calibrator must change the string."""
    original = svc.model_version
    saved_a = svc.calibrator.a_
    try:
        svc.calibrator.a_ = saved_a + 1.0
        assert svc._version() != original
    finally:
        svc.calibrator.a_ = saved_a
    assert svc._version() == original


def test_threshold_used_is_finite_and_in_range(svc, batch):
    for i in range(3):
        resp, _ = svc.score(
            ScoreRequest.model_validate(_payload_for(svc, batch["rows"].iloc[i]))
        )
        assert 0.0 < resp.threshold_used <= 1.0


# ---------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------

def test_same_payload_same_response(svc, batch):
    payload = _payload_for(svc, batch["rows"].iloc[0])
    a, _ = svc.score(ScoreRequest.model_validate(payload), payload)
    b, _ = svc.score(ScoreRequest.model_validate(payload), payload)
    assert a.model_dump() == b.model_dump()


def test_scoring_does_not_mutate_the_source_caches(svc, batch):
    """The request's line items are injected for one call and must not persist."""
    before = len(svc.loader.load_table("items"))
    payload = _payload_for(svc, batch["rows"].iloc[0])
    svc.score(ScoreRequest.model_validate(payload), payload)
    assert len(svc.loader.load_table("items")) == before
