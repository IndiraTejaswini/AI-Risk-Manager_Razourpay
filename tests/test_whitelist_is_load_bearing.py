"""The column whitelist, rather than the request blocklist, is the leakage guarantee."""

from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api import service as service_module
from api.contract import FORBIDDEN_PAYLOAD_FIELDS


@pytest.fixture(scope="module")
def client():
    with TestClient(api_main.app) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return {
        "entity": "event",
        "account_id": "acc_WHITELIST_TEST",
        "event": "order.created",
        "contains": ["order", "payment"],
        "created_at": 1_500_000_000,
        "payload": {
            "order": {
                "entity": {
                    "id": "order_whitelist_test",
                    "entity": "order",
                    "amount": 10_000,
                    "currency": "BRL",
                    "created_at": 1_500_000_000,
                }
            },
            "payment": {
                "entity": {
                    "entity": "payment",
                    "amount": 10_000,
                    "currency": "BRL",
                    "method": "card",
                }
            },
            "shipping_address": {"entity": {"zipcode": "14409"}},
            "line_items": [
                {
                    "product_id": "product_whitelist_test",
                    "seller_id": "seller_whitelist_test",
                    "amount": 10_000,
                    "quantity": 1,
                    "shipping_amount": 9_000,
                }
            ],
        },
    }


def _with_forbidden_fields(payload: dict) -> dict:
    enriched = deepcopy(payload)
    values = {
        "order_status": "canceled",
        "status": "canceled",
        "order_delivered_carrier_date": "1970-01-01T00:00:00Z",
        "order_delivered_customer_date": "1970-01-01T00:00:00Z",
        "delivered_at": "1970-01-01T00:00:00Z",
        "shipped_at": "1970-01-01T00:00:00Z",
        "order_approved_at": "1970-01-01T00:00:00Z",
        "order_estimated_delivery_date": "1970-01-01T00:00:00Z",
        "review_score": 1,
        "label": True,
        "label_a": True,
        "label_b": True,
    }
    assert set(values) == set(FORBIDDEN_PAYLOAD_FIELDS)
    enriched["payload"]["order"]["entity"].update(values)
    return enriched


def test_whitelist_is_load_bearing_when_edge_blocklist_is_disabled(
    client: TestClient, payload: dict, monkeypatch: pytest.MonkeyPatch
):
    """Every currently enumerated edge field is ignored without changing the score."""
    monkeypatch.setattr(service_module, "FORBIDDEN_PAYLOAD_FIELDS", frozenset())
    enriched = _with_forbidden_fields(payload)

    with_forbidden = client.post("/score", json=enriched)
    without_forbidden = client.post("/score", json=payload)

    assert with_forbidden.status_code == 200
    assert without_forbidden.status_code == 200
    assert with_forbidden.json()["risk"] == pytest.approx(
        without_forbidden.json()["risk"], abs=1e-12
    )


def test_request_blocklist_still_returns_422(client: TestClient, payload: dict):
    enriched = _with_forbidden_fields(payload)

    response = client.post("/score", json=enriched)

    assert response.status_code == 422
    assert response.json()["error"] == "post_outcome_field_in_payload"
