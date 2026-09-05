"""Webhook deduplication behavior at the HTTP boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import _state, app


def _worked_example() -> dict:
    docs = (Path(__file__).parents[2] / "docs" / "API.md").read_text(encoding="utf-8")
    match = re.search(r"### Request\s+```json\s*(.*?)\s*```", docs, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_webhook_redelivery():
    with TestClient(app) as client:
        first = client.post("/score", json=_worked_example(),
                            headers={"x-razorpay-event-id": "event-redelivery"})
        second = client.post("/score", json=_worked_example(),
                             headers={"x-razorpay-event-id": "event-redelivery"})

        assert first.status_code == second.status_code == 200
        assert first.content == second.content
        rows = _state["ingest"].connection.execute(
            "SELECT state FROM decision_log WHERE state = 'SCORED'"
        ).fetchall()
        assert len(rows) == 1


def test_missing_event_id_falls_back():
    with TestClient(app) as client:
        first = client.post("/score", json=_worked_example())
        second = client.post("/score", json=_worked_example())

        assert first.content == second.content
        row = _state["ingest"].connection.execute(
            "SELECT state_reason FROM decision_log WHERE state = 'SCORED'"
        ).fetchone()
        assert row == ("canonical body hash",)
