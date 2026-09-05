"""Customer response and sweeper behavior."""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from api.main import app
from responder.states import State
from responder.store.db import connect
from responder.sweeper import Sweeper


def _seed(connection, state=State.SENT, occurred_at="2026-01-01T00:00:00+00:00"):
    values = (
        "d1", "o1", "a1", "m1", "p1", "c1", "e1", "g1", None, .1,
        "confirm", .2, 1., 2., 3., "address_quality", "[]", 0, state.value,
        None, "scorer", occurred_at,
    )
    connection.execute(
        "INSERT INTO decision_log("
        "decision_id, order_id, account_id, model_version, policy_version, "
        "cost_constants_id, effectiveness_prior_id, gate_set_version, template_version, "
        "calibrated_p, tier, threshold_used, c_fp_impression, c_fp_triggered, c_fn, "
        "top_reason_class, reasons_json, features_missing, state, state_reason, actor, occurred_at) "
        "VALUES (" + ",".join("?" for _ in values) + ")",
        values,
    )
    connection.commit()


def test_respond_idempotent(tmp_path, monkeypatch):
    connection = connect(tmp_path / "respond.sqlite3")
    _seed(connection)
    monkeypatch.setattr("api.main.ingest", lambda: type("I", (), {"connection": connection})())
    with TestClient(app) as client:
        first = client.post("/respond", json={"decision_id": "d1", "response": "confirmed"})
        second = client.post("/respond", json={"decision_id": "d1", "response": "confirmed"})
    assert first.json() == second.json() == {"decision_id": "d1", "state": "CONFIRMED"}
    assert connection.execute(
        "SELECT COUNT(*) FROM decision_log WHERE decision_id = 'd1'"
    ).fetchone() == (2,)


def test_respond_rejects_unknown_decision(tmp_path, monkeypatch):
    connection = connect(tmp_path / "respond.sqlite3")
    monkeypatch.setattr("api.main.ingest", lambda: type("I", (), {"connection": connection})())
    with TestClient(app) as client:
        response = client.post("/respond", json={"decision_id": "missing", "response": "yes"})
    assert response.status_code == 404


def test_sweeper_escalates_once_only(tmp_path):
    connection = connect(tmp_path / "sweep.sqlite3")
    _seed(connection)
    sweeper = Sweeper(connection, dispatch_cutoff=datetime(2026, 1, 3, tzinfo=timezone.utc))
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert sweeper.sweep(now=now) == [("d1", State.ESCALATED)]
    assert sweeper.sweep(now=now) == []


def test_sweeper_respects_dispatch_cutoff(tmp_path):
    connection = connect(tmp_path / "sweep.sqlite3")
    _seed(connection)
    sweeper = Sweeper(connection, dispatch_cutoff=datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
    assert sweeper.sweep(now=datetime(2026, 1, 2, tzinfo=timezone.utc)) == [
        ("d1", State.NO_RESPONSE)
    ]
