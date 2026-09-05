"""Database-level guarantees for the responder substrate."""

from __future__ import annotations

import sqlite3

import pytest

from responder.store.db import connect


def _inbox_row(event_key: str = "event-1") -> tuple[str, str, str, str]:
    return event_key, "account-1", "2026-09-05T00:00:00Z", "{}"


def _decision_row(decision_id: str = "decision-1") -> tuple:
    return (
        decision_id,
        "order-1",
        "account-1",
        "model-1",
        "policy-1",
        "cost-1",
        "effectiveness-1",
        "gates-1",
        None,
        0.1,
        "allow",
        0.2,
        0.3,
        0.4,
        1.0,
        "category",
        "[]",
        0,
        "scored",
        None,
        "scorer",
        "2026-09-05T00:00:00Z",
    )


def test_decision_log_rejects_update(tmp_path):
    connection = connect(tmp_path / "responder.sqlite3")
    try:
        connection.execute(
            "INSERT INTO decision_log("
            "decision_id, order_id, account_id, model_version, policy_version, "
            "cost_constants_id, effectiveness_prior_id, gate_set_version, "
            "template_version, calibrated_p, tier, threshold_used, c_fp_impression, "
            "c_fp_triggered, c_fn, top_reason_class, reasons_json, features_missing, "
            "state, state_reason, actor, occurred_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _decision_row(),
        )
        connection.commit()
    finally:
        connection.close()

    connection = connect(tmp_path / "responder.sqlite3")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE decision_log SET state = 'sent' WHERE seq = 1")
    finally:
        connection.close()


def test_decision_log_rejects_delete(tmp_path):
    connection = connect(tmp_path / "responder.sqlite3")
    try:
        connection.execute(
            "INSERT INTO decision_log("
            "decision_id, order_id, account_id, model_version, policy_version, "
            "cost_constants_id, effectiveness_prior_id, gate_set_version, "
            "template_version, calibrated_p, tier, threshold_used, c_fp_impression, "
            "c_fp_triggered, c_fn, top_reason_class, reasons_json, features_missing, "
            "state, state_reason, actor, occurred_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _decision_row(),
        )
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM decision_log WHERE seq = 1")
    finally:
        connection.close()


def test_inbox_unique_event_key(tmp_path):
    connection = connect(tmp_path / "responder.sqlite3")
    try:
        connection.execute(
            "INSERT INTO inbox(event_key, account_id, received_at, response_json) "
            "VALUES (?, ?, ?, ?)",
            _inbox_row(),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO inbox(event_key, account_id, received_at, response_json) "
                "VALUES (?, ?, ?, ?)",
                _inbox_row(),
            )
    finally:
        connection.close()


def test_schema_creation_is_idempotent(tmp_path):
    path = tmp_path / "responder.sqlite3"
    first = connect(path)
    first.close()
    second = connect(path)
    try:
        tables = {
            row[0]
            for row in second.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"inbox", "decision_log", "action_outbox"} <= tables
    finally:
        second.close()
