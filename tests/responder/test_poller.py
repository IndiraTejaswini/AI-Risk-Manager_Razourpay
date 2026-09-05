"""Outbox lease and terminal-failure behavior."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from responder.poller import Poller
from responder.states import State
from responder.store.db import connect


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _seed(connection, *, state=State.QUEUED, claimed_until=None, terminal=0):
    connection.execute(
        "INSERT INTO decision_log("
        "decision_id, order_id, account_id, model_version, policy_version, "
        "cost_constants_id, effectiveness_prior_id, gate_set_version, template_version, "
        "calibrated_p, tier, threshold_used, c_fp_impression, c_fp_triggered, c_fn, "
        "top_reason_class, reasons_json, features_missing, state, state_reason, actor, "
        "occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("d1", "o1", "a1", "m1", "p1", "c1", "e1", "g1", None, .1,
         "confirm", .2, 1., 2., 3., "address_quality", "[]", 0, state.value,
         None, "scorer", "2026-01-01T00:00:00+00:00"),
    )
    connection.execute(
        "INSERT INTO action_outbox(decision_id, created_at, claimed_until, terminal) "
        "VALUES (?, ?, ?, ?)",
        ("d1", "2026-01-01T00:00:00+00:00", claimed_until, terminal),
    )
    connection.commit()


def test_lease_prevents_double_claim(tmp_path):
    connection = connect(tmp_path / "poller.sqlite3")
    _seed(connection)
    poller = Poller(connection)
    assert poller.claim("d1", now=NOW)
    assert not poller.claim("d1", now=NOW)


def test_expired_lease_reclaimable(tmp_path):
    connection = connect(tmp_path / "poller.sqlite3")
    _seed(connection, claimed_until="2025-12-31T23:59:00+00:00")
    assert Poller(connection).claim("d1", now=NOW)


def test_terminal_rows_not_claimed(tmp_path):
    connection = connect(tmp_path / "poller.sqlite3")
    _seed(connection, terminal=1)
    assert not Poller(connection).claim("d1", now=NOW)


def test_abandoned_releases_lease(tmp_path):
    connection = connect(tmp_path / "poller.sqlite3")
    _seed(connection, claimed_until="2026-01-01T00:00:30+00:00")
    poller = Poller(connection)
    poller.abandon("d1")
    outbox = connection.execute(
        "SELECT claimed_until, terminal FROM action_outbox WHERE decision_id = 'd1'"
    ).fetchone()
    state = connection.execute(
        "SELECT state FROM decision_log WHERE decision_id = 'd1' ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert outbox == (None, 1)
    assert state == (State.ABANDONED.value,)
