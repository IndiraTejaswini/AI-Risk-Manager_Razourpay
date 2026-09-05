"""State-machine and generated-diagram invariants."""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from responder.states import MAX_SEND_ATTEMPTS, TERMINAL, TRANSITIONS, State
from responder.store.db import connect
from responder.transitions import TransitionError, transition

ROOT = Path(__file__).parents[2]


def _seed(connection, state: State = State.SCORED, decision_id: str = "d1"):
    connection.execute(
        "INSERT INTO decision_log("
        "decision_id, order_id, account_id, model_version, policy_version, "
        "cost_constants_id, effectiveness_prior_id, gate_set_version, template_version, "
        "calibrated_p, tier, threshold_used, c_fp_impression, c_fp_triggered, c_fn, "
        "top_reason_class, reasons_json, features_missing, state, state_reason, actor, "
        "occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (decision_id, "o1", "a1", "m1", "p1", "c1", "e1", "g1", None, .1,
         "allow", .2, 1., 2., 3., "reason", "[]", 0, state.value, None,
         "scorer", "2026-01-01T00:00:00Z"),
    )
    connection.commit()


@pytest.mark.parametrize("source,target", [
    (source, target)
    for source, target in itertools.product(State, State)
    if target not in TRANSITIONS[source]
])
def test_illegal_transition_rejected(tmp_path, source, target):
    connection = connect(tmp_path / "state.sqlite3")
    _seed(connection, source)
    with pytest.raises(TransitionError):
        transition(connection, "d1", target, "test", "illegal")


@pytest.mark.parametrize("state", list(TERMINAL))
def test_terminal_is_terminal(tmp_path, state):
    connection = connect(tmp_path / "state.sqlite3")
    _seed(connection, state)
    with pytest.raises(TransitionError):
        transition(connection, "d1", State.SCORED, "test", "terminal")


def test_single_escalation(tmp_path):
    connection = connect(tmp_path / "state.sqlite3")
    _seed(connection, State.SENT)
    transition(connection, "d1", State.ESCALATED, "test", "first")
    transition(connection, "d1", State.SENT, "test", "retry")
    with pytest.raises(TransitionError, match="escalation"):
        transition(connection, "d1", State.ESCALATED, "test", "second")


def test_diagram_matches_declaration():
    diagram = ROOT / "responder" / "state_diagram.md"
    subprocess.run([sys.executable, "scripts/92_render_state_diagram.py"], cwd=ROOT, check=True)
    expected = diagram.read_text(encoding="utf-8")
    subprocess.run([sys.executable, "scripts/92_render_state_diagram.py"], cwd=ROOT, check=True)
    assert diagram.read_text(encoding="utf-8") == expected


def test_send_attempts_capped(tmp_path):
    connection = connect(tmp_path / "state.sqlite3")
    _seed(connection, State.QUEUED)
    for _ in range(MAX_SEND_ATTEMPTS):
        transition(connection, "d1", State.SENT, "test", "send")
        transition(connection, "d1", State.SEND_FAILED, "test", "failed")
        transition(connection, "d1", State.QUEUED, "test", "retry")
    transition(connection, "d1", State.SENT, "test", "fourth")
    state = connection.execute(
        "SELECT state FROM decision_log WHERE decision_id = ? ORDER BY seq DESC LIMIT 1",
        ("d1",),
    ).fetchone()[0]
    assert state == State.ABANDONED.value
