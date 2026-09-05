from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from responder.candidate import CandidateAction, MessageClass
from responder.channels.dry_run import DryRunChannel
from responder.channels.live import LiveChannel
from responder.channels.recorded import RecordedChannel
from responder.dispatch import dispatch
from responder.gates.types import GateResult
from responder.templates.registry import Tier
from responder.trust import Tagged, Tier as TrustTier


def _action() -> CandidateAction:
    return CandidateAction(
        "d1", Tier.CONFIRM, "address_quality_confirm", "v1", MessageClass.SERVICE,
        "Please verify your delivery address.", {"address": Tagged(1, TrustTier.MEASURED, "observed")},
        (GateResult("test", "v1", True),),
    )


def test_live_channel_refuses_olist(monkeypatch):
    monkeypatch.setenv("RESPONDER_LIVE", "1")
    with pytest.raises(RuntimeError, match="Olist"):
        LiveChannel("Olist")


def test_dry_run_is_absence_of_dispatch():
    dry = DryRunChannel()
    recorded = RecordedChannel()
    action = _action()
    dispatch(action, dry)
    dispatch(action, recorded)
    assert recorded.sent == [action]


def test_no_dry_run_branch_in_gate_chain():
    root = Path(__file__).parents[2]
    for path in (root / "responder" / "gates").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Name) and "dry_run" in node.id
            for node in ast.walk(tree)
        )
    for path in (root / "responder" / "templates").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.Name) and "dry_run" in node.id
            for node in ast.walk(tree)
        )
