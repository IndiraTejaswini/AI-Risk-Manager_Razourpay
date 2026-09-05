"""Replay uses the complete test-window denominator and is deterministic."""

from responder.replay import replay, render_report


def test_replay_denominator():
    result = replay()
    assert sum(result["statuses"].values()) == result["rows"]


def test_replay_deterministic():
    assert render_report(replay()) == render_report(replay())
