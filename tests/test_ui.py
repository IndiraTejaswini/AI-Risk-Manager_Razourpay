"""UI source guardrails."""

from pathlib import Path


def test_ui_no_mock_fallback():
    source = (Path(__file__).parents[1] / "ui" / "app.py").read_text(encoding="utf-8")

    assert '"risk":' not in source
    assert '"tier": "allow"' not in source
    assert "mock response" not in source.lower()
    assert "Backend unavailable" in source
