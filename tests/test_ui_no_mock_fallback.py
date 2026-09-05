"""The UI must never hide an API disconnect behind fabricated data."""

from pathlib import Path


def test_ui_has_no_mock_fallback():
    source = (Path(__file__).parents[1] / "ui" / "app.py").read_text(encoding="utf-8").lower()
    assert "mock response" not in source
    assert "fallback" not in source
    assert "backend unavailable" in source
    assert "tier 2" in source
    assert "assumed" in source
    assert "measured" in source
