"""Tier-1 evaluation lock and additive score-response contract tests."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import app

ROOT = Path(__file__).parents[1]
LOCK_PATH = ROOT / "eval" / "TIER1_LOCK.json"
GOLDEN_PATH = ROOT / "tests" / "golden" / "score_response.json"


def _assert_same(expected: Any, actual: Any, *, path: str, tolerance: float) -> None:
    if isinstance(expected, float):
        assert isinstance(actual, (int, float))
        assert actual == pytest.approx(expected, abs=tolerance), path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(expected) == set(actual), path
        for key, value in expected.items():
            _assert_same(value, actual[key], path=f"{path}.{key}", tolerance=tolerance)
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(expected) == len(actual), path
        for index, value in enumerate(expected):
            _assert_same(value, actual[index], path=f"{path}[{index}]", tolerance=tolerance)
    else:
        assert actual == expected, path


#: The three fields ``scripts/90_capture_tier1_lock.py`` writes for the record and
#: excludes from comparison, quoted from its own docstring:
#:
#:   ``metadata.captured_at``  wall clock; differs on every run by construction.
#:   ``metadata.git_sha``      records provenance, not identity - a file cannot contain
#:                             the hash of the commit that introduces it, so after
#:                             committing it necessarily names the parent.
#:   ``latency.*``             wall-clock measurements on an unquiesced desktop. Five
#:                             captures produced p50 between ~90ms and ~325ms on
#:                             unchanged code at the same SHA, and the instability is
#:                             between process invocations rather than within them.
#:
#: This test used to exclude only the first, so the generator's stated contract and the
#: assertion enforcing it disagreed.  Both other fields had already moved: HEAD advanced
#: past the recorded SHA, and eval/latency.md was regenerated one commit after the lock
#: was captured, taking the pooled p50 from 322.1ms to 125.3ms.  Re-capturing therefore
#: failed on two fields that the generator says are not evidence of anything changing.
#:
#: An intermittently-red lock is worse than no lock, because people learn to ignore it.
#: What the lock is for - the evaluation numbers and the three integrity digests - is
#: still compared exactly.
UNLOCKED_FIELDS = (
    ("metadata", "captured_at"),
    ("metadata", "git_sha"),
    ("latency",),
)


def _copy_unlocked(expected: dict, actual: dict) -> None:
    """Overwrite the excluded fields in ``actual`` with ``expected``'s values."""
    for path in UNLOCKED_FIELDS:
        target, source = actual, expected
        for key in path[:-1]:
            target, source = target[key], source[key]
        assert path[-1] in source and path[-1] in target, (
            f"{'.'.join(path)} is missing from the lock; UNLOCKED_FIELDS names a field "
            "that no longer exists, which would silently exclude nothing"
        )
        target[path[-1]] = source[path[-1]]


@pytest.mark.tier1
@pytest.mark.skipif(
    os.environ.get("RUN_TIER1_LOCK") != "1",
    reason="slow Tier-1 regeneration; set RUN_TIER1_LOCK=1 to run",
)
def test_tier1_numbers_unchanged():
    expected = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    subprocess.run(
        [sys.executable, "scripts/90_capture_tier1_lock.py"],
        cwd=ROOT,
        check=True,
    )
    actual = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    _copy_unlocked(expected, actual)
    assert (
        actual["integrity"]["elkan_thresholds_sha256"]
        == expected["integrity"]["elkan_thresholds_sha256"]
    ), "lock.integrity.elkan_thresholds_sha256"
    _assert_same(expected, actual, path="lock", tolerance=1e-9)


def _worked_example() -> dict[str, Any]:
    docs = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    match = re.search(r"### Request\s+```json\s*(.*?)\s*```", docs, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_score_response_contract():
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    with TestClient(app) as client:
        response = client.post("/score", json=_worked_example())

    assert response.status_code == 200
    actual = response.json()
    assert set(expected).issubset(actual)
    for key, value in expected.items():
        _assert_same(value, actual[key], path=f"response.{key}", tolerance=1e-12)
