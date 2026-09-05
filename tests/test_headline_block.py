"""
The headline block in README.md and ARCHITECTURE.md must match eval/TIER1_LOCK.json.

WHY THIS FILE EXISTS
--------------------------------------------------------------------------------------
Every stale-number defect this repo has found took the same form: a figure computed once,
transcribed by hand, and never re-derived. Two generated reports contradicted their own
tables. Four documentation files carried latency claims the measurement had moved past.
A README bullet quoted rejection rates of 10.0%/4.5% that turned out to be 14.5%/5.5%
when someone finally computed them.

The headline block is the most-read text in the repository, so it is the worst place for
that to happen again. It is generated from the lock, and this test fails if either copy
drifts - whether because the lock moved and the docs did not, or because someone edited
the block by hand.

Byte-for-byte, not "the numbers look right". A tolerance here would defeat the purpose:
the failure mode is a stale transcription, and a stale transcription is usually still
numerically plausible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_generator():
    """Import ``scripts/91_headline_block.py``, whose name is not an identifier."""
    path = REPO_ROOT / "scripts" / "91_headline_block.py"
    spec = importlib.util.spec_from_file_location("headline_block", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


headline = _load_generator()


@pytest.fixture(scope="module")
def expected() -> str:
    return headline.render(headline.load_lock())


def _extract(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find(headline.BEGIN)
    end = text.find(headline.END)
    assert start != -1, f"{path.name} has no HEADLINE:BEGIN marker"
    assert end != -1, f"{path.name} has no HEADLINE:END marker"
    assert start < end, f"{path.name} has the headline markers in the wrong order"
    return text[start:end + len(headline.END)]


@pytest.mark.parametrize("filename", ["README.md", "ARCHITECTURE.md"])
def test_headline_block_matches_the_lock(filename: str, expected: str) -> None:
    """
    The committed block is exactly what the generator produces from the current lock.

    A failure means one of two things, and the fix differs:
      * the lock moved and the documents were not regenerated - run
        `python scripts/91_headline_block.py`;
      * someone edited the block by hand - the same command reverts it, and the number
        they were trying to change belongs in the pipeline, not in the prose.
    """
    found = _extract(REPO_ROOT / filename)
    assert found == expected, (
        f"{filename}'s headline block does not match eval/TIER1_LOCK.json. "
        "Run `python scripts/91_headline_block.py` to regenerate it."
    )


def test_both_documents_carry_identical_blocks() -> None:
    """
    README and ARCHITECTURE must not diverge.

    Asserted directly rather than inferred from both matching the lock, so that the
    failure message says which of the two problems occurred when a future refactor
    renders them from different sources.
    """
    assert _extract(REPO_ROOT / "README.md") == _extract(REPO_ROOT / "ARCHITECTURE.md")


def test_block_does_not_repeat_the_superseded_framing(expected: str) -> None:
    """
    The block must not say the model cannot be shown to beat the base rate.

    `eval/model_report.md` §1 reached that conclusion by comparing a point estimate
    against a planning heuristic; `eval/significance.md` §5 corrects it - the paired
    bootstrap CI excludes zero and the permutation null gives p = 0.00060. This is a
    content assertion rather than a numeric one because the failure it guards against is
    a sentence, not a figure.
    """
    lowered = expected.lower()
    for banned in (
        "cannot demonstrate that this model beats",
        "cannot be shown to beat",
        "not resolvable",
        "does not beat the prevalence baseline",
    ):
        assert banned not in lowered, (
            f"the headline block repeats superseded framing: {banned!r}. "
            "eval/significance.md §5 overturns it."
        )


def test_generator_refuses_a_lock_of_another_version() -> None:
    """
    A lock of a different shape is refused, not rendered against the wrong field names.

    Same principle as the lock test's own `lock_version` assertion: silently rendering a
    v3 lock with v2 field lookups would produce a plausible block from the wrong data.
    """
    lock = headline.load_lock()
    lock["lock_version"] = 999
    path = REPO_ROOT / "eval" / "TIER1_LOCK.json"
    original = path.read_text(encoding="utf-8")
    import json

    try:
        path.write_text(json.dumps(lock), encoding="utf-8")
        with pytest.raises(headline.HeadlineError, match="lock_version"):
            headline.load_lock()
    finally:
        path.write_text(original, encoding="utf-8")
