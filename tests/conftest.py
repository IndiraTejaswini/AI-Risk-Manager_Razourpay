"""Shared fixtures.

The loader reads ~100k rows, so it is built once per session and shared.  Every test
that mutates a frame works on its own copy; the loader's accessors all return copies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import OlistLoader  # noqa: E402


@pytest.fixture(scope="session")
def loader() -> OlistLoader:
    return OlistLoader()


@pytest.fixture(scope="session")
def labelled(loader: OlistLoader):
    return loader.labelled()


@pytest.fixture(scope="session")
def risk_set(loader: OlistLoader):
    return loader.risk_set()
