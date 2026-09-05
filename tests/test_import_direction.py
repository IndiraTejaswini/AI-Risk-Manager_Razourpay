"""Enforce the one-way dependency boundary around the frozen detector."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FROZEN_PACKAGES = ("data", "features", "models", "policy")
FORBIDDEN_ROOTS = {"responder", "simulation"}


def _import_root(module: str | None) -> str | None:
    if not module:
        return None
    return module.split(".", 1)[0]


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules = (node.module,)
        else:
            continue
        for module in modules:
            if _import_root(module) in FORBIDDEN_ROOTS:
                found.append(module or "")
    return found


@pytest.mark.parametrize("package", FROZEN_PACKAGES)
def test_frozen_detector_does_not_import_responder_or_simulation(package: str):
    package_dir = ROOT / package
    violations = {
        str(path.relative_to(ROOT)): imports
        for path in package_dir.rglob("*.py")
        if (imports := _forbidden_imports(path))
    }
    assert not violations, f"forbidden detector imports: {violations}"
