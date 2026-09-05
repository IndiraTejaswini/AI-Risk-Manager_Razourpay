"""
Parse data/COLUMN_WHITELIST.md into an enforceable structure.

The document is the stated rule and this makes it the *enforced* one.  Without this,
COLUMN_WHITELIST.md is decoration that drifts from the code the moment someone edits one
and not the other; tests/test_feature_builder.py asserts the two agree.
"""

from __future__ import annotations

import re
from pathlib import Path

WHITELIST_PATH = Path(__file__).resolve().parent / "COLUMN_WHITELIST.md"

_SECTION = "## Admissible at checkout"
_BACKTICKED = re.compile(r"`([^`]+)`")


def parse_column_whitelist(path: Path | None = None) -> dict[str, set[str]]:
    """
    Return ``{table: {column, ...}}`` from the "Admissible at checkout" table.

    Only that section is read.  The label-construction and post-checkout tables earlier
    in the document are exclusions, not a whitelist, and must not be picked up here.
    """
    path = path or WHITELIST_PATH
    text = path.read_text(encoding="utf-8")

    try:
        body = text.split(_SECTION, 1)[1]
    except IndexError:
        raise ValueError(f"{path} has no '{_SECTION}' section") from None

    out: dict[str, set[str]] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            # Stop at the first non-table line after the table has started.
            if out and not line:
                continue
            if out:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        table = _BACKTICKED.findall(cells[0])
        if len(table) != 1:
            continue  # header or separator row
        columns = _BACKTICKED.findall(cells[1])
        if columns:
            out[table[0]] = set(columns)

    if not out:
        raise ValueError(f"parsed no whitelist entries from {path}")
    return out
