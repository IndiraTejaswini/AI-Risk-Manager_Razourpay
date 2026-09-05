#!/usr/bin/env python3
"""Render the responder state diagram from responder.states."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from responder.states import TERMINAL, TRANSITIONS  # noqa: E402

OUT = ROOT / "responder" / "state_diagram.md"


def render() -> str:
    lines = ["```mermaid", "stateDiagram-v2"]
    for source in sorted(TRANSITIONS, key=lambda item: item.value):
        targets = TRANSITIONS[source]
        if not targets:
            lines.append(f"    {source.value} --> [{source.value}]")
        for target in sorted(targets, key=lambda item: item.value):
            lines.append(f"    {source.value} --> {target.value}")
    lines.extend(["```", "", f"Terminal states: {', '.join(sorted(s.value for s in TERMINAL))}", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {OUT}")
