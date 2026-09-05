#!/usr/bin/env python3
"""
Task D0: capture the Tier-1 lock.

Writes ``eval/TIER1_LOCK.json`` - the frozen record of what this repo's evaluation
pipeline produces at the current commit, so that a later change which moves a headline
number fails a test instead of quietly shipping.

--------------------------------------------------------------------------------------
WHAT THIS SCRIPT IS ALLOWED TO DO
--------------------------------------------------------------------------------------

Nothing except read and serialise.  It modifies no training script, no feature
definition, no calibrator, no cost constant and no evaluation generator; it imports the
same modules the pipeline imports and calls them with the same arguments
``scripts/08_policy.py`` does.  If capturing the lock could change the thing being
locked, the lock would be worthless.

--------------------------------------------------------------------------------------
WHERE EVERY NUMBER COMES FROM
--------------------------------------------------------------------------------------

Two sources, and the rule for choosing between them is fixed:

1.  **The committed reports in ``eval/``**, parsed at run time.  This is the default.
    No figure is transcribed into this file from ARCHITECTURE.md, README.md,
    MODEL_CARD.md or a task description - several documents in this repo have been
    confirmed to carry stale numbers, and so has the *prose* of at least one generated
    report (``scripts/08_policy.py`` hard-codes "123" and "500" in a sentence whose own
    generated table says 144 and 629).  **Only generated tables and generated f-string
    lines are parsed; never hand-written prose.**

2.  **A live pipeline run**, for the two things ``eval/`` does not contain: the per-order
    Elkan threshold state (needed for ``integrity.elkan_thresholds_sha256``) and the
    model-version fingerprint from ``api/service.py``.

Where a number exists in both places it is parsed from ``eval/`` *and* cross-checked
against the live run.  A disagreement raises - it means the committed reports no longer
describe the code, and locking them would freeze a fiction.

Row lookups require **exactly one** matching row.  A report reformatted so that a label
matches twice, or no longer matches at all, fails loudly rather than silently locking
the wrong cell.

--------------------------------------------------------------------------------------
THE HASHES IN ``integrity``
--------------------------------------------------------------------------------------

Compared by exact string equality, never float tolerance.

``feature_order_sha256``
    The trained column order from ``artifacts/models/primary/contract.json``.  Kept
    separate from ``model_version`` so that when the fingerprint moves you can tell
    whether the feature contract moved with it.

``cost_constants_sha256``
    The ``policy/constants.py`` registry - name, value, unit and basis tag for every
    assumption, sorted by name.  Rationale prose is excluded on purpose: editing a
    justification must not look like editing a cost.

``elkan_thresholds_sha256``
    The load-bearing one.  ``policy/elkan.py`` computes thresholds from costs and
    effectiveness *only* - by design no label and no probability enters it - so hashing
    ``PolicyResult.thresholds`` alone would **not** move when the calibrator changes.
    The requirement is that a change to the model, the calibrator or the cost constants
    moves this digest, so it covers the whole per-order threshold *policy state*, in
    fixed order:

        test order_id sequence   ->  the population
        calibrated p per order   ->  model + calibrator
        p*(o, t) per action tier ->  cost constants + effectiveness + SHAP reason buckets
        selected tier per order  ->  the decision the three above produce

--------------------------------------------------------------------------------------
DETERMINISM, AND WHAT THE LOCK TEST MUST NOT COMPARE
--------------------------------------------------------------------------------------

Output is byte-identical across runs except ``metadata.captured_at``, the only
wall-clock field; everything else is a pure function of repo contents.  Pin
``SOURCE_DATE_EPOCH`` for byte-identity checks.  ``--selfcheck`` builds the lock twice in
one process and asserts the two serialisations match.

**Three fields are written for the record and excluded from comparison.**  Each is real
and each is worth having in the file; none of them is evidence that anything changed.

``metadata.captured_at``
    Wall-clock capture time.  Differs on every run by construction.

``metadata.git_sha``
    The SHA the capture ran against.  A file cannot contain the hash of the commit that
    introduces it, so after committing, this necessarily names the *parent*.  Asserting
    it equals HEAD would fail immediately and on every commit thereafter.  It records
    provenance, not identity.

``latency.*``
    ``p50_ms``, ``p99_ms`` and ``verdict_pass_ms`` are wall-clock measurements on an
    unquiesced desktop.  Five captures produced p50 between ~90ms and ~325ms on unchanged
    code at the same SHA.

    The instability is **between process invocations, not within them**: one run's three
    passes agreed to within 1.3% of each other at ~315ms while another run's sat at
    ~121ms.  So the worst-of-three protocol does not sample the thing that actually
    varies, and a single run cannot establish which regime the host is in.  An assertion
    on those numbers would fail for reasons unrelated to any change being tested, and an
    intermittently-red lock is worse than no lock - people learn to ignore it.

    Nothing is lost.  The latency work's real claim was never the speedup; it was that
    precomputing the store's per-group cumulative sums fixed a summation disagreement
    with the batch row-scan.  That property is deterministic and is covered by
    ``tests/test_store.py`` and ``tests/test_pit_exactness.py``, independent of any
    wall-clock timing.

--------------------------------------------------------------------------------------
LOCK VERSION 2 - WHY ``resolution`` CARRIES MORE THAN ``mdd_ap``
--------------------------------------------------------------------------------------

Version 1 stored ``mdd_ap: 0.043`` and nothing else about resolution.  That was a trap.
The MDD is a *planning* figure, and ``eval/significance.md`` §5 explicitly overturns the
"cannot be shown to beat the prevalence baseline" conclusion that was drawn from it:
the paired bootstrap gives a 95% CI on the AP difference that excludes zero, and the
permutation null gives p = 0.00060.  §6 explains why the planning figure was 3.5x too
pessimistic - it assumed AP ~0.10 against a realised ~0.02.

A reader of the version-1 lock alone got a conclusion this repo had already corrected.
That is the same defect class as every other stale number found in it: a frozen figure
the evidence moved past.  Version 2 therefore carries ``mdd_superseded_by`` naming the
correction, plus the two tests that replaced it, all parsed from
``eval/significance.md`` at run time.

--------------------------------------------------------------------------------------
WHAT THE SEVEN CROSS-CHECKS DO NOT COVER
--------------------------------------------------------------------------------------

The cross-checks reconcile the parsed half against the live half on: primary test rows,
primary test positives, treated orders, treated positives, the action-ladder total, and
SHAP additivity.  The seventh ties the two *reports* together rather than the two halves:
``paired_bootstrap_delta`` must equal ``average_precision - test_prevalence`` to 1.5e-4
(the reports' own print precision - see the check for why 1e-6 is unreachable), so
``significance.md`` and ``model_report.md`` cannot drift into describing different runs.

They are row counts, treated counts, an additivity error and one arithmetic identity.

**They do not reconcile any calibration value.**  ``brier``, ``smece`` and
``top_decile_calibration_gap`` are parsed from ``eval/calibration.md``, which reports
whichever window ``scripts/07_calibrate.py`` compared best; the live half here - and
``08_policy.py``, ``09_reasons.py``, ``11_fairness.py`` and ``api/service.py`` - all use
a hardcoded ``CALIBRATION_WINDOW_DAYS = 30``.  If those two ever diverged, the lock's
parsed half would describe a calibrator the system does not serve, and no check in this
script would notice.

That divergence is not currently possible to trigger by accident: 30 days is a fixed
choice, and ``eval/calibration.md`` is a window *comparison* rather than a selection
(the candidates are statistically indistinguishable - see that report).  It is recorded
here so the gap is known rather than unknown.

Usage
-----
    python scripts/90_capture_tier1_lock.py [--selfcheck]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL = REPO_ROOT / "eval"
OUT_PATH = EVAL / "TIER1_LOCK.json"
CONTRACT_PATH = REPO_ROOT / "artifacts" / "models" / "primary" / "contract.json"

LOCK_VERSION = 2
GENERATOR = "scripts/90_capture_tier1_lock.py"
CALIBRATION_WINDOW_DAYS = 30      # eval/calibration.md; same constant as 08_policy.py
LATENCY_PROTOCOL = "worst-of-three-passes"


class LockError(RuntimeError):
    """A source artifact did not contain what the lock needs, in the shape it needs."""


# =====================================================================================
# The required-key list lives here, in the script - never as null placeholders in the
# committed JSON.  Every path below is either present in the output with the type given,
# or listed in ``not_computed`` with a reason.  Never both, never neither.
# =====================================================================================

REQUIRED_KEYS: dict[str, type] = {
    "lock_version": int,
    "metadata.captured_at": str,
    "metadata.git_sha": str,
    "metadata.model_version": str,
    "metadata.generator": str,

    "targets.primary.label": str,
    "targets.primary.population": str,
    "targets.primary.dataset.test_rows": int,
    "targets.primary.dataset.test_positives": int,
    "targets.primary.dataset.test_prevalence": float,
    "targets.primary.resolution.mdd_ap": float,
    "targets.primary.resolution.mdd_superseded_by": str,
    "targets.primary.resolution.paired_bootstrap_delta": float,
    "targets.primary.resolution.paired_bootstrap_ci": list,
    "targets.primary.resolution.permutation_p": float,
    "targets.primary.resolution.permutation_null_p99": float,
    "targets.primary.performance.average_precision": float,
    "targets.primary.performance.roc_auc": float,
    "targets.primary.performance.brier": float,
    "targets.primary.performance.smece": float,
    "targets.primary.performance.top_decile_calibration_gap": float,
    "targets.primary.operating_points.elkan_policy.precision": float,
    "targets.primary.operating_points.elkan_policy.recall": float,
    "targets.primary.operating_points.elkan_policy.treated_frac": float,
    "targets.primary.operating_points.top_1pct.precision": float,
    "targets.primary.operating_points.top_1pct.recall": float,
    "targets.primary.operating_points.top_5pct.precision": float,
    "targets.primary.operating_points.top_5pct.recall": float,

    "targets.secondary.label": str,
    "targets.secondary.population": str,
    "targets.secondary.dataset.test_rows": int,
    "targets.secondary.dataset.test_positives": int,
    "targets.secondary.dataset.test_prevalence": float,
    "targets.secondary.resolution.mdd_ap": float,
    "targets.secondary.performance.average_precision": float,

    "policy.four_row_table.nothing.cost": float,
    "policy.four_row_table.nothing.ci_low": float,
    "policy.four_row_table.nothing.ci_high": float,
    "policy.four_row_table.everything.cost": float,
    "policy.four_row_table.everything.ci_low": float,
    "policy.four_row_table.everything.ci_high": float,
    "policy.four_row_table.hand_rule.cost": float,
    "policy.four_row_table.hand_rule.ci_low": float,
    "policy.four_row_table.hand_rule.ci_high": float,
    "policy.four_row_table.model.cost": float,
    "policy.four_row_table.model.ci_low": float,
    "policy.four_row_table.model.ci_high": float,

    "explainability.shap_additivity_max_abs_error": float,
    "explainability.shap_pred_contrib_agreement": float,

    "latency.p50_ms": float,
    "latency.p99_ms": float,
    "latency.verdict_pass_ms": float,
    "latency.protocol": str,

    "integrity.cost_constants_sha256": str,
    "integrity.elkan_thresholds_sha256": str,
    "integrity.feature_order_sha256": str,

    "not_computed": list,
}


# =====================================================================================
# Markdown parsing.  Generated tables only.
# =====================================================================================

def _read(name: str) -> str:
    path = EVAL / name
    if not path.exists():
        raise LockError(f"{path} is missing; run `make all` before capturing the lock")
    return path.read_text(encoding="utf-8")


def _section(text: str, heading_re: str) -> str:
    """Body of the ``## ...`` section whose heading matches.  ``###`` does not match."""
    parts = re.split(r"(?m)^(##[ \t]+.*)$", text)
    hits = [parts[i + 1] for i in range(1, len(parts), 2)
            if re.search(heading_re, parts[i])]
    if len(hits) != 1:
        raise LockError(
            f"expected exactly one '## ' section matching {heading_re!r}, "
            f"found {len(hits)}"
        )
    return hits[0]


def _subsection(block: str, heading_re: str) -> str:
    parts = re.split(r"(?m)^(###[ \t]+.*)$", block)
    hits = [parts[i + 1] for i in range(1, len(parts), 2)
            if re.search(heading_re, parts[i])]
    if len(hits) != 1:
        raise LockError(
            f"expected exactly one '### ' subsection matching {heading_re!r}, "
            f"found {len(hits)}"
        )
    return hits[0]


def _lead(block: str) -> str:
    """The part of a section before its first ``###`` subheading."""
    return re.split(r"(?m)^###[ \t]+", block)[0]


def _clean(cell: str) -> str:
    s = cell.replace("**", "").replace("`", "")
    s = re.sub(r"\*(.*?)\*", r"\1", s)          # italics, e.g. *(not the headline)*
    return re.sub(r"\s+", " ", s).strip()


def _rows(block: str) -> list[list[str]]:
    out: list[list[str]] = []
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue                             # separator row
        out.append(cells)
    return out


def _tables(block: str) -> list[list[list[str]]]:
    """Split a block into contiguous markdown tables, each a list of rows."""
    out: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not (cells and all(re.fullmatch(r":?-{3,}:?", c) for c in cells)):
                current.append(cells)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def _table(block: str, header_re: str) -> list[list[str]]:
    """
    The one table in ``block`` whose header row matches.

    Sections here routinely hold several tables with overlapping row labels - policy.md
    §3 carries the four-row cost table and the paired-difference table, and both have a
    row starting "Intervene on everything".  Selecting the table by its header first is
    what keeps a row lookup unambiguous.
    """
    hits = [t for t in _tables(block)
            if t and re.search(header_re, " | ".join(_clean(c) for c in t[0]))]
    if len(hits) != 1:
        raise LockError(
            f"expected exactly one table whose header matches {header_re!r}, "
            f"found {len(hits)}"
        )
    return hits[0][1:]      # drop the header row


def _row(rows: list[list[str]], pattern: str, col: int = 0) -> list[str]:
    hits = [r for r in rows if len(r) > col and re.search(pattern, _clean(r[col]))]
    if len(hits) != 1:
        raise LockError(
            f"expected exactly one row whose column {col} matches {pattern!r}, "
            f"found {len(hits)}"
        )
    return hits[0]


_NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _numbers(cell: str) -> list[float]:
    s = _clean(cell).replace("−", "-").replace(",", "")   # U+2212 MINUS SIGN
    return [float(m) for m in re.findall(_NUM_RE, s)]


def _num(cell: str) -> float:
    got = _numbers(cell)
    if not got:
        raise LockError(f"no number in cell {cell!r}")
    return got[0]


def _int(cell: str) -> int:
    v = _num(cell)
    if v != int(v):
        raise LockError(f"expected an integer in cell {cell!r}, got {v}")
    return int(v)


def _pct(cell: str) -> float:
    """A percentage cell, returned as a fraction."""
    if "%" not in cell:
        raise LockError(f"expected a percentage in cell {cell!r}")
    return _num(cell) / 100.0


def _ms(cell: str) -> float:
    if "ms" not in cell:
        raise LockError(f"expected a millisecond figure in cell {cell!r}")
    return _num(cell)


def _ci(cell: str) -> tuple[float, float]:
    got = _numbers(cell)
    if len(got) != 2:
        raise LockError(f"expected exactly two bounds in CI cell {cell!r}, got {got}")
    return got[0], got[1]


def _close(a: float, b: float, *, rel: float = 5e-3, abs_: float = 0.0) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise LockError(message)


# =====================================================================================
# Extractors
# =====================================================================================

def read_model_report() -> dict:
    md = _read("model_report.md")

    size = _rows(_section(md, r"Sample size and what it can resolve"))
    r_rows = _row(size, r"^Test rows$")
    r_pos = _row(size, r"^Test positives$")
    r_prev = _row(size, r"^Test prevalence$")
    r_mdd = _row(size, r"^Minimum detectable AP difference$")

    primary = _section(md, r"^##[ \t]+1\.")
    p_lead = _rows(_lead(primary))
    ap_p = _num(_row(p_lead, r"^Average precision$")[1])
    roc_p = _num(_row(p_lead, r"^ROC-AUC")[1])

    # | Budget | Orders treated | True positives | Precision | Recall | Lift |
    k_rows = _rows(_subsection(primary, r"Precision@k"))
    top1 = _row(k_rows, r"^top 1%$")
    top5 = _row(k_rows, r"^top 5%$")

    secondary = _rows(_lead(_section(md, r"^##[ \t]+5\.")))
    ap_s = _num(_row(secondary, r"^Average precision$")[1])
    s_pair = _numbers(_row(secondary, r"^Test rows / positives$")[1])
    _check(len(s_pair) == 2,
           "could not read the secondary rows/positives pair from model_report.md §5")

    out = {
        "primary": {
            "test_rows": _int(r_rows[1]),
            "test_positives": _int(r_pos[1]),
            "test_prevalence_reported": _pct(r_prev[1]),
            "mdd_ap": _num(r_mdd[1]),
            "average_precision": ap_p,
            "roc_auc": roc_p,
            "top_1pct": {"precision": _pct(top1[3]), "recall": _pct(top1[4]),
                         "treated": _int(top1[1]), "tp": _int(top1[2])},
            "top_5pct": {"precision": _pct(top5[3]), "recall": _pct(top5[4]),
                         "treated": _int(top5[1]), "tp": _int(top5[2])},
        },
        "secondary": {
            "test_rows": _int(r_rows[2]),
            "test_positives": _int(r_pos[2]),
            "test_prevalence_reported": _pct(r_prev[2]),
            "mdd_ap": _num(r_mdd[2]),
            "average_precision": ap_s,
        },
    }

    # The sample-size table and §5's own table must agree about the secondary
    # population, or one of the two is stale.
    _check(int(s_pair[0]) == out["secondary"]["test_rows"]
           and int(s_pair[1]) == out["secondary"]["test_positives"],
           "model_report.md §5 disagrees with the sample-size table on the secondary "
           f"population: {s_pair} vs ({out['secondary']['test_rows']}, "
           f"{out['secondary']['test_positives']})")

    # Prevalence is emitted as the exact ratio of two integers read from the report,
    # not as the report's three-significant-figure rendering of it.  They must agree.
    for side in ("primary", "secondary"):
        d = out[side]
        exact = d["test_positives"] / d["test_rows"]
        _check(_close(exact, d["test_prevalence_reported"], rel=2e-3),
               f"{side}: reported prevalence {d['test_prevalence_reported']} does not "
               f"match positives/rows = {exact}")
        d["test_prevalence"] = exact

    # precision = tp / treated and recall = tp / positives, in the report's own cells.
    for k in ("top_1pct", "top_5pct"):
        d = out["primary"][k]
        _check(_close(d["tp"] / d["treated"], d["precision"]),
               f"primary {k}: precision {d['precision']} != {d['tp']}/{d['treated']}")
        _check(_close(d["tp"] / out["primary"]["test_positives"], d["recall"]),
               f"primary {k}: recall {d['recall']} != {d['tp']}/"
               f"{out['primary']['test_positives']}")
    return out


MDD_SUPERSEDED_BY = (
    "eval/significance.md §6 — MDD assumed AP ~0.10 against a realised ~0.02, so it was "
    "3.5x too pessimistic here"
)


def read_significance() -> dict:
    """
    The tests that supersede the MDD.

    ``mdd_ap`` alone is a trap for a reader of the lock: it is a *planning* figure, and
    ``eval/significance.md`` §5 explicitly overturns the "not resolvable" conclusion that
    was drawn from it.  Locking the planning figure without the tests that replaced it
    would freeze a conclusion this repo has already corrected - the same defect class as
    every other stale number found in it.
    """
    md = _read("significance.md")

    perm = _rows(_section(md, r"^##[ \t]+2\."))
    p_value = _num(_row(perm, r"^Permutation p-value$")[1])
    null_pcts = _numbers(_row(perm, r"^Null 95th / 99th percentile$")[1])
    _check(len(null_pcts) == 2,
           "could not read the null 95th/99th percentile pair from significance.md §2")

    paired = _table(_section(md, r"^##[ \t]+3\."),
                    r"^Comparison \| Observed .* \| Bootstrap mean \| 95% CI \|")
    row = _row(paired, r"^model . prevalence baseline$")
    delta = _num(row[1])
    ci_low, ci_high = _ci(row[3])
    _check(ci_low <= delta <= ci_high,
           f"paired bootstrap delta {delta} sits outside its CI [{ci_low}, {ci_high}]")

    return {
        "mdd_superseded_by": MDD_SUPERSEDED_BY,
        "paired_bootstrap_delta": delta,
        "paired_bootstrap_ci": [ci_low, ci_high],
        "permutation_p": p_value,
        "permutation_null_p99": null_pcts[1],
    }


def read_calibration() -> dict:
    md = _read("calibration.md")

    brier = _num(_row(_rows(_section(md, r"^##[ \t]+3\.")), r"^Platt-calibrated$")[1])

    rel = _section(md, r"^##[ \t]+4\.")
    m = re.search(r"smECE\s*=\s*(" + _NUM_RE + r")", rel)
    _check(m is not None, "could not find the generated smECE line in calibration.md §4")
    smece = float(m.group(1))

    decile = _rows(_lead(_section(md, r"^##[ \t]+5\.")))
    gap = _pct(_row(decile, r"^Calibration gap")[1])
    predicted = _pct(_row(decile, r"^Mean predicted$")[1])
    observed = _pct(_row(decile, r"^Observed$")[1])
    _check(_close(predicted - observed, gap, abs_=5e-6),
           f"top-decile gap {gap} != predicted {predicted} - observed {observed}")

    return {"brier": brier, "smece": smece, "top_decile_calibration_gap": gap}


def read_policy() -> dict:
    md = _read("policy.md")

    four = _table(_section(md, r"^##[ \t]+3\."),
                  r"^Policy \| Orders treated \| Cost \(BRL\) \| 95% CI \|")
    _check(len(four) == 4,
           f"the four-row policy cost table has {len(four)} rows, not 4")
    table: dict[str, dict[str, float]] = {}
    for key, pattern in (
        ("nothing", r"^Intervene on nothing$"),
        ("everything", r"^Intervene on everything"),
        ("hand_rule", r"^Hand rule:"),
        ("model", r"^Model \+ per-order cost policy$"),
    ):
        row = _row(four, pattern)
        lo, hi = _ci(row[3])
        table[key] = {"cost": _num(row[2]), "ci_low": lo, "ci_high": hi,
                      "treated": _int(row[1])}
        _check(lo <= table[key]["cost"] <= hi,
               f"four-row table: {key} cost {table[key]['cost']} sits outside its CI "
               f"[{lo}, {hi}]")

    ladder = _table(_section(md, r"^##[ \t]+6\."),
                    r"^Tier \| Orders \| Intervention rate \| Positives \| Precision \|")
    tiers: dict[str, dict[str, int]] = {}
    for tier in ("allow", "confirm", "prepaid_only", "defer"):
        row = _row(ladder, rf"^{tier}$")
        orders = _int(row[1])
        # A tier with no orders renders positives as an em dash, not a zero.
        tiers[tier] = {"orders": orders, "positives": 0 if orders == 0 else _int(row[3])}

    action = ("confirm", "prepaid_only", "defer")
    treated = sum(tiers[t]["orders"] for t in action)
    treated_pos = sum(tiers[t]["positives"] for t in action)
    total = treated + tiers["allow"]["orders"]
    _check(treated > 0,
           "the Elkan policy treated no orders, so there is no operating point to lock")
    _check(table["everything"]["treated"] == total,
           f"policy.md: 'intervene on everything' treats {table['everything']['treated']}"
           f" but the action ladder totals {total} orders")
    _check(table["model"]["treated"] == treated,
           f"policy.md: the four-row table treats {table['model']['treated']} orders but "
           f"the action ladder treats {treated}")

    return {
        "four_row_table": table,
        "elkan_policy": {
            "precision": treated_pos / treated,
            "treated_frac": treated / total,
            "treated": treated,
            "treated_positives": treated_pos,
            "total": total,
        },
    }


def read_reasons(expected_rows: int) -> dict:
    md = _read("reasons.md")
    sec1 = _section(md, r"^##[ \t]+1\.")
    m = re.search(
        r"Max reconstruction error across ([\d,]+) orders:\s*`(" + _NUM_RE + r")`", sec1
    )
    _check(m is not None, "could not find the generated additivity line in reasons.md §1")
    n = int(m.group(1).replace(",", ""))
    _check(n == expected_rows,
           f"reasons.md computed additivity over {n} orders, but the primary test "
           f"population is {expected_rows}")
    err = float(m.group(2))

    assertion = _num(
        _row(_rows(_section(md, r"^##[ \t]+9\.")), r"^SHAP sums to margin")[1]
    )
    _check(_close(assertion, err, rel=1e-6),
           f"reasons.md §9 reports max error {assertion}, §1 reports {err}")
    return {"shap_additivity_max_abs_error": err}


def read_latency() -> dict:
    md = _read("latency.md")
    verdict = _section(md, r"^##[ \t]+1\.")
    total = _row(_rows(_lead(verdict)), r"^Total \(pooled\)$")
    p50, p99 = _ms(total[1]), _ms(total[2])

    passes = _rows(_subsection(verdict, r"The measurement is unstable"))
    per_pass = [_ms(r[1]) for r in passes
                if len(r) > 1 and re.fullmatch(r"\d+", _clean(r[0]))]
    _check(len(per_pass) >= 2,
           f"expected a per-pass p50 table in latency.md §1, found {len(per_pass)} passes")

    st_total = _row(_rows(_section(md, r"^##[ \t]+2\.")), r"^Total$")
    _check(_close(_ms(st_total[1]), p50, rel=1e-9)
           and _close(_ms(st_total[2]), p99, rel=1e-9),
           "latency.md §1 and §2 disagree on the pooled totals")

    return {"p50_ms": p50, "p99_ms": p99, "verdict_pass_ms": max(per_pass),
            "protocol": LATENCY_PROTOCOL}


# =====================================================================================
# Live pipeline - the same calls scripts/08_policy.py makes, plus api/service.py's
# version fingerprint.  Nothing here is re-implemented and nothing is written.
# =====================================================================================

def run_pipeline() -> dict:
    from data.loader import PRIMARY_LABEL, OlistLoader
    from features.builder import FeatureBuilder
    from models.calibration import PlattCalibrator
    from models.explain import ReasonExplainer
    from models.train import predict, prepare_matrix, train
    from policy.costs import ACTION_TIERS
    from policy.elkan import apply_policy

    loader = OlistLoader()
    boundary = loader.split_boundary

    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    matrix = FeatureBuilder(loader=loader).build(risk)
    split = risk["split"].to_numpy()
    y_all = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y_all[tr]), X.loc[va], pd.Series(y_all[va]),
        target=PRIMARY_LABEL, population="risk_set", category_levels=levels,
    )
    s_va = predict(bundle, X.loc[va], raw_score=True)
    s_te = predict(bundle, X.loc[te], raw_score=True)

    ts_va = risk.loc[va, "order_purchase_timestamp"]
    fit_mask = (ts_va >= boundary - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)).to_numpy()
    calibrator = PlattCalibrator().fit(s_va[fit_mask], y_all[va][fit_mask])
    p = calibrator.predict(s_te)

    feats = matrix.loc[te]
    explainer = ReasonExplainer(bundle)
    expl = explainer.explain(X.loc[te])
    additivity = explainer.assert_additive(expl, s_te)

    # The independent cross-check tests/test_explain.py pins: LightGBM's own TreeSHAP,
    # a separate implementation of the same algorithm.  eval/reasons.md states this
    # agreement in prose rather than as a generated figure, so it is measured here
    # instead of parsed.
    contrib = bundle.booster.predict(
        X.loc[te], num_iteration=bundle.best_iteration, pred_contrib=True
    )
    pred_contrib_agreement = float(np.abs(contrib[:, :-1] - expl.values).max())

    reasons = explainer.reason_buckets(expl)
    res = apply_policy(p, feats, reasons)

    treated = np.isin(res.tier, list(ACTION_TIERS))
    y_te = y_all[te].astype(bool)

    from api.service import ScoringService
    model_version = ScoringService(loader=loader).model_version

    return {
        "order_ids": risk.loc[te, "order_id"].astype(str).tolist(),
        "p": np.asarray(p, dtype=float),
        "thresholds": {t: np.asarray(res.thresholds[t], dtype=float)
                       for t in ACTION_TIERS},
        "tier": [str(t) for t in res.tier],
        "action_tiers": list(ACTION_TIERS),
        "feature_names": [str(f) for f in bundle.feature_names],
        "test_rows": int(te.sum()),
        "test_positives": int(y_te.sum()),
        "treated": int(treated.sum()),
        "treated_positives": int((treated & y_te).sum()),
        "additivity": float(additivity),
        "pred_contrib_agreement": pred_contrib_agreement,
        "model_version": str(model_version),
    }


# =====================================================================================
# Integrity digests
# =====================================================================================

def feature_order_sha256(names: list[str]) -> str:
    """
    Digest of the trained column order.

    Taken from the live bundle rather than from ``artifacts/models/primary/contract.json``
    because ``artifacts/`` is gitignored - a fresh clone has no contract file until
    ``make all`` has run, and a lock that cannot be recaptured from a clean checkout is
    not a lock.  The contract file is still cross-checked when it happens to be present:
    ``predict()`` addresses features positionally, so a silent disagreement between the
    two would be exactly the failure this digest exists to catch.
    """
    _check(bool(names) and all(isinstance(n, str) for n in names),
           "the trained bundle carries no feature-name list")
    if CONTRACT_PATH.exists():
        on_disk = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["feature_names"]
        _check(list(on_disk) == list(names),
               f"{CONTRACT_PATH.name} records a different column order from the trained "
               "bundle; one of the two is stale")
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def cost_constants_sha256() -> str:
    from policy.constants import ASSUMPTIONS, CURRENCY

    h = hashlib.sha256()
    h.update(f"currency={CURRENCY}\n".encode("utf-8"))
    for name in sorted(ASSUMPTIONS):
        a = ASSUMPTIONS[name]
        # ``:.17g`` round-trips a float exactly and renders NaN as "nan" on every
        # platform, so the two OBSERVED placeholders hash stably.  Rationale prose is
        # deliberately excluded: editing a justification is not editing a cost.
        h.update(f"{a.name}|{a.value:.17g}|{a.unit}|{a.basis}\n".encode("utf-8"))
    return h.hexdigest()


def elkan_thresholds_sha256(live: dict) -> str:
    h = hashlib.sha256()
    h.update(b"elkan-threshold-policy-state\x00v1\x00")
    h.update("\n".join(live["order_ids"]).encode("utf-8"))
    h.update(b"\x00")
    h.update(np.ascontiguousarray(live["p"], dtype="<f8").tobytes())
    for tier in live["action_tiers"]:
        h.update(tier.encode("utf-8"))
        h.update(b"\x00")
        h.update(np.ascontiguousarray(live["thresholds"][tier], dtype="<f8").tobytes())
    h.update("\n".join(live["tier"]).encode("utf-8"))
    return h.hexdigest()


# =====================================================================================
# Assembly
# =====================================================================================

def git_sha() -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def captured_at() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    when = (datetime.fromtimestamp(int(epoch), tz=timezone.utc) if epoch
            else datetime.now(timezone.utc))
    return when.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dig(tree: dict, path: str):
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


def validate(lock: dict) -> None:
    """Every required key is present and correctly typed, or listed in not_computed."""
    listed = {e["key"] for e in lock["not_computed"]}
    for entry in lock["not_computed"]:
        _check(set(entry) == {"key", "reason"} and str(entry["reason"]).strip(),
               f"not_computed entry {entry!r} needs exactly a key and a non-empty reason")
        _check(entry["key"] in REQUIRED_KEYS,
               f"not_computed names {entry['key']!r}, which is not a schema key")

    for path, want in REQUIRED_KEYS.items():
        value, present = _dig(lock, path)
        if path in listed:
            _check(not present,
                   f"{path} is in not_computed but also emitted; it must be one or the "
                   "other")
            continue
        _check(present, f"{path} is neither emitted nor listed in not_computed")
        if want is float:
            _check(isinstance(value, float) and not isinstance(value, bool),
                   f"{path} must be a float, got {type(value).__name__}: {value!r}")
            _check(bool(np.isfinite(value)),
                   f"{path} is {value!r}, which JSON cannot express")
        elif want is int:
            _check(isinstance(value, int) and not isinstance(value, bool),
                   f"{path} must be an int, got {type(value).__name__}: {value!r}")
        elif want is str:
            _check(isinstance(value, str) and bool(value.strip()),
                   f"{path} must be a non-empty string, got {value!r}")
        else:
            _check(isinstance(value, want), f"{path} must be {want.__name__}")

    def no_nulls(node, path: str = "") -> None:
        if node is None:
            raise LockError(f"null at {path or '<root>'}; the lock carries no nulls")
        if isinstance(node, dict):
            for k, v in node.items():
                no_nulls(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                no_nulls(v, f"{path}[{i}]")
        elif isinstance(node, float):
            _check(bool(np.isfinite(node)), f"non-finite float at {path}")

    no_nulls(lock)


def build_lock(live: dict, *, verbose: bool = True) -> dict:
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    not_computed: list[dict[str, str]] = []

    model = read_model_report()
    calib = read_calibration()
    pol = read_policy()
    reasons = read_reasons(model["primary"]["test_rows"])
    lat = read_latency()
    sig = read_significance()
    say("  parsed eval/: model_report, significance, calibration, policy, reasons, "
        "latency")

    # -- the committed reports must still describe the code -----------------------------
    _check(live["test_rows"] == model["primary"]["test_rows"],
           f"live pipeline has {live['test_rows']} primary test rows, eval/ reports "
           f"{model['primary']['test_rows']}")
    _check(live["test_positives"] == model["primary"]["test_positives"],
           f"live pipeline has {live['test_positives']} primary test positives, eval/ "
           f"reports {model['primary']['test_positives']}")
    _check(live["treated"] == pol["elkan_policy"]["treated"],
           f"live policy treats {live['treated']} orders, eval/policy.md reports "
           f"{pol['elkan_policy']['treated']}")
    _check(live["treated_positives"] == pol["elkan_policy"]["treated_positives"],
           f"live policy catches {live['treated_positives']} positives, eval/policy.md "
           f"reports {pol['elkan_policy']['treated_positives']}")
    _check(pol["elkan_policy"]["total"] == model["primary"]["test_rows"],
           f"policy.md's action ladder covers {pol['elkan_policy']['total']} orders, "
           f"model_report.md reports {model['primary']['test_rows']} test rows")
    _check(_close(live["additivity"], reasons["shap_additivity_max_abs_error"], rel=1e-3),
           f"live SHAP additivity error {live['additivity']:.3e} does not match "
           f"eval/reasons.md's {reasons['shap_additivity_max_abs_error']:.3e}")
    # Cross-check 7: the significance figures must describe the performance block they
    # sit beside.  The paired bootstrap's observed delta IS average precision minus the
    # prevalence baseline, so if the two were parsed from reports generated at different
    # times this disagrees and the capture fails rather than locking a mismatched pair.
    #
    # Tolerance is set by the reports' own print precision, not by taste.
    # model_report.md renders average precision at 4 decimals, which alone puts 3.6e-5
    # between the two sides; significance.md renders the delta at 4 decimals, adding
    # 3.6e-6.  1e-6 is therefore unreachable without more decimals in the reports, and
    # asserting it would fail on correct data.  1.5e-4 covers the rounding and keeps the
    # check's teeth: the drift this exists to catch - the hardcoded +0.0125 that used to
    # sit in significance.md §1 - is 3.2e-3 out, more than twenty times the tolerance.
    ap_minus_prevalence = (model["primary"]["average_precision"]
                           - model["primary"]["test_prevalence"])
    _check(_close(sig["paired_bootstrap_delta"], ap_minus_prevalence,
                  rel=0.0, abs_=1.5e-4),
           f"significance.md's paired bootstrap delta {sig['paired_bootstrap_delta']} "
           f"does not equal average_precision - test_prevalence = {ap_minus_prevalence} "
           "to 1e-6; the two reports describe different runs")
    say("  cross-checked eval/ against the live pipeline: consistent (7 checks)")

    lock = {
        "lock_version": LOCK_VERSION,
        "metadata": {
            "captured_at": captured_at(),
            "git_sha": git_sha(),
            "model_version": live["model_version"],
            "generator": GENERATOR,
        },
        "targets": {
            "primary": {
                "label": "b",
                "population": "risk_set",
                "dataset": {
                    "test_rows": model["primary"]["test_rows"],
                    "test_positives": model["primary"]["test_positives"],
                    "test_prevalence": model["primary"]["test_prevalence"],
                },
                "resolution": {
                    "mdd_ap": model["primary"]["mdd_ap"],
                    "mdd_superseded_by": sig["mdd_superseded_by"],
                    "paired_bootstrap_delta": sig["paired_bootstrap_delta"],
                    "paired_bootstrap_ci": sig["paired_bootstrap_ci"],
                    "permutation_p": sig["permutation_p"],
                    "permutation_null_p99": sig["permutation_null_p99"],
                },
                "performance": {
                    "average_precision": model["primary"]["average_precision"],
                    "roc_auc": model["primary"]["roc_auc"],
                    "brier": calib["brier"],
                    "smece": calib["smece"],
                    "top_decile_calibration_gap": calib["top_decile_calibration_gap"],
                },
                "operating_points": {
                    "elkan_policy": {
                        "precision": pol["elkan_policy"]["precision"],
                        "recall": (pol["elkan_policy"]["treated_positives"]
                                   / model["primary"]["test_positives"]),
                        "treated_frac": pol["elkan_policy"]["treated_frac"],
                    },
                    "top_1pct": {
                        "precision": model["primary"]["top_1pct"]["precision"],
                        "recall": model["primary"]["top_1pct"]["recall"],
                    },
                    "top_5pct": {
                        "precision": model["primary"]["top_5pct"]["precision"],
                        "recall": model["primary"]["top_5pct"]["recall"],
                    },
                },
            },
            "secondary": {
                "label": "a",
                "population": "all_matured",
                "dataset": {
                    "test_rows": model["secondary"]["test_rows"],
                    "test_positives": model["secondary"]["test_positives"],
                    "test_prevalence": model["secondary"]["test_prevalence"],
                },
                "resolution": {"mdd_ap": model["secondary"]["mdd_ap"]},
                "performance": {
                    "average_precision": model["secondary"]["average_precision"],
                },
            },
        },
        "policy": {
            "four_row_table": {
                key: {
                    "cost": pol["four_row_table"][key]["cost"],
                    "ci_low": pol["four_row_table"][key]["ci_low"],
                    "ci_high": pol["four_row_table"][key]["ci_high"],
                }
                for key in ("nothing", "everything", "hand_rule", "model")
            }
        },
        "explainability": {
            "shap_additivity_max_abs_error": reasons["shap_additivity_max_abs_error"],
            "shap_pred_contrib_agreement": live["pred_contrib_agreement"],
        },
        "latency": {
            "p50_ms": lat["p50_ms"],
            "p99_ms": lat["p99_ms"],
            "verdict_pass_ms": lat["verdict_pass_ms"],
            "protocol": lat["protocol"],
        },
        "integrity": {
            "cost_constants_sha256": cost_constants_sha256(),
            "elkan_thresholds_sha256": elkan_thresholds_sha256(live),
            "feature_order_sha256": feature_order_sha256(live["feature_names"]),
        },
        "not_computed": not_computed,
    }

    validate(lock)
    return lock


def serialise(lock: dict) -> str:
    return json.dumps(lock, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture eval/TIER1_LOCK.json.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="build the lock twice in one process and assert byte-identity")
    args = ap.parse_args()

    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        print("WARNING: tracked files are modified, so metadata.git_sha will not "
              "describe the working tree this lock was captured from:\n" + dirty,
              file=sys.stderr)

    print("Running the evaluation pipeline (nothing is modified) ...")
    live = run_pipeline()
    print(f"  model_version {live['model_version']}")

    lock = build_lock(live)
    text = serialise(lock)

    if args.selfcheck:
        if serialise(build_lock(live, verbose=False)) != text:
            print("SELFCHECK FAILED: two builds of the lock differ", file=sys.stderr)
            return 1
        print("  selfcheck: two in-process builds are byte-identical")

    OUT_PATH.write_bytes(text.encode("utf-8"))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT).as_posix()} "
          f"({len(text.encode('utf-8')):,} bytes, "
          f"{len(lock['not_computed'])} not_computed)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LockError as exc:
        print(f"LOCK CAPTURE FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
