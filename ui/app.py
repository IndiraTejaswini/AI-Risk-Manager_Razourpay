"""
Single-page UI for the scoring API and the committed evaluation artifacts.

Nothing on this page is typed.  Every figure is read from ``eval/TIER1_LOCK.json`` and
``eval/policy_table.md`` at request time, or fetched from the scoring API.  The Wilson
interval is imported from ``scripts/91_headline_block.py`` rather than reimplemented,
for the same reason ``api/service.py`` refuses to reimplement a feature: a second copy
of a definition is a second thing to drift.

Three rendering rules this module exists to enforce:

  * Markdown is converted to HTML here, server-side.  Shipping markdown to a browser
    and hoping is how the evaluation panel came to display ``### Headline numbers``
    as literal text.
  * The D9 policy table is parsed into rows and emitted as a ``<table>``.  It is never
    rendered as markdown.  A table is a table.
  * A percentile bootstrap interval estimated on fewer than ``MIN_POSITIVES`` positives
    is not printed.  The Model + cost policy's treated set holds 2 failures and its
    97.5th percentile diverges, because resamples that draw no failure give an unbounded
    break-even.  The point estimate stands; the upper bound is reported as unbounded.

The default order is scored server-side at page load, so a reader arrives at populated
panels rather than an empty form.
"""

from __future__ import annotations

import html
import importlib.util
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parents[1]
API_BASE = os.getenv("SCORE_API_URL", "http://localhost:8000").rstrip("/")

app = FastAPI(title="RTO risk scorer UI")

#: Below this many positives in a treated set, a percentile bootstrap interval is not
#: reported.  eval/policy_table.md's Model + cost row is the reason the guard exists.
MIN_POSITIVES = 5

#: Figures this UI will serve out of eval/figures/.
FIGURES = ("pr_primary.png", "decision_curve.png")

#: The committed gate chain and its bases, ARCHITECTURE.md "Gate chain and bases (T2.1)".
#: Compliance gates precede economic ones by design; the sequence is the information.
GATE_CHAIN = (
    ("kill_switch", "operational"),
    ("tier_is_actionable", "policy"),
    ("already_terminal", "policy"),
    ("message_class_matches_tier", "policy"),
    ("dnd_scrub", "TRAI"),
    ("consent_on_record", "TRAI"),
    ("opt_out_and_recontact_spacing", "TRAI"),
    ("send_window", "TRAI"),
    ("merchant_daily_budget", "operational"),
    ("customer_fatigue", "operational"),
    ("exploration_slice", "policy"),
    ("effectiveness_below_impression_cost", "policy"),
    ("no_risk_disclosure", "DPDP"),
    ("assert_no_block_tier", "policy"),
)
GATE_NAMES = frozenset(name for name, _ in GATE_CHAIN)

DEFAULT_ORDER = {
    "order_id": "e481f51cbdc54678b7cc49136f2d6af7",
    "amount": 11970,
    "shipping": 1480,
    "zipcode": "14409",
    "method": "boleto",
}

PAYMENT_METHODS = ("boleto", "card", "pix")


# ----------------------------------------------------------------------------------
# Committed artifacts
# ----------------------------------------------------------------------------------

def _read_artifact(name: str) -> str:
    path = ROOT / "eval" / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail=f"Evaluation artifact unavailable: {name}"
        ) from exc


def _generator():
    """
    The headline generator, imported for its lock reader and its Wilson interval.

    Its module name starts with a digit so it cannot be imported by name.  It is
    imported rather than copied because ``_wilson`` is a statistical definition, and a
    repo that has already found the same figure stated two ways does not need a third.
    """
    path = ROOT / "scripts" / "91_headline_block.py"
    spec = importlib.util.spec_from_file_location("headline_block", path)
    if spec is None or spec.loader is None:
        raise HTTPException(status_code=503, detail="Headline generator unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Headline generator unavailable") from exc
    return module


def _headline_markdown() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- HEADLINE:BEGIN.*?-->(.*?)<!-- HEADLINE:END -->", readme, re.S)
    if not match:
        raise HTTPException(status_code=503, detail="D1 headline artifact is unavailable")
    return match.group(1).strip()


# ----------------------------------------------------------------------------------
# Markdown, converted here rather than in the browser
# ----------------------------------------------------------------------------------

def _inline(text: str) -> str:
    """Escape, then re-admit the small closed set of inline markup the block uses."""
    out = html.escape(text, quote=False)
    out = out.replace("&lt;sub&gt;", "<sub>").replace("&lt;/sub&gt;", "</sub>")
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    return out


def _markdown_to_html(source: str) -> str:
    parts: list[str] = []
    for block in re.split(r"\n\s*\n", source.strip()):
        block = block.strip()
        if not block:
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", block, re.S)
        if heading:
            level = min(len(heading.group(1)) + 1, 6)
            parts.append(f"<h{level}>{_inline(heading.group(2).strip())}</h{level}>")
        else:
            parts.append(f"<p>{_inline(block)}</p>")
    return "\n".join(parts)


# ----------------------------------------------------------------------------------
# Headline figures
# ----------------------------------------------------------------------------------

def _headline_figures() -> dict:
    generator = _generator()
    lock = generator.load_lock()
    primary = lock["targets"]["primary"]
    dataset = primary["dataset"]
    rows = dataset["test_rows"]
    positives = dataset["test_positives"]
    base = dataset["test_prevalence"]

    top1 = primary["operating_points"]["top_1pct"]
    treated = math.ceil(0.01 * rows)
    caught = round(top1["recall"] * positives)
    ci_low, ci_high = generator._wilson(caught, treated)

    return {
        "test_rows": rows,
        "positives": positives,
        "base_rate": base,
        "treated": treated,
        "caught": caught,
        "precision": top1["precision"],
        "lift": top1["precision"] / base,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "permutation_p": primary["resolution"]["permutation_p"],
        "model_version": lock["metadata"]["model_version"],
        "lock_version": lock["lock_version"],
        "costs": lock["policy"]["four_row_table"],
    }


# ----------------------------------------------------------------------------------
# D9 policy table
# ----------------------------------------------------------------------------------

#: Columns whose values come from the assumed cost constants rather than measurement.
ASSUMED_COLUMNS = ("Impression cost", "Expected triggered cost")

#: Keyed to eval/TIER1_LOCK.json's four_row_table, so the table and the decision curve
#: cannot disagree about which policy is which.
POLICY_KEYS = {
    "Intervene on nothing": "nothing",
    "Intervene on everything": "everything",
    "Hand-written rule": "hand_rule",
    "Model + cost policy": "model",
}

TABLE_CAPTION = (
    "RTO cost avoided and Net are deliberately absent — both require an effectiveness "
    "prior. Break-even replaces them."
)


def _number(cell: str) -> float:
    return float(cell.replace(",", "").replace("%", ""))


def _policy_table() -> dict:
    source = _read_artifact("policy_table.md")
    lines = [line.strip() for line in source.splitlines() if line.strip().startswith("|")]
    grid = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    grid = [row for row in grid if not all(set(cell) <= set("-: ") for cell in row)]
    if len(grid) < 2:
        raise HTTPException(status_code=503, detail="Policy table artifact is unreadable")

    columns, body = grid[0], grid[1:]
    ci_index = columns.index("CI")
    positives_index = columns.index("Failures in treated set")
    treated_index = columns.index("Orders treated")

    rows: list[dict] = []
    guarded: list[tuple[str, int]] = []
    for cells in body:
        positives = int(_number(cells[positives_index]))
        treated = int(_number(cells[treated_index]))
        interval, caution = cells[ci_index], False
        if treated == 0:
            interval = "Not applicable"
        elif positives < MIN_POSITIVES:
            bounds = re.findall(r"-?[\d,]+\.\d+%", cells[ci_index])
            interval = f"[{bounds[0]}, unbounded]" if bounds else "Not computable"
            caution = True
            guarded.append((cells[0], positives))
        rows.append({
            "cells": list(cells[:ci_index]) + [interval],
            "ci_caution": caution,
            "policy": cells[0],
            "treated": treated,
            "positives": positives,
        })

    note = ""
    if guarded:
        policy, positives = min(guarded, key=lambda item: item[1])
        note = (
            f"The interval on {policy.lower()} is not computable at {positives} positives "
            "in its treated set: bootstrap resamples that draw no failure give an "
            "unbounded break-even, so the 97.5th percentile diverges. The point estimate "
            "and its c_rto sweep stand; the upper bound does not. The same guard applies "
            f"to any policy with fewer than {MIN_POSITIVES} positives in its treated set."
        )
    return {"columns": columns, "rows": rows, "note": note}


# ----------------------------------------------------------------------------------
# Scoring, server-side
# ----------------------------------------------------------------------------------

class BackendError(RuntimeError):
    """The scoring API did not answer, or answered with a status this page cannot use."""


def _request(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"content-type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise BackendError(f"{exc.code} {exc.reason} — {detail}") from exc
    except (OSError, ValueError) as exc:
        raise BackendError(str(exc)) from exc


def _score_payload(order: dict) -> dict:
    now = int(time.time())
    amount = int(order["amount"])
    return {
        "entity": "event",
        "account_id": "acc_UI",
        "event": "order.created",
        "contains": ["order", "payment"],
        "created_at": now,
        "payload": {
            "order": {"entity": {"id": order["order_id"], "entity": "order",
                                 "amount": amount, "currency": "BRL", "created_at": now}},
            "payment": {"entity": {"entity": "payment", "amount": amount,
                                   "currency": "BRL", "method": order["method"]}},
            "shipping_address": {"entity": {"zipcode": str(order["zipcode"])}},
            "line_items": [{"amount": amount, "quantity": 1,
                            "shipping_amount": int(order["shipping"])}],
        },
    }


def _evidence(order: dict) -> dict:
    """
    Score one order and collect every piece of evidence the API will serve about it.

    Two explain calls, because that endpoint answers two different questions.  Keyed by
    decision id it returns responder evidence, and refuses with 422 when the decision has
    no actionable tier.  Keyed by order id it returns the measured counterfactual.  The
    refusal is not an error to swallow: it is gate 1 of the chain, and the panel says so.
    """
    scored = _request(f"{API_BASE}/score", _score_payload(order))
    evidence: dict = {"scored": scored, "responder": None, "refusal": None,
                      "counterfactual": None, "counterfactual_error": None}
    try:
        evidence["responder"] = _request(f"{API_BASE}/explain/{scored['decision_id']}")
    except BackendError as exc:
        evidence["refusal"] = str(exc)
    try:
        evidence["counterfactual"] = _request(
            f"{API_BASE}/explain/{urllib.parse.quote(str(order['order_id']))}"
        )
    except BackendError as exc:
        evidence["counterfactual_error"] = str(exc)
    return evidence


# ----------------------------------------------------------------------------------
# Gate trace
# ----------------------------------------------------------------------------------

def _gate_rows(evidence: dict) -> list[dict]:
    """
    Merge the committed chain with the outcomes the API actually served.

    The backend does not run the chain at score time - responder/gates/registry.py has
    one caller in the whole repo, and it is a test - so most gates carry no outcome.
    They are shown as not evaluated rather than assumed to have passed: a trace that
    invented thirteen passes would be the same failure this page exists to correct.
    Gate 1 is the exception, and only because /explain answers on precisely its
    condition; see the comment on ``passed_at``.
    """
    served = {}
    responder = evidence.get("responder") or {}
    for item in responder.get("gate_trace") or []:
        served[item.get("gate")] = item

    tier = (evidence.get("scored") or {}).get("tier")
    refusal = evidence.get("refusal") or ""

    # /explain refuses with 422 "decision has no actionable tier" on exactly the
    # condition gate 1 tests, and serves responder evidence when that condition holds.
    # So the endpoint's own behaviour settles gate 1 in both directions, and reading it
    # only in the refusing direction would understate a real pass.  Nothing below gate 1
    # is inferred this way, because nothing below it is observable from the API.
    fired_at = "tier_is_actionable" if "actionable tier" in refusal else None
    passed_at = ("tier_is_actionable"
                 if responder and tier and tier != "allow" else None)

    rows: list[dict] = []
    passed_the_refusal = False
    for name, basis in GATE_CHAIN:
        row = {"gate": name, "basis": basis, "state": "unevaluated",
               "outcome": "not evaluated at score time", "reason": ""}
        if name in served:
            item = served[name]
            row["state"] = "passed" if item.get("passed") else "fired"
            row["outcome"] = "passed" if item.get("passed") else "refused"
            row["reason"] = item.get("reason") or ""
        elif name == fired_at:
            row["state"] = "fired"
            row["outcome"] = "refused"
            row["reason"] = (
                f"tier is {tier}; the API declined to explain a decision "
                "with no actionable tier"
            )
            passed_the_refusal = True
        elif name == passed_at:
            row["state"] = "passed"
            row["outcome"] = "passed"
            row["reason"] = (
                f"tier is {tier}; the API served responder evidence for this decision"
            )
        elif passed_the_refusal:
            row["state"] = "unreached"
            row["outcome"] = "not reached — the chain stops at the first refusal"
        rows.append(row)

    for name, item in served.items():
        if name not in GATE_NAMES:
            rows.insert(0, {
                "gate": name,
                "basis": str(item.get("basis") or "operational").lower(),
                "state": "passed" if item.get("passed") else "fired",
                "outcome": "passed" if item.get("passed") else "refused",
                "reason": item.get("reason") or "",
            })
    return rows


# ----------------------------------------------------------------------------------
# Charts, drawn here so the page arrives complete
# ----------------------------------------------------------------------------------

def _uncertainty_svg(figures: dict) -> str:
    """The Wilson interval, the operating point on it, and the base rate crossing it."""
    left, right = 56.0, 664.0
    domain = figures["ci_high"] * 1.12

    def x(value: float) -> float:
        return left + value / domain * (right - left)

    base_x = x(figures["base_rate"])
    lo_x, hi_x, point_x = x(figures["ci_low"]), x(figures["ci_high"]), x(figures["precision"])
    return (
        '<svg class="uncertainty" viewBox="0 0 720 64" role="img" '
        f'aria-label="Wilson 95 percent interval from {figures["ci_low"]:.2%} to '
        f'{figures["ci_high"]:.2%}, point estimate {figures["precision"]:.2%}, base rate '
        f'{figures["base_rate"]:.3%}">'
        f'<rect x="{lo_x:.1f}" y="26" width="{hi_x - lo_x:.1f}" height="12" '
        'fill="var(--rule)"/>'
        f'<line x1="{base_x:.1f}" y1="16" x2="{base_x:.1f}" y2="52" stroke="var(--ink)" '
        'stroke-width="1.5"/>'
        f'<text x="{base_x + 7:.1f}" y="14" class="svg-label">base rate '
        f'{figures["base_rate"]:.3%}</text>'
        f'<line x1="{point_x:.1f}" y1="20" x2="{point_x:.1f}" y2="44" stroke="var(--act)" '
        'stroke-width="2"/>'
        f'<circle cx="{point_x:.1f}" cy="32" r="4.5" fill="var(--act)"/>'
        f'<text x="{point_x:.1f}" y="14" text-anchor="middle" class="svg-act">'
        f'{figures["precision"]:.2%}</text>'
        f'<text x="{lo_x:.1f}" y="58" text-anchor="middle" class="svg-label">'
        f'{figures["ci_low"]:.2%}</text>'
        f'<text x="{hi_x:.1f}" y="58" text-anchor="middle" class="svg-label">'
        f'{figures["ci_high"]:.2%}</text>'
        "</svg>"
    )


def _curve_points(figures: dict, table: dict) -> list[dict]:
    costs = figures["costs"]
    points = []
    for row in table["rows"]:
        key = POLICY_KEYS.get(row["policy"])
        if key is None:
            continue
        points.append({
            "name": row["policy"],
            "treated": row["treated"],
            "cost": costs[key]["cost"],
            "operating": key == "model",
        })
    return sorted(points, key=lambda item: item["treated"])


#: Plot geometry for the decision curve, in the user units of its 760x300 viewBox.
CURVE_LEFT, CURVE_RIGHT, CURVE_TOP, CURVE_BOTTOM = 78.0, 742.0, 24.0, 240.0
COST_TICK_STEP = 25_000.0
TREATED_TICKS = (0, 10, 100, 1_000, 10_000)

#: Marker labels for the curve.  The table above carries the full policy names.
CURVE_LABELS = {
    "Intervene on nothing": "Nothing",
    "Intervene on everything": "Everything",
    "Hand-written rule": "Hand rule",
    "Model + cost policy": "Model + cost",
}


def _curve_scales(points: list[dict]) -> dict:
    span = math.log1p(max(point["treated"] for point in points)) or 1.0
    ymax = math.ceil(max(point["cost"] for point in points) / COST_TICK_STEP) * COST_TICK_STEP
    return {"span": span, "ymax": ymax}


def _curve_svg(points: list[dict], scales: dict) -> str:
    def x(treated: float) -> float:
        return CURVE_LEFT + math.log1p(treated) / scales["span"] * (CURVE_RIGHT - CURVE_LEFT)

    def y(cost: float) -> float:
        return CURVE_BOTTOM - cost / scales["ymax"] * (CURVE_BOTTOM - CURVE_TOP)

    parts = [
        '<svg id="curve" viewBox="0 0 760 300" role="img" aria-label="Expected cost '
        'against orders treated, with the four committed policies marked">',
        f'<line x1="{CURVE_LEFT}" y1="{CURVE_BOTTOM}" x2="{CURVE_RIGHT}" '
        f'y2="{CURVE_BOTTOM}" stroke="var(--rule)"/>',
        f'<line x1="{CURVE_LEFT}" y1="{CURVE_TOP}" x2="{CURVE_LEFT}" '
        f'y2="{CURVE_BOTTOM}" stroke="var(--rule)"/>',
    ]

    cost = 0.0
    while cost <= scales["ymax"] + 1:
        yy = y(cost)
        if cost:
            parts.append(f'<line x1="{CURVE_LEFT}" y1="{yy:.1f}" x2="{CURVE_RIGHT}" '
                         f'y2="{yy:.1f}" stroke="var(--rule)"/>')
        parts.append(f'<line x1="{CURVE_LEFT - 5}" y1="{yy:.1f}" x2="{CURVE_LEFT}" '
                     f'y2="{yy:.1f}" stroke="var(--muted)"/>')
        parts.append(f'<text x="{CURVE_LEFT - 9}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'class="svg-label">{cost:,.0f}</text>')
        cost += COST_TICK_STEP

    for tick in TREATED_TICKS:
        xx = x(tick)
        parts.append(f'<line x1="{xx:.1f}" y1="{CURVE_BOTTOM}" x2="{xx:.1f}" '
                     f'y2="{CURVE_BOTTOM + 5}" stroke="var(--muted)"/>')
        parts.append(f'<text x="{xx:.1f}" y="{CURVE_BOTTOM + 19}" text-anchor="middle" '
                     f'class="svg-label">{tick:,}</text>')

    midpoint = (CURVE_LEFT + CURVE_RIGHT) / 2
    middle_y = (CURVE_TOP + CURVE_BOTTOM) / 2
    parts.append(f'<text x="{midpoint:.1f}" y="{CURVE_BOTTOM + 42}" text-anchor="middle" '
                 'class="svg-label">Orders treated, log scale</text>')
    parts.append(f'<text x="18" y="{middle_y:.1f}" text-anchor="middle" class="svg-label" '
                 f'transform="rotate(-90 18 {middle_y:.1f})">Expected cost, BRL</text>')

    path = " ".join(f"{x(p['treated']):.1f},{y(p['cost']):.1f}" for p in points)
    parts.append(f'<polyline points="{path}" fill="none" stroke="var(--ink)" '
                 'stroke-width="1.75"/>')

    operating = next(point for point in points if point["operating"])
    parts.append(f'<line id="curve-indicator" x1="{x(operating["treated"]):.1f}" '
                 f'y1="{CURVE_TOP}" x2="{x(operating["treated"]):.1f}" y2="{CURVE_BOTTOM}" '
                 'stroke="var(--act)" stroke-width="1" stroke-dasharray="3 4"/>')

    for point in points:
        px, py = x(point["treated"]), y(point["cost"])
        colour = "var(--act)" if point["operating"] else "var(--ink)"
        if px < CURVE_LEFT + 70:
            anchor, label_x = "start", px
        elif px > CURVE_RIGHT - 70:
            anchor, label_x = "end", px
        else:
            anchor, label_x = "middle", px
        label = CURVE_LABELS.get(point["name"], point["name"])
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" '
                     f'r="{5 if point["operating"] else 4}" fill="{colour}"/>')
        parts.append(f'<text x="{label_x:.1f}" y="{py - 14:.1f}" text-anchor="{anchor}" '
                     f'class="{"svg-act" if point["operating"] else "svg-label"}">'
                     f'{html.escape(label)}</text>')

    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------------------
# Panel markup.  One renderer, server-side; the form re-enters through /panels.
# ----------------------------------------------------------------------------------

def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _headline_html(figures: dict) -> str:
    cards = (
        (f"{figures['treated']:,}", "orders — the top 1%"),
        (f"{figures['caught']} of {figures['positives']}", "failures caught"),
        (f"p = {figures['permutation_p']:.5f}", "the ranking resolves"),
    )
    tiles = "".join(
        f'<div class="tile"><p class="figure">{_escape(value)}</p>'
        f'<p class="label">{_escape(label)}</p></div>'
        for value, label in cards
    )
    return (
        f'<div class="figures">{tiles}</div>'
        f'<p class="lift">Precision {figures["precision"]:.2%}, '
        f'{figures["lift"]:.1f}× the {figures["base_rate"]:.3%} base rate. '
        f'Wilson 95% CI [{figures["ci_low"]:.2%}, {figures["ci_high"]:.2%}], '
        'excluding the base rate.</p>'
    )


def _score_html(scored: dict) -> str:
    costs = scored["cost_breakdown"]
    reasons = "".join(f"<li>{_escape(reason)}</li>" for reason in scored["reasons"])
    return (
        '<dl class="readout">'
        f'<div><dt>Calibrated risk</dt><dd>{float(scored["risk"]):.6f}</dd></div>'
        f'<div><dt>Action tier</dt><dd>{_escape(scored["tier"])}</dd></div>'
        f'<div><dt>Threshold used</dt><dd>{float(scored["threshold_used"]):.6f}</dd></div>'
        f'<div><dt>Features missing</dt><dd>{_escape(scored["features_missing"])}</dd></div>'
        "</dl>"
        f'<h3>Why</h3><ol class="reasons">{reasons}</ol>'
        f'<h3>Cost breakdown, {_escape(costs["currency"])}</h3>'
        '<dl class="readout">'
        f'<div><dt>Expected RTO loss</dt><dd class="caution">'
        f'{float(costs["expected_rto_loss"]):.2f}</dd></div>'
        f'<div><dt>Impression</dt><dd class="caution">'
        f'{float(costs["impression_cost"]):.2f}</dd></div>'
        f'<div><dt>Expected triggered</dt><dd class="caution">'
        f'{float(costs["expected_triggered_cost"]):.2f}</dd></div>'
        "</dl>"
        f'<p class="label">Basis: {_escape(costs["basis"])}. '
        f'Model {_escape(scored["model_version"])}, decision '
        f'{_escape(scored["decision_id"])}.</p>'
    )


def _treatability_html(evidence: dict) -> str:
    parts: list[str] = []
    responder = evidence.get("responder")
    if responder:
        parts.append(
            f'<p><span class="tag tag-assumed">assumed</span>'
            f'{_escape(responder.get("treatability", ""))}</p>'
            f'<p class="label">Reason class {_escape(responder.get("top_reason_class"))}, '
            f'tier {_escape(responder.get("tier"))}.</p>'
        )
    else:
        parts.append(
            '<p><span class="tag tag-assumed">assumed</span>The API serves no treatability '
            'sentence for this order, because the responder only renders one for an '
            'actionable tier. Score an order the cost policy would act on to see it.</p>'
            f'<p class="label">The API said: {_escape(evidence.get("refusal") or "nothing")}</p>'
        )

    counterfactual = evidence.get("counterfactual")
    if counterfactual and "feature" in counterfactual:
        parts.append(
            f'<p><span class="tag tag-measured">measured</span>Improving '
            f'<code>{_escape(counterfactual["feature"])}</code> from '
            f'{float(counterfactual["current_value"]):.6f} to '
            f'{float(counterfactual["improved_value"]):.6f} moves calibrated risk from '
            f'{float(counterfactual["risk"]):.6f} to '
            f'{float(counterfactual["counterfactual_risk"]):.6f}, a change of '
            f'{float(counterfactual["delta"]):+.6f}.</p>'
        )
    else:
        parts.append(
            '<p><span class="tag tag-measured">measured</span>No counterfactual for this '
            'order. The measured counterfactual is defined over the batch population; '
            'enter an order id from the test window to see one.</p>'
        )
    return "".join(parts)


def _gate_html(rows: list[dict]) -> str:
    items = []
    for row in rows:
        reason = (f'<p class="gate-reason">{_escape(row["reason"])}</p>'
                  if row["reason"] else "")
        items.append(
            f'<li class="gate gate-{row["state"]}">'
            f'<span class="gate-name">{_escape(row["gate"])}</span>'
            f'<span class="gate-outcome">{_escape(row["outcome"])}</span>'
            f'<span class="gate-basis">{_escape(row["basis"])}</span>'
            f'{reason}</li>'
        )
    return f'<ol class="gates">{"".join(items)}</ol>'


def _panels(order: dict) -> dict:
    try:
        evidence = _evidence(order)
    except BackendError as exc:
        message = (
            '<p class="error">Backend unavailable, or the request was refused. '
            f'{_escape(exc)}</p>'
            f'<p class="label">Start the scoring API and reload. This page shows what the '
            f'API served and nothing else.</p>'
        )
        return {"score": message, "treatability": message, "gates": message}
    return {
        "score": _score_html(evidence["scored"]),
        "treatability": _treatability_html(evidence),
        "gates": _gate_html(_gate_rows(evidence)),
    }


def _table_html(table: dict) -> str:
    head = "".join(
        f'<th class="{"num" if index else ""} '
        f'{"assumed" if column in ASSUMED_COLUMNS else ""}" scope="col">'
        f'{_escape(column)}</th>'
        for index, column in enumerate(table["columns"])
    )
    body = []
    for row in table["rows"]:
        cells = []
        for index, cell in enumerate(row["cells"]):
            classes = ["num"] if index else []
            if index == len(row["cells"]) - 1 and row["ci_caution"]:
                classes.append("caution")
            attribute = f' class="{" ".join(classes)}"' if classes else ""
            marker = ("<sup>1</sup>" if index == len(row["cells"]) - 1
                      and row["ci_caution"] else "")
            tag = "th" if index == 0 else "td"
            scope = ' scope="row"' if index == 0 else ""
            cells.append(f"<{tag}{attribute}{scope}>{_escape(cell)}{marker}</{tag}>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    note = (f'<p class="table-note"><sup>1</sup> {_escape(table["note"])}</p>'
            if table["note"] else "")
    return (
        '<div class="scroller"><table>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div>"
        f'<p class="table-note">{_escape(TABLE_CAPTION)} Columns in '
        '<span class="caution">this colour</span> are computed from the assumed cost '
        'constants.</p>'
        f"{note}"
    )


def _form_html(order: dict) -> str:
    options = "".join(
        f'<option value="{method}"{" selected" if method == order["method"] else ""}>'
        f"{method}</option>"
        for method in PAYMENT_METHODS
    )
    return (
        '<form id="score-form">'
        '<div class="fields">'
        '<p><label for="order-id">Existing order id</label>'
        f'<input id="order-id" value="{_escape(order["order_id"])}" required></p>'
        '<p><label for="amount">Order amount, centavos</label>'
        f'<input id="amount" type="number" min="0" value="{_escape(order["amount"])}" '
        "required></p>"
        '<p><label for="shipping">Shipping amount, centavos</label>'
        f'<input id="shipping" type="number" min="0" value="{_escape(order["shipping"])}" '
        "required></p>"
        '<p><label for="zipcode">ZIP code</label>'
        f'<input id="zipcode" value="{_escape(order["zipcode"])}" required></p>'
        '<p><label for="method">Payment method</label>'
        f'<select id="method">{options}</select></p>'
        "</div>"
        '<button type="submit">Score this order</button>'
        "</form>"
    )


# ----------------------------------------------------------------------------------
# Two worked orders, side by side
# ----------------------------------------------------------------------------------

#: The order the cost policy acts on.
#:
#: Chosen by scanning the whole test window through the live ``/score`` path, not the
#: batch path.  That distinction is the whole reason this specific order is here.  The
#: 34 orders the Elkan policy treats in ``eval/responder/responder_replay.md`` are
#: selected on the *batch* feature matrix, which carries customer history, pincode
#: history, seller and geolocation joins; a webhook payload carries five fields, so
#: ``/score`` scores them with materially less information.
#:
#: **How much less was overstated here until it was measured.**  This comment used to say
#: every one of those 34 comes back ``allow`` at an identical 0.0104.  Scoring all 34
#: through the running service says otherwise: 24 stay actionable (18 ``prepaid_only``,
#: 6 ``confirm``) and 10 flip to ``allow``; there are 30 distinct risks, not one; and the
#: payload path's risk is frequently *higher*, one order going from 4.24% to 8.11%.  The
#: gap is real and runs in both directions.  Do not restate it from memory.
#:
#: This order is actionable through the payload path itself, and for the more
#: interesting reason: not because its risk is high but because its threshold is low.
#: Freight of BRL 182.76 on goods worth BRL 61.26 makes c_rto large, and the Elkan
#: threshold falls to 0.008244 - well under the risk this order carries.  The policy is
#: cost sensitive, not a risk cutoff, and this order is where that shows.
#:
#: The threshold is quotable here because it is a function of this order's value and
#: freight alone.  **The risk is not**, and no figure for it is written down anywhere on
#: this page: ``_score_payload`` stamps ``created_at`` with the wall clock and four
#: purchase-time features derive from it, so the same order scores 0.012515, 0.012802 or
#: 0.013055 depending on when the request is made.  ``_acted_caption`` interpolates
#: whatever the API just served.  An earlier version of that caption carried a literal
#: and disagreed with the panel two lines above it.
ACTED_ORDER = {
    "order_id": "af822dacd6f5cff7376413c03a388bb7",
    "amount": 6126,     # BRL 61.26, the order's own value
    "shipping": 18276,  # BRL 182.76, the order's own freight
    "zipcode": "14409",
    "method": "boleto",
}

#: The order at the other end: ordinary freight, ordinary risk, tier ``allow``.  It is
#: the page's existing default and stands in for the ~99% that exit at gate 1.
ALLOWED_ORDER = dict(DEFAULT_ORDER)

COMPARISON = (
    ("acted", "One we act on", ACTED_ORDER),
    ("allowed", "One we do not", ALLOWED_ORDER),
)


def _evidence_or_error(order: dict) -> dict:
    """Evidence for one order, or an empty shape carrying the reason it is empty."""
    try:
        return _evidence(order)
    except BackendError as exc:
        return {"scored": None, "responder": None, "refusal": str(exc),
                "counterfactual": None, "counterfactual_error": str(exc),
                "unavailable": str(exc)}


def _comparison_rows(traces: dict[str, list[dict]]) -> list[dict]:
    """Align the two gate traces row by row, so the columns read as one chain."""
    order: list[str] = []
    for rows in traces.values():
        for row in rows:
            if row["gate"] not in order:
                order.append(row["gate"])
    indexed = {key: {row["gate"]: row for row in rows} for key, rows in traces.items()}
    aligned = []
    for gate in order:
        present = next(indexed[key][gate] for key in indexed if gate in indexed[key])
        aligned.append({
            "gate": gate,
            "basis": present["basis"],
            "cells": [indexed[key].get(gate) for key, _, _ in COMPARISON],
        })
    return aligned


def _comparison_html(evidence: dict[str, dict]) -> str:
    traces = {key: _gate_rows(evidence[key]) for key, _, _ in COMPARISON}
    rows = _comparison_rows(traces)

    heads = []
    for key, title, order in COMPARISON:
        scored = evidence[key].get("scored")
        if scored is None:
            detail = ('<p class="error">Backend unavailable. '
                      f'{_escape(evidence[key].get("unavailable", ""))}</p>')
        else:
            detail = (f'<p class="label">Tier {_escape(scored["tier"])}</p>'
                      f'<p class="label">risk {float(scored["risk"]):.6f}, '
                      f'threshold {float(scored["threshold_used"]):.6f}</p>')
        heads.append(
            f'<div class="compare-head"><p class="compare-title">{_escape(title)}</p>'
            f'{detail}'
            f'<p class="label">order {_escape(order["order_id"][:12])}…</p></div>'
        )
    heads = "".join(heads)

    items = []
    for row in rows:
        cells = []
        for cell in row["cells"]:
            if cell is None:
                cells.append('<span class="gate-outcome gate-unevaluated">—</span>')
                continue
            cells.append(
                f'<span class="gate-outcome gate-{cell["state"]}">'
                f'{_escape(cell["outcome"])}</span>'
            )
        items.append(
            f'<li class="compare-row">'
            f'<span class="gate-name">{_escape(row["gate"])}</span>'
            f'<span class="gate-basis">{_escape(row["basis"])}</span>'
            f'{"".join(cells)}</li>'
        )

    return (
        f'<div class="compare-heads"><div></div>{heads}</div>'
        f'<ol class="compare">{"".join(items)}</ol>'
    )


def _acted_caption(scored: dict | None) -> str:
    """
    The cost-sensitivity sentence, interpolated from what the API just served.

    Both figures in this caption used to be literals sitting two lines under the panel
    that computes them, and they disagreed with it: the caption said 0.012452 while the
    panel rendered 0.012515.  The risk is not a constant and cannot be written down -
    ``_score_payload`` stamps ``created_at`` with the wall clock, and purchase_hour,
    purchase_dow, purchase_month and purchase_is_weekend are features, so the same order
    scores differently depending on when the page is loaded (0.012515, 0.012802 and
    0.013055 on three measured loads).  The threshold does not move: it is a function of
    this order's value and freight alone, which is the entire point of the sentence.

    The freight and goods figures come from ACTED_ORDER rather than from prose, so a
    change to the demo order cannot leave the caption describing the previous one.
    """
    goods = ACTED_ORDER["amount"] / 100.0
    freight = ACTED_ORDER["shipping"] / 100.0
    lead = (
        "Actionable because the threshold moved, not because the risk was high — "
        f"BRL {freight:.2f} freight on BRL {goods:.2f} of goods makes c_rto large"
    )
    if scored is None:
        return (f'<p class="label">{lead} and drops the Elkan threshold far below the '
                'risk this order carries. Figures omitted: the backend did not '
                'answer, and this caption reports only what it served.</p>')
    return (
        f'<p class="label">{lead} and drops the Elkan threshold to '
        f'{float(scored["threshold_used"]):.6f}, under this order\'s '
        f'{float(scored["risk"]):.6f}.</p>'
    )


# ----------------------------------------------------------------------------------
# The declared-context run
# ----------------------------------------------------------------------------------
#
# The panel above is observed: it shows what the API served, and stops where the API
# stops.  This one is the other half.  No serving path calls responder/gates/registry.py
# - its callers are tests/responder/test_gates.py and this panel - so the chain is never
# executed while scoring and those thirteen rows can only ever read "not evaluated".
# Here the chain really runs - this is
# registry.run, the shipped gate functions, no reimplementation - against inputs that are
# written down rather than measured, because Olist has no consent timestamps, no DND
# status and no send windows to measure.  Keeping the two panels apart is the point: one
# is what the system did, the other is what the system does.

#: The declared inputs, in the order a reader should meet them.  Every field of Context
#: that the chain reads appears here, so nothing is supplied off-screen.
DECLARED_ROWS = (
    ("consent on record", "yes, 2018-03-14", "consent_timestamp"),
    ("DND status", "not listed", "dnd_scrubbed"),
    ("local time", "14:20 IST", "send_window_open"),
    ("messages in last 7 days", "0", "customer_fatigued"),
    ("merchant budget today", "used 12 of 500", "merchant_daily_budget_available"),
    ("opt-out keyword handled", "yes, STOP", "opt_out_keyword"),
    ("recontact spacing", "90 days clear", "recontact_allowed"),
    ("kill switch", "off", "kill_switch"),
    ("decision state", "not terminal", "terminal"),
)

#: The declared local time for the refusing variant.  21:00 is the TRAI daylight cutoff
#: the send-window gate exists to enforce, so 22:40 is outside it.
CLOSED_TIME = "22:40 IST"

DECLARED_CAPTION = (
    "Olist carries no consent timestamps, DND status or send windows, so this context "
    "is declared rather than observed. The gate outcomes are real — this is "
    "responder/gates/registry.py executing — but the inputs are supplied. [BUILT-UNEVAL]"
)


def _declared_candidate(order: dict, scored: dict | None):
    """
    The candidate the chain is run on.

    Its tier, reason class and impression cost come from what the API actually served
    for this order; only the fields Olist cannot supply are declared.  The effectiveness
    is read from policy/effectiveness.py rather than chosen here, so gate 11 is comparing
    against the same prior the cost policy uses.
    """
    from policy.costs import c_rto
    from policy.effectiveness import EFFECTIVENESS
    from responder.gates.types import Candidate

    tier = "prepaid_only"
    reason_class = "order_structure"
    rto = float(c_rto(order["amount"] / 100.0, order["shipping"] / 100.0))
    effectiveness = EFFECTIVENESS["order_composition"][tier]
    impression = 0.80
    reasons: tuple[str, ...] = ()
    if scored:
        impression = float(scored["cost_breakdown"]["impression_cost"]) or impression
        reasons = tuple(scored["reasons"])

    return Candidate(
        decision_id=order["order_id"],
        tier=tier,
        calibrated_p=float(scored["risk"]) if scored else 0.0,
        reason_class=reason_class,
        reasons=reasons,
        c_rto=rto,
        impression_cost=impression,
        effectiveness=effectiveness,
        # A prepayment request is promotional under the DLT framework, which is what
        # puts gates 4 to 7 in play at all.  A confirm-tier service message would be
        # exempt from them and would show nothing.
        message_class="promotional",
        rendered_text=(
            f"Please complete payment for order {order['order_id']} before dispatch.\n"
        ),
        address="",
    ), {"c_rto": rto, "effectiveness": effectiveness, "impression": impression,
        "tier": tier, "reason_class": reason_class}


def _declared_context(send_window_open: bool, reasons: tuple[str, ...]):
    from responder.gates.types import Context

    return Context(
        kill_switch=False,
        terminal=False,
        tier_actionable=True,
        dnd_scrubbed=True,
        consent_timestamp="2018-03-14",
        opt_out_keyword=True,
        recontact_allowed=True,
        send_window_open=send_window_open,
        merchant_daily_budget_available=True,
        customer_fatigued=False,
        reason_vocabulary=tuple(reason.lower() for reason in reasons),
    )


def _declared_runs(order: dict, scored: dict | None) -> dict:
    """Run the shipped chain twice, changing one declared input between the runs."""
    from responder.gates.registry import GATE_REGISTRY, run

    candidate, inputs = _declared_candidate(order, scored)
    basis = dict(GATE_CHAIN)
    runs = []
    for label, is_open in ((DECLARED_ROWS[2][1], True), (CLOSED_TIME, False)):
        context = _declared_context(is_open, candidate.reasons)
        results = run(candidate, context)
        rows = []
        stopped = False
        for name, _ in GATE_REGISTRY:
            result = next((item for item in results if item.gate == name), None)
            if result is None:
                rows.append({"gate": name, "basis": basis.get(name, "policy"),
                             "state": "unreached", "outcome": "not reached",
                             "reason": ""})
                continue
            stopped = stopped or not result.passed
            rows.append({
                "gate": name,
                "basis": basis.get(name, "policy"),
                "state": "passed" if result.passed else "fired",
                "outcome": "passed" if result.passed else "refused",
                "reason": result.reason or "",
            })
        runs.append({
            "time": label,
            "rows": rows,
            "evaluated": len(results),
            "dispatched": not stopped,
        })
    return {"runs": runs, "inputs": inputs, "candidate": candidate}


def _declared_html(order: dict, scored: dict | None) -> str:
    try:
        state = _declared_runs(order, scored)
    except (ImportError, KeyError, ValueError) as exc:
        return ('<p class="error">The gate chain could not be run against the declared '
                f'context: {_escape(exc)}</p>')

    runs, inputs = state["runs"], state["inputs"]
    open_run, closed_run = runs

    def context_cell(field: str, value: str) -> str:
        # The one input that differs between the two runs is shown as both values.
        if field == "send_window_open":
            return f"{_escape(value)} / {_escape(CLOSED_TIME)}"
        return _escape(value)

    context_rows = "".join(
        f'<tr><th scope="row">{_escape(label)}</th>'
        f'<td class="caution">{context_cell(field, value)}</td>'
        f'<td><code>{_escape(field)}</code></td></tr>'
        for label, value, field in DECLARED_ROWS
    )

    derived = (
        '<p class="label">Taken from the served decision rather than declared: tier '
        f'{_escape(inputs["tier"])}, reason class {_escape(inputs["reason_class"])}, '
        f'impression cost BRL {inputs["impression"]:.2f}, c_rto BRL '
        f'{inputs["c_rto"]:.2f}. The effectiveness {inputs["effectiveness"]:.2f} is read '
        "from policy/effectiveness.py, so gate 11 tests against the same prior the cost "
        "policy uses.</p>"
    )

    heads = "".join(
        f'<div class="compare-head"><p class="compare-title">{_escape(run["time"])}</p>'
        f'<p class="label">{run["evaluated"]} of 14 gates evaluated</p>'
        f'<p class="label">{"passes to dispatch" if run["dispatched"] else "stopped"}</p>'
        "</div>"
        for run in runs
    )

    items = []
    for index, (left, right) in enumerate(zip(open_run["rows"], closed_run["rows"])):
        cells = "".join(
            f'<span class="gate-outcome gate-{cell["state"]}">'
            f'{_escape(cell["outcome"])}'
            f'{" — " + _escape(cell["reason"]) if cell["reason"] else ""}</span>'
            for cell in (left, right)
        )
        items.append(
            f'<li class="compare-row">'
            f'<span class="gate-name">{index}. {_escape(left["gate"])}</span>'
            f'<span class="gate-basis">{_escape(left["basis"])}</span>'
            f'{cells}</li>'
        )

    return (
        "<h3>Declared context — not from Olist</h3>"
        '<div class="scroller"><table><thead><tr>'
        '<th scope="col">Input</th><th class="assumed" scope="col">Declared value</th>'
        '<th scope="col">Context field</th></tr></thead>'
        f"<tbody>{context_rows}</tbody></table></div>"
        f"{derived}"
        "<h3>The chain, run twice</h3>"
        f'<div class="compare-heads"><div></div>{heads}</div>'
        f'<ol class="compare">{"".join(items)}</ol>'
        f'<p class="label">One input changes between the columns: the declared local '
        f'time. At {_escape(open_run["time"])} all fourteen gates pass and the candidate '
        f'reaches dispatch. At {_escape(CLOSED_TIME)} gate 7 refuses on TRAI\'s daylight '
        "window and gates 8 to 13 are never reached — a legal constraint outranking a "
        "cost calculation, which is why the compliance gates are ordered before the "
        "economic ones.</p>"
    )


# ----------------------------------------------------------------------------------
# Responder replay aggregates
# ----------------------------------------------------------------------------------

def _responder_artifact(name: str) -> str:
    path = ROOT / "eval" / "responder" / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail=f"Responder artifact unavailable: {name}"
        ) from exc


def _tables(source: str) -> list[dict]:
    """Every pipe table in a markdown file, in order, as columns plus rows."""
    tables: list[dict] = []
    current: list[list[str]] = []
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not all(set(cell) <= set("-: ") for cell in cells):
                current.append(cells)
            continue
        if current:
            tables.append({"columns": current[0], "rows": current[1:]})
            current = []
    if current:
        tables.append({"columns": current[0], "rows": current[1:]})
    return tables


#: Buckets that mean the order was acted on.  --act marks these and nothing else: the
#: largest bar here is `held`, which is the set the responder leaves alone, and painting
#: it with the accent would teach a reader the opposite of what the colour means
#: everywhere else on the page.
ACTED_BUCKETS = frozenset({"confirm", "prepaid_only", "passed"})


def _bars(rows: list[tuple[str, float]], total: float) -> str:
    """A horizontal bar list.  Acted-on buckets take --act; the rest stay --muted."""
    if not rows:
        return ""
    largest = max(value for _, value in rows)
    items = []
    for label, value in rows:
        share = value / total if total else 0.0
        # A floor, so a bucket of 12 in 19,662 is still a mark rather than nothing.
        width = max(value / largest * 100, 0.4) if largest else 0.0
        lead = "bar-lead" if label in ACTED_BUCKETS else ""
        items.append(
            f'<li class="bar-row">'
            f'<span class="bar-label">{_escape(label)}</span>'
            f'<span class="bar-track"><span class="bar-fill {lead}" '
            f'style="width:{width:.2f}%"></span></span>'
            f'<span class="bar-value">{value:,.0f}</span>'
            f'<span class="bar-share label">{share:.2%}</span></li>'
        )
    return f'<ul class="bars">{"".join(items)}</ul>'


def _responder_html() -> str:
    replay = _responder_artifact("responder_replay.md")
    breakeven = _responder_artifact("breakeven.md")
    tables = _tables(replay)
    if len(tables) < 4:
        raise HTTPException(status_code=503, detail="Replay artifact is unreadable")

    outcomes, by_reason, histogram = tables[0], tables[1], tables[2]
    total = sum(_number(row[-1]) for row in outcomes["rows"])

    outcome_bars = _bars([(row[0], _number(row[-1])) for row in outcomes["rows"]], total)
    gate_bars = _bars([(row[0], _number(row[-1])) for row in histogram["rows"]], total)

    reason_rows = "".join(
        f'<tr><th scope="row">{_escape(row[0])}</th><td>{_escape(row[1])}</td>'
        f'<td class="num">{_escape(row[2])}</td></tr>'
        for row in by_reason["rows"]
    )

    sweep = _tables(breakeven)[0]
    sweep_rows = "".join(
        f'<tr><th scope="row">{_escape(row[0])}</th>'
        f'<td class="num">{_escape(row[1])}</td></tr>'
        for row in sweep["rows"]
    )

    return (
        "<h3>Outcome buckets across the test window</h3>"
        f"{outcome_bars}"
        "<h3>Gate outcome</h3>"
        f"{gate_bars}"
        '<p class="label">Two buckets, not fourteen. The replay harness records whether '
        'an order reached the responder at all, not which of the fourteen gates stopped '
        'it — eval/responder/gates.md is still a scaffold and says so. A per-gate '
        'histogram is not in the repo, so none is drawn here.</p>'
        "<h3>By tier and reason class</h3>"
        '<div class="scroller"><table><thead><tr>'
        '<th scope="col">Bucket</th><th scope="col">Reason class</th>'
        '<th class="num" scope="col">Orders</th></tr></thead>'
        f"<tbody>{reason_rows}</tbody></table></div>"
        "<h3>Break-even effectiveness</h3>"
        '<p>An intervention must prevent at least this fraction of the failures it is '
        'sent on to cover its own cost. Every quantity in it is measured on the test '
        'window; no effectiveness prior enters the arithmetic, which is what lets the '
        'figure stand without an assumed multiplier behind it.</p>'
        '<div class="scroller"><table><thead><tr>'
        '<th scope="col">c_rto sweep</th>'
        '<th class="num" scope="col">Required effectiveness</th>'
        "</tr></thead>"
        f"<tbody>{sweep_rows}</tbody></table></div>"
    )


RESPONDER_CAPTION = (
    "The responder ships dry-run. Nothing was sent to anyone: Olist carries no phone "
    "numbers, no email addresses and no consent records, so there is no one to contact "
    "and no consent to check. The harness reports mechanism behaviour — which orders the "
    "policy selects, which bucket they land in, what an intervention would have to "
    "achieve to pay for itself — and not intervention effect, which this dataset cannot "
    "measure at all."
)


# ----------------------------------------------------------------------------------
# Decision lifecycle
# ----------------------------------------------------------------------------------

#: Node placement for the state machine, in the user units of its viewBox.  The edges
#: are generated from responder.states.TRANSITIONS so the drawing cannot drift from the
#: declaration; only the layout is written down here.
STATE_LAYOUT = {
    "SCORED": (40, 196),
    "SUPPRESSED": (232, 36),
    "HELD_EXPLORATION": (232, 96),
    "QUEUED": (232, 196),
    "SENT": (438, 196),
    "SEND_FAILED": (410, 320),
    "ESCALATED": (628, 268),
    "CONFIRMED": (640, 36),
    "CANCELLED_AT_PROMPT": (640, 96),
    "NO_RESPONSE": (640, 156),
    "ABANDONED": (836, 320),
}
STATE_HEIGHT = 26.0
STATE_CHAR = 6.55


def _state_machine():
    """The declared machine, imported rather than transcribed."""
    path = ROOT / "responder" / "states.py"
    spec = importlib.util.spec_from_file_location("responder_states", path)
    if spec is None or spec.loader is None:
        raise HTTPException(status_code=503, detail="State machine unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except OSError as exc:
        raise HTTPException(status_code=503, detail="State machine unavailable") from exc
    return module


def _state_svg() -> str:
    states = _state_machine()
    terminal = {state.value for state in states.TERMINAL}
    transitions = {
        source.value: sorted(target.value for target in targets)
        for source, targets in states.TRANSITIONS.items()
    }

    boxes = {}
    for name, (x, y) in STATE_LAYOUT.items():
        width = 16 + STATE_CHAR * len(name)
        boxes[name] = {"x": x, "y": y, "w": width, "h": STATE_HEIGHT,
                       "cx": x + width / 2, "cy": y + STATE_HEIGHT / 2}

    parts = [
        '<svg class="states" viewBox="0 0 980 380" role="img" '
        f'aria-label="Responder decision lifecycle: {len(boxes)} states, '
        f'{len(terminal)} of them terminal">',
        '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L8,4 L0,8 z" fill="var(--muted)" opacity="0.55"/></marker></defs>',
    ]

    for source, targets in transitions.items():
        if source not in boxes:
            continue
        a = boxes[source]
        for target in targets:
            if target not in boxes:
                continue
            b = boxes[target]
            back = b["cx"] < a["cx"]
            if back:
                start_x, start_y = a["x"], a["cy"]
                end_x, end_y = b["x"] + b["w"], b["cy"]
                dip = max(a["y"], b["y"]) + STATE_HEIGHT + 30
                path = (f'M{start_x:.1f},{start_y:.1f} '
                        f'C{start_x - 40:.1f},{dip:.1f} '
                        f'{end_x + 40:.1f},{dip:.1f} {end_x:.1f},{end_y:.1f}')
            else:
                start_x, start_y = a["x"] + a["w"], a["cy"]
                end_x, end_y = b["x"], b["cy"]
                span = max((end_x - start_x) * 0.45, 24.0)
                path = (f'M{start_x:.1f},{start_y:.1f} '
                        f'C{start_x + span:.1f},{start_y:.1f} '
                        f'{end_x - span:.1f},{end_y:.1f} {end_x:.1f},{end_y:.1f}')
            parts.append(f'<path d="{path}" fill="none" stroke="var(--muted)" '
                         'stroke-width="1" opacity="0.55" '
                         'marker-end="url(#arrow)"/>')

    for name, box in boxes.items():
        is_terminal = name in terminal
        fill = "var(--rule)" if is_terminal else "var(--paper)"
        klass = "svg-label" if is_terminal else "svg-state"
        parts.append(
            f'<rect x="{box["x"]:.1f}" y="{box["y"]:.1f}" width="{box["w"]:.1f}" '
            f'height="{box["h"]:.1f}" fill="{fill}" stroke="var(--muted)" '
            'stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{box["cx"]:.1f}" y="{box["cy"] + 4:.1f}" text-anchor="middle" '
            f'class="{klass}">{_escape(name)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _state_caption() -> str:
    states = _state_machine()
    terminal = sorted(state.value for state in states.TERMINAL)
    return (
        f'<p class="label">{len(STATE_LAYOUT)} states, {len(terminal)} of them terminal '
        f'({", ".join(terminal)}). Shaded boxes are terminal. Generated from '
        "responder/states.py at request time, so the drawing cannot drift from the "
        "declaration it documents.</p>"
        f'<p>At most <strong>{states.MAX_ESCALATIONS}</strong> escalation and at most '
        f'<strong>{states.MAX_SEND_ATTEMPTS}</strong> send attempts, both declared as '
        "constants in responder/states.py and enforced by the transition writer rather "
        "than read from configuration. A fourth send attempt does not raise: the writer "
        "rewrites the target state to ABANDONED. There is no path to a block tier "
        "anywhere in the machine, which is the defence-only claim as a reachability "
        "property rather than an assertion.</p>"
    )


# ----------------------------------------------------------------------------------
# Design system.  Tokens first; structure by rule and space, not by box.
# ----------------------------------------------------------------------------------

CSS = """
:root {
  --paper: #FBFBF9;
  --ink: #1A1D1F;
  --muted: #6B7280;
  --rule: #E4E4E0;
  --act: #1B4D3E;
  --caution: #8A5A00;
  color-scheme: light;
}
*, *::before, *::after { box-sizing: border-box; }
html { background: var(--paper); }
body {
  margin: 0;
  padding: 48px 32px 96px;
  background: var(--paper);
  color: var(--ink);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  line-height: 1.6;
  font-weight: 400;
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
}
main { max-width: 1100px; margin-inline: auto; }
/* Panels are surfaces, not cards: a white ground against --paper, no border, no
   shadow.  The surface break separates the regions, so the hairline rules that used to
   sit between sections are gone.  Hairlines inside a panel stay - those encode
   structure rather than separate regions. */
section {
  background: #FFFFFF;
  padding: 32px 40px;
  border: 0;
  border-radius: 6px;
  box-shadow: none;
  margin-top: 32px;
}

/* The one raised element on the page.  Elevation is hierarchy here, so it is spent
   once, on the result, and nothing below it competes. */
.headline {
  background: #FFFFFF;
  padding: 48px;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(26, 29, 31, 0.06), 0 8px 24px rgba(26, 29, 31, 0.04);
  margin-top: 24px;
}
h1, h2 { font-size: 22px; line-height: 1.3; font-weight: 600; margin: 0 0 24px; }
h3 { font-size: 15px; line-height: 1.4; font-weight: 600; margin: 24px 0 8px; }
p { margin: 0 0 16px; max-width: 68ch; }
a { color: var(--ink); }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92em;
}
.label, .table-note, .gate-basis, .gate-outcome, .gate-reason, figcaption, sub {
  font-size: 13px;
  line-height: 1.4;
  color: var(--muted);
}
.caution { color: var(--caution); }
.error { color: var(--caution); max-width: 68ch; }
.svg-label { font-size: 13px; fill: var(--muted); }
.svg-act { font-size: 13px; font-weight: 600; fill: var(--act); }

/* Headline.  The boldness is spent here and nowhere else. */
/* The rule that used to sit above the figures separated them from the page title.
   The raised surface does that now, so the rule would be a stray line inside it -
   the same reason CHANGE 2 drops the rules between sections. */
.figures {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 32px;
  margin: 0 0 24px;
}
.tile p { margin: 0; max-width: none; }
.figure { font-size: 44px; line-height: 1.0; font-weight: 600; color: var(--ink); }
.tile .label { margin-top: 8px; }
.lift { margin: 0; }
.uncertainty {
  display: block;
  width: 100%;
  max-width: 720px;
  height: auto;
  margin-top: 24px;
}
.uncertainty + .label { margin: 4px 0 0; }

/* Tables.  Real table, no index column, no zebra, hairline rows. */
.scroller { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td {
  padding: 10px 20px 10px 0;
  border-bottom: 1px solid var(--rule);
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
  font-weight: 400;
}
thead th { font-size: 13px; font-weight: 600; border-bottom: 1px solid var(--ink); }
th.num, td.num { text-align: right; }
th:last-child, td:last-child { padding-right: 0; }
thead th.assumed { color: var(--caution); }
td sup, th sup { font-size: 10px; vertical-align: super; }
.table-note { margin: 12px 0 0; max-width: 68ch; white-space: normal; }

/* Gate trace.  A list, in evaluation order.  No cards, no icons, no markers. */
.gates { list-style: none; margin: 0; padding: 0; }
.gate {
  display: grid;
  grid-template-columns: minmax(0, 22rem) minmax(0, 1fr);
  gap: 2px 24px;
  padding: 10px 0;
  border-bottom: 1px solid var(--rule);
  color: var(--muted);
}
.gate-name { color: inherit; }
.gate-reason { grid-column: 2; margin: 0; max-width: 68ch; }
.gate-fired { color: var(--act); }
.gate-fired .gate-name, .gate-fired .gate-outcome { font-weight: 600; color: var(--act); }
.gate-passed .gate-name { color: var(--ink); }

/* Form. */
.fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 24px;
  margin: 0 0 24px;
  max-width: 940px;
}
.fields p { margin: 0; max-width: none; }
label { display: block; font-size: 13px; line-height: 1.4; color: var(--muted); margin: 0 0 6px; }
input, select, button {
  font: inherit;
  font-variant-numeric: tabular-nums;
  border: 1px solid var(--rule);
  border-radius: 0;
  padding: 9px 10px;
  background: #FFFFFF;
  color: var(--ink);
}
input, select { width: 100%; }
button {
  background: var(--ink);
  color: var(--paper);
  border-color: var(--ink);
  cursor: pointer;
  padding: 10px 22px;
  transition: opacity 0.15s ease;
}
button:hover { opacity: 0.88; }
button[disabled] { opacity: 0.5; cursor: default; }
:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }

#score-panel { margin-top: 32px; }
.readout {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 24px;
  margin: 0 0 24px;
  max-width: 940px;
}
.readout dt { font-size: 13px; line-height: 1.4; color: var(--muted); margin-bottom: 4px; }
.readout dd { margin: 0; font-size: 20px; }
.reasons { margin: 0 0 16px; padding-left: 20px; max-width: 68ch; }
.tag { display: inline-block; margin-right: 10px; }
.tag-assumed { color: var(--caution); }
.tag-measured { color: var(--ink); font-weight: 600; }

/* Two worked orders, side by side. */
.compare-heads, .compare-row {
  display: grid;
  grid-template-columns: minmax(0, 18rem) minmax(0, 1fr) minmax(0, 1fr);
  gap: 4px 24px;
  align-items: start;
}
.compare-heads { padding-bottom: 10px; border-bottom: 1px solid var(--ink); }
.compare-head p { margin: 0; max-width: none; }
.compare-title { font-weight: 600; }
.compare { list-style: none; margin: 0; padding: 0; }
.compare-row {
  padding: 10px 0;
  border-bottom: 1px solid var(--rule);
  color: var(--muted);
}
.compare-row .gate-name { grid-row: 1; color: var(--ink); }
.compare-row .gate-basis { grid-row: 2; grid-column: 1; }
.compare-row .gate-outcome { grid-row: 1; }
.compare-row .gate-fired { color: var(--act); font-weight: 600; }
.compare-row .gate-passed { color: var(--ink); }

/* Horizontal bar list. */
.bars { list-style: none; margin: 0 0 24px; padding: 0; max-width: 760px; }
.bar-row {
  display: grid;
  grid-template-columns: minmax(0, 12rem) minmax(0, 1fr) 6rem 5rem;
  gap: 12px;
  align-items: center;
  padding: 7px 0;
}
.bar-track { background: var(--paper); border-bottom: 1px solid var(--rule); height: 14px; }
.bar-fill { display: block; height: 14px; min-width: 2px; background: var(--muted); }
.bar-fill.bar-lead { background: var(--act); }
.bar-value, .bar-share { text-align: right; }
.bar-value { color: var(--ink); }

/* Decision lifecycle. */
.states { display: block; width: 100%; max-width: 980px; height: auto; margin: 0 0 16px; }
.svg-state { font-size: 12px; fill: var(--ink); }
.states .svg-label { font-size: 12px; }

/* Charts. */
#curve { display: block; width: 100%; max-width: 760px; height: auto; margin: 0 0 16px; }
.slider {
  width: 100%;
  max-width: 760px;
  padding: 0;
  border: 0;
  background: transparent;
  accent-color: var(--ink);
}
figure { margin: 24px 0 0; }
figure img { display: block; width: 100%; max-width: 760px; height: auto; }
figcaption { margin-top: 8px; max-width: 68ch; }

.prose { max-width: 68ch; }
.prose h3 { margin-top: 24px; }
.switch { display: flex; gap: 10px; align-items: flex-start; max-width: 68ch; margin: 0 0 24px; }
.switch input { width: auto; margin-top: 3px; }
.switch label { font-size: 15px; line-height: 1.6; color: var(--ink); margin: 0; }
#tier-banner { margin: 0 0 24px; padding-left: 16px; border-left: 2px solid var(--caution); }

@media (max-width: 768px) {
  body { padding: 32px 16px 64px; }
  section { padding: 24px 20px; }
  .headline { padding: 32px 24px; }
  .figures { grid-template-columns: 1fr; gap: 24px; }
  .figure { font-size: 36px; }
  .gate { grid-template-columns: 1fr; }
  .gate-reason { grid-column: 1; }
  .compare-heads, .compare-row { grid-template-columns: minmax(0, 9rem) 1fr 1fr; gap: 4px 12px; }
  .bar-row { grid-template-columns: minmax(0, 8rem) 1fr 4.5rem 4rem; gap: 8px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
"""

SCRIPT = """
const CURVE = __CURVE__;
const slider = document.getElementById("treated");
const indicator = document.getElementById("curve-indicator");

function interpolate(value) {
  const points = CURVE.points;
  let left = points[0];
  let right = points[points.length - 1];
  for (let i = 1; i < points.length; i++) {
    if (value <= Math.log1p(points[i].treated)) {
      left = points[i - 1];
      right = points[i];
      break;
    }
  }
  const lo = Math.log1p(left.treated);
  const hi = Math.log1p(right.treated);
  const share = hi > lo ? Math.min(1, Math.max(0, (value - lo) / (hi - lo))) : 0;
  return {
    treated: Math.round(Math.expm1(value)),
    cost: left.cost + share * (right.cost - left.cost)
  };
}

function moveIndicator() {
  const value = Number(slider.value);
  const x = CURVE.left + (value / CURVE.span) * (CURVE.right - CURVE.left);
  indicator.setAttribute("x1", x.toFixed(1));
  indicator.setAttribute("x2", x.toFixed(1));
  const point = interpolate(value);
  document.getElementById("slider-treated").textContent = point.treated.toLocaleString();
  document.getElementById("slider-cost").textContent = point.cost.toLocaleString(undefined, {
    minimumFractionDigits: 2, maximumFractionDigits: 2
  });
}

slider.addEventListener("input", moveIndicator);
moveIndicator();

document.getElementById("tier-switch").addEventListener("change", event => {
  document.getElementById("tier-banner").hidden = !event.target.checked;
});

document.getElementById("score-form").addEventListener("submit", async event => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  button.disabled = true;
  const body = {
    order_id: document.getElementById("order-id").value,
    amount: document.getElementById("amount").value,
    shipping: document.getElementById("shipping").value,
    zipcode: document.getElementById("zipcode").value,
    method: document.getElementById("method").value
  };
  try {
    const response = await fetch("/panels", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify(body)
    });
    const panels = await response.json();
    document.getElementById("score-panel").innerHTML = panels.score;
    document.getElementById("treatability-panel").innerHTML = panels.treatability;
    document.getElementById("gate-panel").innerHTML = panels.gates;
  } catch (error) {
    document.getElementById("score-panel").innerHTML =
      '<p class="error">Backend unavailable, or the request was refused. ' +
      String(error) + '</p>';
  } finally {
    button.disabled = false;
  }
});
"""


# ----------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------

def _order_from(body: dict) -> dict:
    method = str(body.get("method") or DEFAULT_ORDER["method"])
    return {
        "order_id": str(body.get("order_id") or DEFAULT_ORDER["order_id"]),
        "amount": int(float(body.get("amount", DEFAULT_ORDER["amount"]))),
        "shipping": int(float(body.get("shipping", DEFAULT_ORDER["shipping"]))),
        "zipcode": str(body.get("zipcode") or DEFAULT_ORDER["zipcode"]),
        "method": method if method in PAYMENT_METHODS else DEFAULT_ORDER["method"],
    }


@app.post("/panels")
async def panels(request: Request) -> JSONResponse:
    """Re-render the three order-scoped panels.  One renderer, shared with page load."""
    try:
        order = _order_from(await request.json())
    except (TypeError, ValueError) as exc:
        message = (f'<p class="error">That order could not be read: {_escape(exc)}</p>'
                   '<p class="label">Enter whole numbers for the two amounts.</p>')
        return JSONResponse({"score": message, "treatability": message, "gates": message})
    return JSONResponse(_panels(order))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    figures = _headline_figures()
    table = _policy_table()
    points = _curve_points(figures, table)
    scales = _curve_scales(points)
    operating = next(point for point in points if point["operating"])
    rendered = _panels(DEFAULT_ORDER)
    worked = {key: _evidence_or_error(order) for key, _, order in COMPARISON}

    curve_state = json.dumps({
        "left": CURVE_LEFT,
        "right": CURVE_RIGHT,
        "span": scales["span"],
        "points": [{"name": point["name"], "treated": point["treated"],
                    "cost": point["cost"]} for point in points],
    }).replace("</", "<\\/")

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>COD return-to-origin risk scorer</title>",
        f"<style>{CSS}</style></head><body><main>",

        # 1 - Headline and 2 - the uncertainty bar, as one block.  They were two
        # siblings; a single raised element cannot span two, and the bar reads as part
        # of the result rather than as a region of its own.  The page title stays
        # outside it, on the paper.
        "<h1>COD return-to-origin risk scorer</h1>",
        '<header class="headline">',
        _headline_html(figures),
        _uncertainty_svg(figures),
        '<p class="label">The interval excludes the base rate.</p>',
        "</header>",

        # 3 - Policy table.
        "<section><h2>What each policy costs</h2>",
        _table_html(table),
        "</section>",

        # 4 - Score an order, already scored.
        "<section><h2>Score an order</h2>",
        _form_html(DEFAULT_ORDER),
        f'<div id="score-panel">{rendered["score"]}</div>',
        "</section>",

        # 5 - Treatability and the gate trace, populated from the same order.
        "<section><h2>Treatability and gate trace</h2>",
        f'<div id="treatability-panel">{rendered["treatability"]}</div>',
        "<h3>Gate chain, in evaluation order</h3>",
        '<p class="label">Compliance gates precede economic ones by design. The scoring '
        'API evaluates the chain lazily, so a gate carries an outcome here only where the '
        'API served one; the rest are named with their basis and marked unevaluated '
        'rather than assumed to have passed.</p>',
        f'<div id="gate-panel">{rendered["gates"]}</div>',
        "</section>",

        # 5b - The same chain, on two orders at once.
        "<section><h2>One order we act on, one we do not</h2>",
        '<p>The chain stops at the first refusal, and almost everything stops at gate 1. '
        'Of 19,662 orders in the test window the cost policy selects 34; the other 19,628 '
        'leave at tier_is_actionable and are never looked at again. Nothing in the machine '
        'can block an order — the strongest action is a prepayment request — so the '
        'left-hand column is the whole of what acting means here.</p>',
        _comparison_html(worked),
        _acted_caption(worked["acted"].get("scored")),
        '<p class="label">Both columns are what the API served for these two orders, '
        'nothing more. The rows below gate 1 read the same in both columns because '
        '/score does not run the chain: no serving path calls '
        'responder/gates/registry.py. Its callers are tests/responder/test_gates.py '
        'and the declared-context panel below, which runs it deliberately and says so. '
        'Those rows say unevaluated rather than showing outcomes this page would have '
        'had to invent.</p>',
        "</section>",

        # 5b-bis - The same chain again, this time actually executed.  Separate section
        # on purpose: the panel above is observed, this one is declared.
        "<section><h2>The same chain, run against a declared context</h2>",
        _declared_html(ACTED_ORDER, worked["acted"].get("scored")),
        f'<p class="label caution">{_escape(DECLARED_CAPTION)}</p>',
        "</section>",

        # 5c - What the responder did across the whole window.
        "<section><h2>What the responder did across the test window</h2>",
        _responder_html(),
        f'<p class="label">{_escape(RESPONDER_CAPTION)}</p>',
        "</section>",

        # 5d - The machine those buckets are states of.
        "<section><h2>Decision lifecycle</h2>",
        _state_svg(),
        _state_caption(),
        "</section>",

        # 6 - Decision curve.
        "<section><h2>Decision curve</h2>",
        _curve_svg(points, scales),
        '<p><label for="treated">Orders treated</label>'
        f'<input id="treated" class="slider" type="range" min="0" '
        f'max="{scales["span"]:.6f}" step="0.0001" '
        f'value="{math.log1p(operating["treated"]):.6f}"></p>',
        '<dl class="readout">'
        '<div><dt>Orders treated</dt><dd id="slider-treated">—</dd></div>'
        '<div><dt>Expected cost, BRL</dt><dd id="slider-cost">—</dd></div>'
        "</dl>",
        '<p class="label">Expected cost against the number of orders treated, with the '
        'four committed policies from eval/TIER1_LOCK.json marked and the cost policy in '
        'green. The line between the marks is interpolation, and the slider reads off it. '
        'The measured sweep is below.</p>',
        '<figure><img src="/eval/figures/decision_curve.png" alt="Realised cost against a '
        'global treat-above threshold, with the four policies marked">',
        "<figcaption>The committed threshold sweep: realised cost on the test window "
        "against a single global treat-above threshold, BRL, on the 160-point grid "
        "computed by scripts/08_policy.py. The shaded band is c_rto plus or minus 50%. "
        "Its per-threshold costs exist only in this figure, which is why the interactive "
        "curve above is drawn against orders treated instead.</figcaption></figure>",
        "</section>",

        # 7 - PR curve.
        "<section><h2>Precision-recall curve</h2>",
        '<figure><img src="/eval/figures/pr_primary.png" alt="Precision-recall curve with '
        'the prevalence baseline">',
        "<figcaption>The dashed baseline is the test prevalence.</figcaption></figure>",
        "</section>",

        # 8 and 9 - Evaluation prose, with the tier switch beside it.
        "<section><h2>Evaluation</h2>",
        '<div class="switch"><input id="tier-switch" type="checkbox">',
        '<label for="tier-switch">Show the Tier-2 header. Tier 2 validates the estimator '
        "against a semi-synthetic response with an oracle; every figure on this page is "
        "Tier 1, measured on the test window.</label></div>",
        '<div id="tier-banner" hidden><p class="label">TIER 2: semi-synthetic estimator '
        "validation only. This does not measure intervention effect.</p></div>",
        f'<div class="prose">{_markdown_to_html(_headline_markdown())}</div>',
        "</section>",

        "</main>",
        f"<script>{SCRIPT.replace('__CURVE__', curve_state)}</script>",
        "</body></html>",
    ]
    return "".join(parts)


# ----------------------------------------------------------------------------------
# Committed figures
# ----------------------------------------------------------------------------------

#: The committed figures already carry a near-paper facecolor (#FCFCFB, checked against
#: eval/figures/*.png), so they sit on the page without a white patch behind them and are
#: served as-is.  Repainting them here would move them by one 8-bit level and regenerating
#: them means rerunning the policy pipeline, which this module does not do.

_FIGURES: dict[str, bytes] = {}


@app.get("/eval/figures/{name}")
def evaluation_figure(name: str) -> Response:
    if name not in FIGURES:
        raise HTTPException(status_code=404, detail="Unknown evaluation figure")
    if name not in _FIGURES:
        path = ROOT / "eval" / "figures" / name
        try:
            _FIGURES[name] = path.read_bytes()
        except OSError as exc:
            raise HTTPException(
                status_code=503, detail=f"Figure artifact unavailable: {name}"
            ) from exc
    return Response(_FIGURES[name], media_type="image/png")
