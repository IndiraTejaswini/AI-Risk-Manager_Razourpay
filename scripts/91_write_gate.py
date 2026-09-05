#!/usr/bin/env python3
"""Measure the synchronous transactional-outbox write gate."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.contract import SUPPORTED_CURRENCY, ScoreRequest  # noqa: E402
from api.service import ScoringService  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "responder" / "write_gate.md"
N_REQUESTS = 60
N_PASSES = 3
LOCK_PATH = REPO_ROOT / "eval" / "TIER1_LOCK.json"
BASELINE_P99_TOLERANCE = 0.15


def _payload(svc: ScoringService, row) -> dict:
    items = svc.loader.load_table("items")
    items = items[items["order_id"] == row.order_id]
    pays = svc.loader.load_table("payments")
    pays = pays[pays["order_id"] == row.order_id]
    cust = svc.loader.load_table("customers")
    cust = cust[cust["customer_id"] == row.customer_id].iloc[0]
    ts = int(row.order_purchase_timestamp.timestamp())
    amount = int(round(float(pays["payment_value"].sum()) * 100))
    return {
        "entity": "event", "account_id": "acc_WRITE_GATE", "event": "order.created",
        "contains": ["order", "payment"],
        "created_at": ts,
        "payload": {
            "order": {"entity": {"id": row.order_id, "entity": "order",
                                 "amount": amount, "currency": SUPPORTED_CURRENCY,
                                 "created_at": ts}},
            "payment": {"entity": {"entity": "payment", "amount": amount,
                                   "currency": SUPPORTED_CURRENCY, "method": "boleto"}},
            "customer": {"entity": {"entity": "customer",
                                    "customer_reference": cust.customer_unique_id}},
            "shipping_address": {"entity": {"zipcode": str(cust.customer_zip_code_prefix)}},
            "line_items": [
                {"product_id": r.product_id, "seller_id": r.seller_id,
                 "amount": int(round(r.price * 100)), "quantity": 1,
                 "shipping_amount": int(round(r.freight_value * 100))}
                for r in items.itertuples()
            ],
        },
    }


def _database() -> tuple[sqlite3.Connection, Path]:
    handle, name = tempfile.mkstemp(prefix="rto-write-gate-", suffix=".sqlite3")
    os.close(handle)
    path = Path(name)
    connection = sqlite3.connect(path)
    journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if str(journal_mode).lower() != "wal":
        raise RuntimeError(f"expected SQLite WAL setup, got {journal_mode!r}")
    connection.executescript(
        """
        CREATE TABLE decision_log (
            decision_id INTEGER PRIMARY KEY,
            order_id TEXT NOT NULL,
            tier TEXT NOT NULL,
            risk REAL NOT NULL,
            model_version TEXT NOT NULL
        );
        CREATE TABLE action_outbox (
            outbox_id INTEGER PRIMARY KEY,
            decision_id INTEGER NOT NULL,
            order_id TEXT NOT NULL,
            action TEXT NOT NULL,
            status TEXT NOT NULL
        );
        """
    )
    return connection, path


def _write_rows(connection: sqlite3.Connection, order_id: str, response: dict) -> None:
    with connection:
        cursor = connection.execute(
            "INSERT INTO decision_log(order_id, tier, risk, model_version) "
            "VALUES (?, ?, ?, ?)",
            (order_id, response["tier"], response["risk"], response["model_version"]),
        )
        connection.execute(
            "INSERT INTO action_outbox(decision_id, order_id, action, status) "
            "VALUES (?, ?, ?, ?)",
            (cursor.lastrowid, order_id, response["tier"], "pending"),
        )


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def _measure(
    svc: ScoringService, payloads: list[dict], *, write: bool
) -> tuple[float, float, list[float], list[float]]:
    database = _database() if write else None
    connection = database[0] if database else None
    totals: list[float] = []
    pass_p50: list[float] = []
    pass_p99: list[float] = []
    try:
        for _ in range(N_PASSES):
            current: list[float] = []
            for raw in payloads:
                req = ScoreRequest.model_validate(raw)
                started = time.perf_counter()
                response, _ = svc.score(req, raw)
                if write:
                    assert connection is not None
                    _write_rows(connection, raw["payload"]["order"]["entity"]["id"],
                                response.model_dump())
                current.append((time.perf_counter() - started) * 1000)
            totals.extend(current)
            pass_p50.append(_percentile(current, 50))
            pass_p99.append(_percentile(current, 99))
    finally:
        if connection is not None:
            connection.close()
        if database is not None:
            database[1].unlink(missing_ok=True)
    return _percentile(totals, 50), _percentile(totals, 99), pass_p50, pass_p99


def main() -> int:
    started = time.perf_counter()
    svc = ScoringService()
    startup_s = time.perf_counter() - started
    risk = svc.loader.risk_set().join(svc.loader.split_labelled()["split"], how="left")
    test_rows = risk[risk["split"] == "test"].reset_index(drop=True)
    payloads = [_payload(svc, test_rows.iloc[i]) for i in range(N_REQUESTS)]
    svc.score(ScoreRequest.model_validate(payloads[0]), payloads[0])

    baseline_p50, baseline_p99, baseline_passes, baseline_p99_passes = _measure(
        svc, payloads, write=False
    )
    write_p50, write_p99, write_passes, write_p99_passes = _measure(
        svc, payloads, write=True
    )
    delta_p50 = max(
        write_p50_pass - baseline_p50_pass
        for write_p50_pass, baseline_p50_pass in zip(write_passes, baseline_passes)
    )
    delta_p99 = max(
        write_p99_pass - baseline_p99_pass
        for write_p99_pass, baseline_p99_pass in zip(write_p99_passes, baseline_p99_passes)
    )

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked_p99 = float(lock["latency"]["p99_ms"])
    comparable = abs(baseline_p99 - locked_p99) <= locked_p99 * BASELINE_P99_TOLERANCE
    verdict = "STOP: host is not comparable to the locked baseline" if not comparable else (
        "synchronous write inside the transaction" if delta_p99 < 20 else
        "retry with PRAGMA synchronous=NORMAL" if delta_p99 <= 50 else
        "do not put the write in the request path; use an in-process bounded queue"
    )

    lines = [
        "# Responder write gate",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`",
        f"- Population: test split, first {N_REQUESTS} orders, same as `eval/latency.md`",
        f"- Protocol: {N_PASSES} independent passes of {N_REQUESTS} requests, warm caches; "
        "verdict uses the worst pass",
        f"- Service startup: {startup_s:.1f}s (excluded from request latency)",
        "- SQLite: WAL mode requested; one `decision_log` row and one `action_outbox` "
        "row in one transaction per response",
        "",
        "## Measurement",
        "",
        "| Path | p50 | p99 | Worst-pass p50 |",
        "|---|---:|---:|---:|",
        f"| baseline `/score` | {baseline_p50:.3f}ms | {baseline_p99:.3f}ms | "
        f"{max(baseline_passes):.3f}ms |",
        f"| `/score` plus two-row transaction | {write_p50:.3f}ms | {write_p99:.3f}ms | "
        f"{max(write_passes):.3f}ms |",
        f"| **delta** | **{delta_p50:.3f}ms** | **{delta_p99:.3f}ms** | — |",
        "",
        f"- Per-pass baseline p99: {', '.join(f'{value:.3f}ms' for value in baseline_p99_passes)}",
        f"- Per-pass write-path p99: {', '.join(f'{value:.3f}ms' for value in write_p99_passes)}",
        "- Delta p99 is the maximum of the three paired per-pass p99 deltas.",
        "",
        "## Pre-committed rule",
        "",
        "The rule is: Δ p99 <20ms keeps synchronous atomicity; 20–50ms retries with "
        "`PRAGMA synchronous=NORMAL`; >50ms moves writes to a bounded in-process queue.",
        "",
        f"Locked baseline p99: **{locked_p99:.3f}ms**; measured baseline p99: "
        f"**{baseline_p99:.3f}ms**; allowed ±15%: "
        f"**{locked_p99 * (1 - BASELINE_P99_TOLERANCE):.3f}–"
        f"{locked_p99 * (1 + BASELINE_P99_TOLERANCE):.3f}ms**.",
        "",
        f"**Comparability: {'PASS' if comparable else 'STOP'}.**",
        "",
        f"**Branch: {verdict}.**",
        "",
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
