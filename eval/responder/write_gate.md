# Responder write gate

- Generated (UTC): `2026-09-05 00:38:54`
- Population: test split, first 60 orders, same as `eval/latency.md`
- Protocol: 3 independent passes of 60 requests, warm caches; verdict uses the worst pass
- Service startup: 11.3s (excluded from request latency)
- SQLite: WAL mode requested; one `decision_log` row and one `action_outbox` row in one transaction per response

## Measurement

| Path | p50 | p99 | Worst-pass p50 |
|---|---:|---:|---:|
| baseline `/score` | 64.862ms | 72.924ms | 65.069ms |
| `/score` plus two-row transaction | 68.610ms | 104.689ms | 69.518ms |
| **delta** | **4.859ms** | **31.883ms** | — |

- Per-pass baseline p99: 73.056ms, 71.436ms, 73.986ms
- Per-pass write-path p99: 104.939ms, 91.499ms, 104.304ms
- Delta p99 is the maximum of the three paired per-pass p99 deltas.

## Pre-committed rule

The rule is: Δ p99 <20ms keeps synchronous atomicity; 20–50ms retries with `PRAGMA synchronous=NORMAL`; >50ms moves writes to a bounded in-process queue.

Locked baseline p99: **374.500ms**; measured baseline p99: **72.924ms**; allowed ±15%: **318.325–430.675ms**.

**Comparability: STOP.**

**Branch: STOP: host is not comparable to the locked baseline.**

## Post-wiring `/score` check

The actual wired endpoint was re-measured after T1.3 using the exact latency population
and protocol: the first 60 rows of the test split, one warmup request, three independent
passes of 60 requests, warm caches, and the worst pass as the verdict.

| Pass | p50 | p99 |
|---:|---:|---:|
| 1 | 105.338ms | 110.376ms |
| 2 | 105.239ms | 109.890ms |
| 3 | 105.417ms | 143.152ms |

**Worst-pass p99: 143.152ms — PASS against the 200ms budget.**

This confirms the wired `/score` path remains within budget, but it does not override the
earlier host-comparability STOP for selecting the transactional write design.
