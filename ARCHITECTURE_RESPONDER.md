# Architecture: the responder

Three claims, stated before anything else, because everything below is either evidence for
one of them or a boundary around them.

> **The mechanism is built.** An idempotent inbox, an append-only decision log, an
> eleven-state machine declared once as data, fourteen ordered gates, deterministic
> templates, runtime trust tiers, a lease poller, a response endpoint and a sweeper. 192
> tests cover it.
>
> **The effect is unvalidated.** Olist carries no phone numbers, no email addresses and no
> consent records. Nothing was sent to anyone. No intervention effect is measured anywhere
> in this repo, and the effectiveness values the policy consumes are assumptions swept in a
> sensitivity table rather than measurements.
>
> **The learning path is instrumented.** Every decision writes its model version, policy
> version, cost-constants id, effectiveness-prior id, gate-set version, calibrated
> probability, threshold and reason class to an append-only log, and a deterministic 2%
> exploration slice is held out. That is what a future randomised readout would need. It is
> not a readout.

The status tag is `[BUILT-UNEVAL]` — implemented, unit-tested, and unevaluable on the
available data. The scoring path is [ARCHITECTURE_DETECTOR.md](ARCHITECTURE_DETECTOR.md).

---

## 1. The separation that makes it shippable

The thing that makes an auto-responder frightening is not that it sends messages. It is that
"decides to send" and "sends" are usually the same act, so a system that is wrong is wrong
*outwardly* and immediately.

This responder splits them.

**Proposing is pure. Committing is one function.** Gates and templates produce a
`CandidateAction` — a frozen dataclass carrying a decision id, a tier, a template id and
version, a message class, the rendered text, a field map of provenance-tagged values, and
the full gate trace. It is a **proposal**. Nothing about constructing it reaches the outside
world.

`responder/dispatch.py` is the entire outbound boundary:

```python
def dispatch(action: CandidateAction, channel: Channel) -> None:
    validate(action)
    channel.send(action)
```

Eleven lines including imports. Every message that could ever leave this system passes through
that one call, which means the audit surface for "what can this thing do to a customer?" is
a single file.

**Autonomy is therefore a property of the channel, not of the logic.** The gate chain does
not know whether it is live. Neither do the templates. Neither does the state machine. That
is what makes the system safe to run in a demo: there is no configuration flag that could be
set wrong, because there is no flag.

---

## 2. Inbox and outbox on one store

`responder/store/schema.sql` defines three tables in one SQLite file: `inbox`,
`decision_log`, `action_outbox`.

**Decision: the log is the queue.** State lives in `decision_log` as an append-only sequence
of rows; `action_outbox` holds only claim bookkeeping — a lease timestamp, an attempt count
and a terminal flag.

*Alternative considered:* a message broker, or a separate queue table holding message
payloads.
*Rejected because* a broker introduces a second source of truth about what a decision's
state is, and reconciling the two after a crash is a distributed-systems problem this system
does not need to have. With the log as the queue, "what happened to this decision" and "what
should happen next" are answered by the same rows, and there is nothing to reconcile.

**Append-only is enforced by the database, not by discipline.** Four triggers:

```sql
CREATE TRIGGER decision_log_no_update BEFORE UPDATE ON decision_log
BEGIN SELECT RAISE(ABORT, 'decision_log is append-only'); END;
```

and the same for `DELETE`, and both again for `inbox`. A future contributor who writes an
`UPDATE` gets an exception, not a silently rewritten audit trail. `tests/responder/
test_schema.py` asserts all four fire.

**Idempotency at the inbox.** `event_key` is the primary key. It is the
`x-razorpay-event-id` header when present, and otherwise a SHA-256 of the canonical body —
`json.dumps(body, sort_keys=True, separators=(",", ":"))`. Which derivation was used is
recorded in the decision's `state_reason`, so an operator can tell a header-keyed event from
a hash-keyed one in the log. A repeated event returns the stored response JSON verbatim
rather than re-scoring, which matters because scoring is not free and because a re-score
under a changed model would return a different answer to the same question.

**Decision: single writer, enforced by a process-level lock.** `ResponderIngest` holds a
`threading.Lock` around the read-check-insert sequence.

*Alternative considered:* rely on SQLite's own transactional guarantees and the primary-key
constraint.
*Rejected because* the check-then-score-then-insert sequence spans an expensive model call.
Two concurrent duplicates would both miss the inbox lookup, both score, and one would then
fail on the primary key — correct, but having paid for a wasted inference and having emitted
a `decision_log` row for a decision that never lands. The lock makes the whole sequence
atomic.

**This is honestly a single-instance design.** One SQLite file, WAL requested, a
process-level lock. Correct for one process and wrong for two, and §10 says what production
would change.

---

## 3. The state machine, declared once as data

Eleven states, six of them terminal, declared in `responder/states.py` as a dict of frozen
sets:

```python
TRANSITIONS = {
    State.SCORED:      {SUPPRESSED, HELD_EXPLORATION, QUEUED},
    State.QUEUED:      {SENT, ABANDONED},
    State.SENT:        {CONFIRMED, CANCELLED_AT_PROMPT, NO_RESPONSE,
                        ESCALATED, SEND_FAILED, ABANDONED},
    State.ESCALATED:   {SENT, ABANDONED},
    State.SEND_FAILED: {QUEUED, ABANDONED},
    **{state: frozenset() for state in TERMINAL},
}
```

**Decision: one declaration, and everything else derived from it.**

*Alternative considered:* transition logic expressed as conditionals in the writer, with a
diagram maintained alongside.
*Rejected because* a hand-maintained diagram is a document that can disagree with the code,
and this repo has been bitten repeatedly by exactly that. `scripts/92_render_state_diagram.py`
imports `TRANSITIONS` and `TERMINAL` and renders `responder/state_diagram.md` as mermaid, and
the suite rejects drift between the two. The picture cannot lie, because the picture is
generated from the thing it describes.

`responder/transitions.py` is the **sole writer**. It reads the current state as the highest
`seq` for the decision, refuses to move out of a terminal state, refuses any transition not
in `TRANSITIONS`, and appends a new row carrying every column of the previous one plus the
new state, reason, actor and timestamp. Nothing is updated in place; the decision's history
is the full sequence of its rows.

Two bounds are enforced by that writer rather than being configuration:

- `MAX_ESCALATIONS = 1` — a second escalation raises `TransitionError`.
- `MAX_SEND_ATTEMPTS = 3` — a fourth send does not raise; it is **rewritten to `ABANDONED`
  with the reason "send attempts exhausted"**. That asymmetry is deliberate: an escalation
  limit being hit is a caller error worth surfacing, while a send limit being hit is an
  expected terminal outcome that the machine should absorb.

`tests/responder/test_states.py` carries 115 tests — the transition matrix is small enough
to enumerate exhaustively, so it is enumerated exhaustively.

---

## 4. Propose vs commit, and why dry-run is the absence of a call

Three channels implement one protocol:

| Channel | `send` does |
|---|---|
| `DryRunChannel` | validates, returns |
| `RecordedChannel` | validates, appends to an in-memory list |
| `LiveChannel` | validates, raises `NotImplementedError` |

**Dry-run is not a branch anywhere in the system.** There is no `if dry_run:` in the gate
chain, in the templates, in the state machine or in dispatch. The dry-run channel is a
`send` that does nothing — the message is simply never transmitted, and every other line of
code runs identically.

*Alternative considered:* a `dry_run` flag threaded through the pipeline, checked before the
send.
*Rejected because* a flag is a thing that can be false when it should be true. A missing call
cannot be. And a flagged system exercises a *different code path* in dry-run than in live, so
the dry run stops being evidence about the live behaviour — which is the entire reason for
running it.

That is asserted, not merely intended.
`tests/responder/test_dispatch.py::test_no_dry_run_branch_in_gate_chain` parses every file
in `responder/gates/` and `responder/templates/` with `ast` and asserts no `Name` node
containing `dry_run` appears in either. A contributor who adds one fails the suite.

`LiveChannel` carries three independent refusals: it will not construct unless
`RESPONDER_LIVE=1`; it will not construct at all if the data source is Olist, whatever the
environment says; and its `send` raises because no transport is configured in this repo.
`tests/responder/test_dispatch.py::test_live_channel_refuses_olist` pins the second one,
which is the one that matters — it makes "we cannot accidentally message the evaluation
dataset" a property of the code rather than a promise.

---

## 5. The gate chain

Fourteen gates in `responder/gates/registry.py`, run in order, **short-circuiting at the
first refusal**. Each is one file, one function, a name and a version, returning a
`GateResult` carrying the gate name, its version, whether it passed, the action it implies
and a human reason.

| # | Gate | Basis | Refuses when |
|---:|---|---|---|
| 0 | `kill_switch` | operational | the operator switch is on |
| 1 | `tier_is_actionable` | policy | tier is `allow`, or the tier is not actionable |
| 2 | `already_terminal` | policy | the decision has reached a terminal state |
| 3 | `message_class_matches_tier` | policy | class is not `service` for `confirm`, or `promotional` otherwise |
| 4 | `dnd_scrub` | **TRAI** | a promotional message has not been DND-scrubbed |
| 5 | `consent_on_record` | **TRAI** | a promotional message has no consent timestamp |
| 6 | `opt_out_and_recontact_spacing` | **TRAI** | opt-out handling is absent or recontact spacing is not clear |
| 7 | `send_window` | **TRAI** | a promotional message falls outside the daylight window |
| 8 | `merchant_daily_budget` | operational | the merchant's daily budget is exhausted |
| 9 | `customer_fatigue` | operational | the customer's contact-frequency limit is reached |
| 10 | `exploration_slice` | policy | the decision lands in the held-out 2% |
| 11 | `effectiveness_below_impression_cost` | policy | `effectiveness × c_rto ≤ impression_cost` |
| 12 | `no_risk_disclosure` | **DPDP** | the rendered text leaks a score, a tier or a reason word |
| 13 | `assert_no_block_tier` | policy | the tier is `block` |

**The chain can only reduce an action; it can never raise one.** Every gate returns pass or
block. There is no gate that promotes `confirm` to `prepaid_only`, and there is no code path
that could add one without changing the `GateResult` type.

### Compliance before economics, and why that ordering is information

Gates 4 through 7 are TRAI. Gate 11 is the cost calculation. **4–7 run first**, so when a
compliance gate refuses, the economic gate is never consulted.

That is not an optimisation. It is a statement about what kind of system this is: **a legal
constraint is not an input to a cost function.** If the cost gate ran first, a sufficiently
expensive order could in principle be argued into a send outside the daylight window; the
ordering makes that argument unrepresentable. The demo makes it visible — change the declared
local time from 14:20 to 22:40 and the chain goes from 14 of 14 gates passing to gate 7
refusing with gates 8 through 13 marked "not reached".

### The TRAI/DPDP derivation, and the line the two tiers straddle

India's DLT framework distinguishes **service** messages, which relate to an existing
transaction, from **promotional** messages, which solicit. Promotional messages require
consent on record, DND scrubbing, an opt-out keyword, recontact spacing, and delivery inside
a daylight window; service messages are exempt from all of it.

**The two action tiers sit on opposite sides of that line.** A `confirm` message asks the
customer to verify a delivery address for an order they placed: service. A `prepaid_only`
message asks them to pay for it before dispatch: that solicits a payment, and it is
promotional.

So gates 4, 5, 6 and 7 all begin with the same clause —
`candidate.message_class == "service" or ...` — and pass unconditionally for a `confirm`.
**The tier the cost policy selects determines which regulatory regime the message lives
under.** That is why gate 3 exists and why it runs at position 3, before any of them: it
asserts that the message class and the tier agree, so a `prepaid_only` cannot be smuggled
through the compliance gates by being labelled a service message.

Gate 12 is DPDP's contribution and it is the strictest text rule in the system. It refuses
any rendered message that contains the reason class, the word "risk", any tier name, any
word from the decision's own reason vocabulary, or **any numeral at all** unless that numeral
is part of the customer's own address. A customer may be asked to confirm their address; they
may not be told that a model scored them.

Gate 10 is the exploration slice: `sha256(decision_id) % 10000 < 200`, a deterministic 2%.
Deterministic rather than random so a replay of the same decisions holds out the same slice,
which is what makes the held-out arm reconstructible after the fact.

Gate 13 is the last gate and asserts something that should be impossible: that the tier is
not `block`. `block` is not in `policy/costs.py::TIERS`, `tests/test_policy.py:196` asserts it
never will be, and this gate is the belt to that braces — the guarantee is stated at the point
of action as well as at the point of definition.

`tests/responder/test_gates.py` runs 24 tests, including one parametrised over
`range(14)` asserting **every** gate has both a passing and a blocking case, so no gate can be
added that is decorative.

---

## 6. Trust tiers at runtime

`responder/trust.py` is 62 lines and does one thing: it makes an unsupported claim
**unrenderable** rather than merely discouraged.

```python
class Tier(IntEnum):
    ASSUMED = 0
    MEASURED = 2
    HUMAN = 3
```

`Tagged(value, tier, source)` implements the arithmetic operators, and every operation takes
the **minimum** tier of its operands and concatenates their sources. Provenance is therefore
not something a developer remembers to carry — it propagates through the arithmetic
automatically, and one assumed input poisons the whole expression.

`render_measured` raises `UngroundedClaim` on anything below `MEASURED`.

**The load-bearing consequence:** a savings figure cannot be printed. Savings requires
`treated × failure_rate × c_rto × effectiveness`, and effectiveness is `ASSUMED`, so the
product is `ASSUMED`, so it raises.
`tests/responder/test_trust.py::test_savings_figure_is_unrenderable` constructs exactly that
expression — three measured factors and one assumed one — and asserts the raise.

*Alternative considered:* a documentation convention that assumed figures carry a label.
*Rejected because* a convention is followed until someone is in a hurry. The single most
tempting number this project could produce is "we would have saved X", it would be the most
persuasive line on any slide, and it would be unsupported. Making it throw an exception is
the only version of that rule that holds under deadline pressure.

`HUMAN = 3` exists for values a person has attested to, above measurement. Nothing in the
repo currently produces one; it is in the enum because the ladder is the design and leaving a
gap would misrepresent it. The gap at `1` is likewise deliberate — it leaves room for a tier
between assumed and measured (an externally-sourced figure, say) without renumbering.

---

## 7. Templates

`responder/templates/registry.py` builds a closed dictionary keyed on
`(ReasonClass, Tier)` — six reason classes by three tiers, **eighteen templates**, all
constructed at import. Each carries an id, a version, a message class, a declared tuple of
required field names, and a render function that raises `KeyError` naming the missing fields
if any are absent.

**Decision: no LLM anywhere in the outbound path.**

*Alternative considered:* generate message text from the reason and the order context.
*Rejected for three reasons, in order of weight.* First, gate 12 has to be able to prove
that no score, tier or reason word appears in the outgoing text; a generated string can only
be *checked* after the fact, and a check that runs after generation is a filter, not a
guarantee. Second, the messages are three sentences long and there are eighteen of them — the
entire output space is small enough to read, and eighteen committed golden fixtures in
`tests/responder/fixtures/rendered/` pin every one byte-for-byte. Third, a template has a
version that can be recorded in the decision log; a generated string does not, so a message
sent last month could not be reproduced.

The reason mapping is total, and enforced: `reason_class()` raises `ValueError` on any
detector reason it does not recognise, rather than falling back to a default. A new reason
string added upstream fails loudly at the boundary instead of quietly rendering the wrong
template.

---

## 8. Escalation, and its empirical justification

**Exactly one retry.** `MAX_ESCALATIONS = 1`, declared in `responder/states.py` and enforced
by the transition writer rather than being a runtime configuration value — so it is not
something an operator can raise under pressure.

The `Sweeper` closes response windows after 24 hours. Before a dispatch cutoff it escalates
once; after the cutoff, or on a decision that has already escalated, it moves to
`NO_RESPONSE`.

**The empirical basis, labelled for what it is.** An external vendor report describes roughly
7% of COD orders failing verification within 24 hours, splitting roughly 40% abandoned, 30%
fake or test, and 30% re-confirming on retry. That last third is the entire argument for
retrying at all, and the other two-thirds are the argument for retrying only once: for 70% of
non-responders a second attempt reaches someone who is not going to answer, and the impression
cost is paid regardless.

**This is an unverified vendor figure, not a result of this project**, and
`eval/responder/escalation.md` says so in those words. It supports one retry as a bounded
recovery attempt; it does not establish those percentages here, and no number in this repo
depends on them.

---

## 9. The replay harness, the denominator rule, and break-even

`responder/replay.py` runs the full policy over all 19,662 test-window orders and reports
what the responder would have done.

**The denominator rule: every count uses the complete test-window denominator.** Held and
suppressed rows appear in the tables. The report opens by stating the population and the
observed failure count before any outcome bucket.

*Alternative considered:* report on the treated set — 34 orders, precision 5.88%.
*Rejected because* reporting only the acted-on set is how an intervention system flatters
itself. 5.88% precision on 34 orders is a real number and a misleading headline; the honest
frame is that the system held 19,628 of 19,662 orders and acted on 34.

| Outcome bucket | Orders |
|---|---:|
| `confirm` | 12 |
| `prepaid_only` | 22 |
| held | 19,628 |

Targeting: 34 selected, 2 failures in the selected set, precision 5.88%, recall 1.30%.

**One caveat about that report.** Its "intervention rate by region proxy" table has a single
row reading `unknown | 34`, because `customer_state` is not in the shipped feature matrix and
the replay reads it from there. The table is vacuous and should be read as such; the real
regional breakdown is `eval/fairness.md`.

### Break-even effectiveness

`responder/breakeven.py` computes the fraction of treated failures that would have to be
prevented for the intervention to pay for itself:

```
break_even = (impression_cost + triggered_cost) / (treated_failures × c_rto)
```

**No effectiveness prior enters the calculation** — that is the entire point. Every input is
a `Tagged` value at `MEASURED`, so the result renders; had a prior entered, it would not.

| `c_rto` sweep | Required effectiveness |
|---:|---:|
| −50% | 11.9602% |
| base | **5.98008%** |
| +50% | 3.98672% |

The plot draws the external vendor bands beside that curve — 5–10 percentage points for OTP
alone, 20–40% for confirmation — as **comparison anchors that are not inputs to the
arithmetic**. A reader can see that the required effectiveness sits at the bottom of the
range vendors claim, and decide for themselves what to make of a vendor claim.

`eval/responder/policy_table.md` is the same quantity decomposed per policy, with every cell
carrying its trust tag inline — `19,662 [MEASURED]`, `9.60 [MEASURED]` — so the provenance is
visible in the artifact rather than only in the code that produced it.

**Two generators, two definitions, and they differ.** `eval/policy_table.py` and
`responder/policy_table.py` both produce a four-row table and they disagree: 81,655.37 against
81,194.01 on the "intervene on everything" triggered cost, and a break-even band of
4.22%–12.65% against a point estimate of 5.98%. The causes are identifiable rather than
mysterious — the `eval/` version multiplies triggered cost by the LTV multiplier and reports
bootstrap medians for the band, the `responder/` version does neither — and both are
internally consistent. The README quotes the `eval/` table. This is recorded rather than
reconciled, because reconciling them means choosing one definition and that choice has not
been made.

---

## 10. Honest status: what is wired and what is not

This section exists because the difference between "implemented" and "reached by the running
system" is exactly the thing a demo can hide.

**The gate chain does not run on the serving path.** `/score` goes to
`ResponderIngest.score`, which validates, scores, allocates a decision id, writes the inbox
row and the `decision_log` row, and returns. It never calls `registry.run`. The callers of
`registry.run` in this repo are `tests/responder/test_gates.py` and the demo page's
declared-context panel. Nothing else.

**So the demo shows two panels, and keeping them apart is the point.**

The first is *observed*: the two-order comparison shows what the API actually served. Gate 1
carries a real outcome — `tier_is_actionable` passes for the freight order and refuses for
the ordinary one, which is the `422 decision has no actionable tier` the API returns — and
gates 2 through 13 read **"not evaluated at score time"**. They are named with their bases and
marked unevaluated rather than shown as passing, because the page has no outcome for them and
inventing one would be the whole problem.

The second is *declared*: the same fourteen gates, the shipped `registry.run`, no
reimplementation, executed against a context that is written down rather than measured —
consent timestamp, DND status, local time, fatigue count, budget, opt-out, spacing. The panel
is captioned `[BUILT-UNEVAL]` and states plainly that Olist carries no consent timestamps, no
DND status and no send windows, so **the gate outcomes are real and the inputs are supplied.**

The candidate those gates run against is not fully invented either: its tier, reason class and
impression cost come from what the API actually served for that order, and the effectiveness
is read from `policy/effectiveness.py` so gate 11 tests against the same prior the cost policy
uses. Only the fields Olist cannot supply are declared.

**Nothing writes `action_outbox` in the shipped path.** The table exists, the `Poller` claims
and leases rows correctly, and `tests/responder/test_poller.py` covers it — but the only code
that inserts rows is the test suite and the standalone write-gate harness. The
inbox → decision_log half is wired to `/score`; the outbox → poller → dispatch half is built
and tested and not connected.

**The write gate measured a representative transaction, not the shipped one.**
`scripts/91_write_gate.py` defines its own two tables rather than importing
`responder/store/schema.sql`, and measures a two-row transaction against a baseline `/score`.
The finding it produced is worth more than the timing anyway: its pre-committed rule required
the measured baseline p99 to sit within ±15% of the locked 374.5ms, the measurement came back
at 72.9ms, and it recorded **"Comparability: STOP — host is not comparable to the locked
baseline"** rather than proceeding to a branch decision. A gate that refuses to fire on
incomparable data is doing its job.

**`eval/responder/gates.md` is a four-line scaffold** that says it is a scaffold. It has never
been populated. The gate evidence is the code, the 24 tests, and the live panel.

---

## 11. What would change for production

Ordered by how much they matter, not by effort.

1. **A real transport, and the consent substrate under it.** `LiveChannel.send` raises. Before
   it could stop raising, gates 4–7 need a real DND registry, a real consent store and a real
   recontact ledger — they currently read booleans off a `Context` that a caller supplies. The
   gates are the right shape; the data behind them does not exist.
2. **Wire the outbox.** `ResponderIngest.score` would enqueue an `action_outbox` row for any
   actionable tier, and a poller loop would claim, run the gate chain, render, dispatch and
   transition. That is the single change that turns this from a scoring service with a
   responder library into a responder.
3. **Move off single-instance SQLite.** The `threading.Lock` single-writer is correct for one
   process. Two processes need either a real queue or advisory locking, and the append-only
   triggers should survive whatever replaces the store.
4. **Authenticate.** No webhook signature verification, no rate limit, no tenant isolation
   beyond an `account_id` that is recorded and never checked.
5. **Read out the exploration slice.** 2% of decisions are held out deterministically and
   nothing analyses them. On real traffic with real outcomes that slice is the difference
   between the assumed effectiveness matrix and a measured one — and measuring it is what
   would let the two absent columns of the policy table be filled in honestly.
6. **Reconcile the two policy-table generators** (§9), by choosing which definition of
   triggered cost is correct rather than leaving both.

Until 1 and 5 are done, the effectiveness matrix stays assumed, the savings figure stays
unrenderable, and the second of the three claims at the top of this document stays exactly as
written.

---

## Cross-references

[README.md](README.md) · [ARCHITECTURE_DETECTOR.md](ARCHITECTURE_DETECTOR.md) ·
[BUILD_CHALLENGES.md](BUILD_CHALLENGES.md) · [ARCHITECTURE.md](ARCHITECTURE.md) §17 ·
[`eval/responder/`](eval/responder/) — `responder_replay`, `breakeven`, `policy_table`,
`write_gate`, `escalation`, `gates`
