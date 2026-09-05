# What broke, how we found it, how we fixed it

Eleven incidents, ordered by how much they are worth knowing rather than by when they
happened.

A pattern runs through them and it is worth stating before the list: **the test suite was
green while every one of these defects was live.** None was surfaced by a failing test. What
surfaced them was re-running something and comparing it against what had been written down.
The last section is about what that changed.

---

## 1. A cross-validation fold generator that did not tile its index range

### What broke

`scripts/07_calibrate.py::_time_folds` returned, for fold `j`:

```python
np.arange(n)[i::1][j * n // k : (j + 1) * n // k]        # with i == j
```

`[i::1]` re-slices a progressively shorter array *before* fold `j`'s own slice is applied. So
every fold after the first was shifted by its own index and dropped its boundary element,
leaving exactly `k - 1` indices unassigned — at `n/k`, `2n/k+1`, `3n/k+2`, `4n/k+3` for
`k = 5`. At n = 7,380, k = 5 the orphans were `[1476, 2953, 4430, 5907]`.

`_crossfit_brier` allocated its output with `np.empty` and never wrote those slots. **A
model-selection metric therefore averaged uninitialised memory.** The function's docstring
said "No randomness." The randomness was the allocator's.

### The point: the harm distribution has no middle

This is the part worth carrying away, and it is not the maximum.

- On the **committed artifacts**, the garbage moved the 30-day slice Brier by **2e-7**.
  Invisible in a diff. A reviewer comparing the regenerated report against the committed one
  would have seen a number that agreed to six decimal places and moved on.
- On **one run**, a slot held approximately **1e73**, the Brier came back as **1e+143**, and
  the calibration-window comparison flipped from 30 days to 60 — changing the reported
  calibrator, its probability ceiling (6.86% → 13.84%), its smECE (0.0025 → 0.0054) and its
  top-decile calibration gap (0.42% → 1.27%).

**Three consecutive runs of unchanged code produced three different selections.** There is
nothing in between those two outcomes: the garbage is either small enough to look completely
plausible or large enough to be absurd, and it is plausible far more often than it is absurd.
**A reviewer diffing artifacts would have seen nothing wrong nearly every time.**

### How it was caught

By **re-running the pipeline three times** to check that the Tier-1 lock had not frozen stale
numbers, and getting three different answers.

**Not by the test suite, which was green at 315 tests throughout.** That figure was verified
for this document rather than recalled: a worktree at the parent of the fix commit collects
exactly 315 tests. No test asserted that the folds tile.

### The fix

- `_time_folds` now tiles `range(n)` exactly — contiguous, ascending, disjoint, for divisible
  and non-divisible `n` alike — with explicit `ValueError` for `k < 2` and `k > n` rather than
  silently returning empty folds.
- `np.empty` became `np.full(nan)`, with a raise before the return naming the count of
  unpredicted rows. **Kept even though the tiling fix makes it unreachable**, because
  "unreachable by current reasoning" is precisely what the previous version assumed.
- The docstring now states the guarantees that are asserted, and records that the previous
  claim was false.

`tests/test_calibration_folds.py` is 50 tests. Verified against the old implementation, **34
of the 50 fail**, with the tiling assertion reporting `missing [1476, 2953, 4430, 5907]`.

### What it changed

Scope was selection only — the shipped Platt calibrator is fit on the full selected window and
the cross-fit output never enters its parameters, though it does select the fit population.
`metadata.model_version` did not move.

But it did not move *by design*, it moved **by disconnection**: `api/service.py` refits its own
calibrator against a hardcoded 30-day window and never reads the selection at all. That
divergence is recorded rather than being taken as reassurance.

---

## 2. The companion lesson: coverage that does not match a defect's shape

Incident 1 has a second half that is more useful than the first, and it happened **inside the
fix**.

When the regression test was written, it got two kinds of assertion:

| Assertion | Against the broken code |
|---|---|
| **Structural** — the folds tile `range(n)`; each fold is a contiguous ascending run | **Caught it on all 14 parametrisations** |
| **Behavioural** — no NaN survives `_crossfit_brier`; the Brier lands in [0, 1] | **Passed** |

The behavioural assertions did not merely happen to pass. **They could not have failed.**
`np.empty` yields whatever was in that memory — garbage, not NaN — so a no-NaN assertion had
nothing to catch, and a Brier assembled from plausible garbage lands on the probability scale
most of the time. Those two tests were testing the allocator's mood.

The fourteen parametrisations are `SIZES = [7380, 14148, 21329, 100, 101, 103, 7]` crossed
with `FOLD_COUNTS = [3, 5]` — the three real window populations, plus small non-divisible
sizes where an off-by-one in the boundary arithmetic has nowhere to hide.

### The same lesson from the opposite direction

Incident 8's store self-check used a **fixed 40-order stride**. It passed, for months, against
two genuinely defective implementations:

- an exclusive prior sum that disagreed with the serving index on **1,169 of 97,658 rows**
- a tie-block collapse that was wrong on **534**

**Neither defect's rows intersected the stride.** That is not luck that ran out — it is what a
fixed stride does to a defect whose affected rows are determined by the data rather than by
position. A check with 0.04% coverage finds a 1% defect essentially never, and reports PASS
with total confidence. The test suite's own 75-order stride missed both as well.

Both defects were found **the first time the check ran over the whole population.**

### One principle, two instances

In incident 8 a *sample* did not cover the population. In incident 1 an *assertion* did not
cover the failure mode. **A check whose coverage does not match the defect's shape will
systematically miss it, and will look green while doing so.**

The operational rule that came out of it: prefer assertions on structure over assertions on
output wherever the structure is what the code claims. The tiling assertion is a statement
about the function's contract. The no-NaN assertion is a statement about one run's luck.

---

## 3. A study quoted at parameters that were never run

### What broke

`eval/significance.md` §2 reported the false-positive rate of the two chance comparisons under
a heading reading **"Measured, not asserted."** Four figures in that section were literals.
Nothing in the repository computed them.

The study itself was real and still present —
`tests/test_significance.py::test_calibration_of_the_two_chance_comparisons` — but it ran
**120 draws at n = 3,000** while the report described **200 draws at n = 4,000**, and it
asserted only structure: that the bootstrap rejection rate exceeds the permutation rate, and
that the permutation rate falls inside a deliberately wide band. It never asserted the values.

**So no run of this repo ever reproduced the numbers on that page.**

### Why this is a third category

It is not a figure contradicting a nearby table — there was no table to contradict. It is not
a fabricated figure with no generator — the generator existed and could be pointed at. It is a
**real measurement quoted at parameters that were never executed**, and that combination
defeats both of the obvious searches:

- **Undetectable by diffing.** Regeneration reproduced the figures *exactly*, because the
  literal was the source.
- **Undetectable by grepping for a missing generator.** The study was right there, under a
  named test, with a docstring explaining the finding.

Only running it at the stated parameters could surface it.

### And it refuted the numbers

Computed at 200 draws and n = 4,000:

| | Reported | Computed | Change |
|---|---:|---:|---|
| Paired bootstrap vs prevalence baseline | 0.100 | **0.145** | **+45%** |
| Permutation null | 0.045 | **0.055** | +22% |
| Over-rejection factor | "twice as often" | **2.9×** | |

**Wrong in the direction that understated the repo's own finding.** The prevalence baseline
over-rejects nearly three times its nominal rate, not twice. The qualitative conclusion
survives and is sharper.

That matters for what the fix should have been. **Deleting the section — the other available
option, and the one that would have looked most conservative — would have destroyed a correct
and stronger result.** The honest move on discovering an unsupported claim is to compute it,
not to remove it, because the claim may well have been true and understated.

### The fix

`scripts/05_significance.py` now runs the study at the parameters the report describes, as
named constants — `NOISE_DRAWS = 200`, `NOISE_N = 4_000` — and the report interpolates from
its return value. The test **keeps its cheaper parameters and its structural assertions**, and
the file comment says why: the test is a fast guard on the qualitative claim, and the
generator is the measurement the report quotes. Those are two different jobs and conflating
them is what produced the defect.

---

## 4. A latency claim that does not reproduce

### What broke

Nothing, exactly — which is the difficulty. The measurement is real and it will not sit still.

Five captures produced **p50 between roughly 90ms and 325ms on unchanged code at the same
SHA.** Four measurements survive in the committed artifacts today, all of the same endpoint on
the same machine:

| Artifact | p50 |
|---|---:|
| `eval/TIER1_LOCK.json` | 322.1ms |
| `eval/latency.md` | 125.3ms |
| `eval/responder/write_gate.md`, baseline | 64.9ms |
| `eval/responder/write_gate.md`, post-wiring | 105.3ms |

### The diagnosis, and why the protocol did not help

The protocol was worst-of-three-passes, on the reasonable theory that a budget met only when
the machine is quiet is not met.

**But the three passes agree with each other.** One run's three passes landed at 317.8, 326.4
and 321.2ms — within 1.3% of the mean. Another run's three sat at 125.0, 125.4 and 125.4ms —
within 0.2%. **The instability is between process invocations, not within them**, so
worst-of-three does not sample the thing that actually varies. Averaging more requests would
not have found it. Only re-running the whole measurement did.

The consequence is sharper than "the number is noisy": **a single run cannot establish which
regime the host is in.** Whichever run gets written down looks internally consistent and
tightly bounded, and carries a verdict.

The write gate is the cleanest evidence, because it was built to check. Its pre-committed rule
required the measured baseline p99 to sit within ±15% of the locked 374.5ms. The measurement
came back at 72.9ms, and it recorded **"Comparability: STOP — host is not comparable to the
locked baseline"** rather than proceeding to a design decision. A gate refusing to fire on
incomparable data is the protocol working.

### The fix

**No verdict is quoted anywhere outside the artifact that measured it, and the range is
reported with its cause.** The headline block generator states latency's absence explicitly
rather than silently omitting it:

> Latency is deliberately absent: it is a wall-clock measurement whose spread across runs
> exceeds the quantity being reported.

And latency was dropped from the lock's assertions, because **an intermittently-red lock is
worse than no lock — people learn to ignore it.**

### The part that was not actually finished, found while writing this document

The capture script's docstring says three fields are "written for the record and excluded from
comparison": `metadata.captured_at`, `metadata.git_sha`, and `latency.*`.

**`tests/test_tier1_lock.py` excluded only the first.** The generator's stated contract and the
assertion enforcing it disagreed, and both of the other two fields had already moved — HEAD had
advanced past the recorded SHA, and `eval/latency.md` was regenerated one commit after the lock
was captured, taking the pooled p50 from 322.1 to 125.3ms. Running the lock's own
`read_latency()` against the committed markdown returns 125.3 / 140.6 / 125.4. So
`RUN_TIER1_LOCK=1` failed on two fields the generator says are not evidence of anything
changing.

Fixed: `UNLOCKED_FIELDS` now implements all three exclusions, with the docstring's reasoning
quoted at the assertion rather than living only in the generator. What the lock exists for —
the evaluation numbers and the three integrity digests — is still compared exactly at 1e-9.

**The lesson is narrower than the others and worth having anyway:** a documented exclusion is
not an exclusion. It has to be in the assertion.

---

## 5. Prose hardcoded beside its own computed table

### What broke

Two generators hard-coded figures into sentences sitting beside tables that computed the same
quantity. Ten figures across eight lines:

| Location | Typed | Computed |
|---|---|---|
| `05_significance.py:122` | +0.0125 | **+0.0093** |
| `05_significance.py:312` | 1.73% / 11.04% | **1.42% / 9.09%** |
| `08_policy.py:547` | 26 | **34** |
| `08_policy.py:566` | 123 | **144** |
| `08_policy.py:576` | 500 | **629** |

### The one that explains how they survive review

`05_significance.py:122` opened the significance document by stating an AP gap of +0.0125
where its own tables said +0.0093 — **and interpolated the minimum detectable difference
correctly, by f-string, in the same sentence.**

Computed and typed, one line apart. That is how it survived: **the line looks generated.** A
reviewer scanning for hardcoded figures sees an f-string and moves on, because the sentence
carries the visual signature of being derived.

### And a second way they hide: numeric-format mismatch defeats grep

A README bullet quoted rejection rates of "10.0%" where the script wrote `0.100`. Searching the
repository for the README's figure finds nothing in the code, and searching the code's figure
finds nothing in the README. **The same number written two ways is invisible to text search**,
which is why the audit that found these had to be done by reading generators rather than by
grepping for constants.

### The fix

All ten now interpolate from the same structures their tables are rendered from.
`08_policy.py::_cell()` reads the sweep list the table is built from rather than recomputing,
so the sentence cannot drift from its own table again — the failure mode is designed out
rather than corrected.

### It happened again this week, and was found by rendering the page

While verifying claims for this document, the demo page was rendered and its comparison panel
read:

```
One we act on    Tier confirm   risk 0.012515, threshold 0.008244
```

Two lines below it, the caption read:

> ...drops the Elkan threshold to 0.008244, under this order's **0.012452**.

Hardcoded, beside its own live panel, disagreeing. A third value — **0.011134** — sat in the
module docstring describing the same order. Three figures for one quantity, none of them
reproducible, because the risk **is not a constant**: `purchase_hour`, `purchase_dow`,
`purchase_month` and `purchase_is_weekend` derive from the payload's `created_at`, and the demo
stamps that with the wall clock. Three live scores of the same order returned 0.012515, 0.012802
and 0.013055.

The threshold, 0.008244, is exact and stable — it depends only on the order's value and freight —
which is precisely why it was safe to write down and the risk was not.

Fixed: `_acted_caption()` interpolates from what the API just served, and the docstring records
that the risk is timestamp-dependent and must not be quoted.

### And a 100× unit error between two artifacts in the same directory

`responder/breakeven.py` rendered its table by appending a literal `%` to a bare fraction, so a
break-even effectiveness of 0.0598 printed as **"0.0598%"** instead of 5.98%. The plot in the
same script had always multiplied by 100, and `responder/policy_table.py` printed the same
quantity with a `.2%` format spec as **5.98%**.

So `eval/responder/breakeven.md` and `eval/responder/policy_table.md` — two committed artifacts
in the same directory, describing the same measurement — disagreed by exactly two orders of
magnitude, and the figure inside `breakeven.md` disagreed with the figure it sits above. Fixed
by scaling to percent before rendering rather than after.

---

## 6. A selection that was a comparison

### What broke

`eval/calibration.md` §2 selected the 30-day calibration window on cross-fitted Brier score.
The margin over the 60-day candidate was **+0.0000077**. The 95% confidence interval on that
difference was **[−0.0000206, +0.0000331]** — roughly **seven times wider than the margin, and
straddling zero.** The 90-day comparison was the same shape.

Selecting on the fifth decimal of a scalar without checking whether the fifth decimal means
anything is reading noise.

### The fix, and what it did not change

The candidates are statistically indistinguishable, so **the word "selection" was wrong** and
the section now says comparison. The pre-registered prevalence-drift argument — written in
`eval/calibration_window.md` before any Brier number existed — carries the decision, and the
Brier table demotes to a tie-break.

`CALIBRATION_WINDOW_DAYS = 30` is a fixed choice hardcoded in five places and **nothing
downstream reads that table.** The comparison exists to show the choice is *not load-bearing* —
that the system would report materially the same thing at 60 or 90 days — rather than to make
it. That no code path consumes its output is the evidence that it is a comparison.

**The fix did not change the conclusion. It changed whether the conclusion was reliable.**
That distinction is the whole value of the incident: nothing about the shipped system moved,
and a claim that could not have been defended under questioning became one that can.

There is a second finding underneath it. The design says to select the window by Brier on the
validation window; taken literally, that is structurally biased, because validation *is* 90
days and the 90-day candidate is therefore evaluated at home. The literal reading selects 90d —
which is worse on test Brier and roughly twice as over-predicted in the decile where the system
acts. **The rule as written picks the wrong window, for a structural reason that recurs on any
dataset where the calibration candidates are nested inside the evaluation window.**

---

## 7. A limitation that was overstated in a source comment

### What broke

The Elkan policy treats 34 of 19,662 test-window orders, selected on the **batch feature
matrix**, which carries customer-history, pincode-history, seller and geolocation joins. A
webhook payload carries five fields, so the serving path scores those orders with materially
less information. That much is real and remains a stated limitation.

**How it was recorded was not.** A comment in `ui/app.py` documented the gap as:

> A webhook payload carries five fields, so `/score` reports `features_missing: 4` and every
> one of those 34 comes back `allow` through the API — their calibrated risk drops from ~0.05
> to 0.0104 once the joined features are gone.

Confident, specific, and load-bearing: it justified why the demo picks the order it picks, and
it had been copied into a limitations section.

### How it was caught

**By re-measuring it while writing the submission documents**, under a rule that every
behavioural claim has to be verified against the code at write time rather than inherited.
Resolving the 34 truncated order ids in `eval/reasons.md` §8 back to full ids, rebuilding each
order's payload from its own items and its customer's real pincode, and posting all 34 to the
running service takes about a minute.

### What it actually does

| | Batch path | Payload path |
|---|---|---|
| Actionable | 34 | **24** — 18 `prepaid_only`, 6 `confirm` |
| `allow` | 0 | **10** |
| Calibrated risk range | 0.81% – 5.25% | 1.15% – 8.11% |
| Distinct risk values | — | 30 of 34 |
| `features_missing` | — | 4 on 32 orders, **7** on two |

**Every clause of the comment is wrong.** Not all 34 come back `allow` — 24 stay actionable,
roughly seven decisions in ten surviving the transition. The risks are not identical — there
are 30 distinct values. And they do not *drop*: the payload path's risk is frequently
**higher**, with one order moving from 4.24% on the batch path to 8.11% through the API. The
disagreement runs in both directions, including one order that moves from `prepaid_only` on
the batch path to `confirm` on the payload path — a tier change that is neither an upgrade nor
a downgrade in the direction the comment described.

### Why this is incident 3 again

Incident 3 was a real study quoted at parameters that were never run, and computing it at the
stated parameters **refuted the numbers while strengthening the conclusion**. This is the same
shape: a real limitation, described in terms nobody had executed, and re-measuring it produced
a smaller and sharper finding than the one on record.

The two are worth reading together because they suggest the same rule. **A claim about
behaviour is not verified by being plausible, by being specific, or by having been written by
whoever built the thing.** It is verified by running it. The comment was written by the person
who had just scanned the test window, and it was still wrong.

### What is fixed and what is not

The comment is corrected, and the README and detector architecture now carry the measured
table rather than the remembered one.

**The gap itself is not closed**, and correcting the description does not make it smaller in
the way that matters: ten of thirty-four decisions still flip between the two paths, and no
number in this repo reconciles them. Closing it means either enriching the payload contract or
re-deriving the policy on payload-only features, and both move every number downstream.

One further detail surfaced by the same check, recorded because it affects how the demo should
be read: the demo's acted-on order carries **pincode 14409**, which is the *default* order's
pincode, not its own — the real customer zip prefix for that order is 31330. Value, freight and
therefore the entire threshold argument are the order's own and are unaffected, since the
Elkan threshold depends only on value and freight and measures 0.008243624 either way. But the
payload is not a faithful reconstruction of that order, and a panel that labels it with the
real order id should say so.

---

## 8. A summation disagreement between two evaluation strategies

### What broke

The history store answers a prior-window query by summing a group's slice; the batch row-scan
accumulates the same values sequentially through pandas. **NumPy sums pairwise.** The two
disagree in the last bits once a group is large.

Customer groups have a median of one prior order and never exposed it. Sellers have thousands,
and `seller_prior_dispatch_window_mean` failed the startup self-check the first time it ran.

### How it was caught

The startup store-vs-row-scan self-check, which requires all 35 features to be bit-identical on
every order in the historical population — the same check whose 40-order stride is incident 2's
other half.

### The fix, and why the ordering of its consequences matters

Precomputing each group's cumulative sum — the same values, accumulated in the same order as the
scan — makes the two paths **bit-identical** rather than nearly so. It also turns an
`O(prior rows)` query into an `O(1)` lookup.

**The optimisation was a correctness fix. The speedup was the side effect**, and that ordering
is not modesty, it is the accurate description: the change was made to make two evaluation
strategies agree, and it happened to be several times faster. Stating it the other way round
would claim a performance engineering result that nobody set out to achieve.

Three further things came out of it, none of which were the point:

**It changed the shipped model.** `cumsum − own_value` does not recover the previous cumsum
entry when the cumsum is Kahan-compensated, as pandas' is. That moved `cust_prior_avg_value` on
**1,169 of 97,658 rows by at most 2.3e-13** — and that was enough to flip LightGBM split
thresholds. Average precision, ROC-AUC and the boosting round count were all unchanged, so the
*ranking* is identical; the model bytes are not, and the calibrated ceiling moved from 9.648% to
9.428% and the treated set from 35 orders to 34.

**Reverting was not the safe option.** The store returns a precomputed per-group cumulative sum,
so the only value it can produce is the previous entry. The subtraction form is not reproducible
from a prior-window index at all — **it was never compatible with the serving path**, and the
exactness checks had been passing on luck.

**Widening the check found a second, larger defect.** Making the self-check exact rather than
sampled meant it ran over the entire historical population for the first time in the repo's
history, which surfaced the `GroupBy.first` null-skip: **534 of 97,658 orders** read a
same-instant order as "the previous order" and reported `cust_days_since_prior_order = 0.0`
where the true answer is "no prior order". NaN and 0.0 are not close substitutes to a tree
booster — one is a native missing-value branch, the other a real split point — so this was a
structural change and it moved test AP from 0.0203 to 0.0171, ROC-AUC from 0.6384 to 0.6450,
and the boosting rounds from 33 to 50. Every downstream artifact was regenerated a second time.

---

## 9. A join whose cardinality encoded the label

### What broke

767 matured orders have no rows in `olist_order_items_dataset.csv`, and **100% of them are
`label_a` positives** — 33.65× the panel rate. An order that resolves to `unavailable` has no
items in the export, so the *absence* of joined rows is a consequence of the outcome. At the
moment of checkout the customer had items in the cart and those rows would have existed.

In the test window that is 98 orders, all positive: **25.9% of the secondary target's 379
positives, identified perfectly by a single artifact.** The secondary target's average
precision was 0.2938. Corrected — restricted to orders that join to items — it is **0.0190**,
against its own MDD of 0.027. It no longer clears its threshold. **The apparent result was the
artifact**, and it was the first number in this repo that looked good.

### Why neither existing guardrail could see it

**The column whitelist checks names**, and every column involved is admissible and genuinely
knowable at checkout. **The perturbation test inspects one column's values**, not a join's
cardinality. Both guardrails are real and both are structurally incapable of catching this.

Stated plainly in the report rather than glossed: the guardrails here did not catch it. A third
kind of check did.

### The fix

`scripts/06_join_audit.py` audits all eight joins against a 3× flag threshold, and
`eval/join_cardinality_audit.md` reports every one. It immediately found a **second,
independent leak**: `order_reviews` misses 768 matured orders whose `label_b` rate is 11.068%,
**9.31× panel — and unlike the items artifact, this one reaches the primary target.**

The primary target is unaffected by the items leak, because the risk set contains exactly one
order without item rows: an order that reached a carrier had items. **The gap between the
primary's AP of 0.0171 and the secondary's 0.2938 was the signal that something was wrong**,
rather than evidence that the broader label was easier — and reading it that way is what
started the audit.

---

## 10. Git Bash rewrites Unix-looking paths inside the `-v` flag

### What broke

```bash
docker run -p 8000:8000 -v "$HOME/.cache/kagglehub:/root/.cache/kagglehub" rto-detector
```

Git Bash auto-translates any argument that looks like a Unix path, and it does so **inside the
`-v` flag**. `/root/.cache/kagglehub` was silently rewritten to a literal Windows path before
Docker ever saw it. The mount never landed, the container fell back to downloading the dataset,
and there was **no error and no warning** — only a slower start that looks like a slow start.

### How it was caught

By following the repo's own instructions and then *verifying the result* rather than assuming
it. `docker inspect <container> --format '{{json .Mounts}}'` shows the destination actually
used. The download-fallback path is what ran the first time.

### The fix

Documented in `docs/API.md`, the README and the demo script, with the check and the workaround:
prefix with `MSYS_NO_PATHCONV=1`, or run from PowerShell or cmd, which do not rewrite the
argument.

Small, and included deliberately. It is the only incident here that a reader can reproduce in
thirty seconds, and it is the same failure mode as everything else on this page in miniature: a
step that reported success while not having done the thing.

---

## 11. A lock that stored a number the repo had already overturned

### What broke

Version 1 of `eval/TIER1_LOCK.json` stored `mdd_ap: 0.043` and nothing else about resolution.

The MDD is a *planning* figure, and `eval/significance.md` §5 explicitly overturns the
conclusion that had been drawn from it: the paired bootstrap CI on the AP difference excludes
zero, and the permutation null gives p = 0.00060. **A reader of the version-1 lock alone got a
conclusion this repo had already corrected** — and got it from the file whose entire purpose is
to be the authoritative frozen record.

That is the same defect class as every stale number on this page: a frozen figure that the
evidence moved past. It is worse than the others because the lock is the artifact people are
told to trust.

### The fix

Version 2 carries `mdd_superseded_by` naming the correction, plus the paired bootstrap delta and
CI, the permutation p-value and the null 99th percentile — all parsed from `eval/significance.md`
at capture time rather than transcribed.

A seventh cross-check was added, and it is a different *kind* of check from the other six. Those
six reconcile the parsed reports against a live pipeline run. The seventh ties two **reports** to
each other: `paired_bootstrap_delta` must equal `average_precision − test_prevalence`, so
`significance.md` and `model_report.md` cannot drift into describing different runs.

Its tolerance is **1.5e-4**, and the reason is worth recording because it is the opposite of
sloppiness. The reports render average precision at four decimals, which alone puts 3.6e-5
between the two sides; 1e-6 is unreachable and **would fail on correct data**. The drift the
check exists to catch — the hardcoded `+0.0125` from incident 5 — is 3.2e-3 out, more than
twenty times the tolerance. A check calibrated to its inputs' precision has teeth; one
calibrated to an aspiration gets disabled.

### What the lock still does not cover

It stores `secondary.average_precision: 0.2938` — incident 9's leaked figure — and does not
carry the corrected 0.0190. **The version-2 upgrade fixed this defect class for the MDD and not
for the secondary AP.** Recorded here rather than quietly fixed, because it is a live gap and a
reader of the lock should know about it.

---

## What this changed about how the repo is tested

The through-line, verified before asserting it: **the pytest suite was green while every one of
these defects was live.** Not one of them was surfaced by a failing test. What surfaced them
instead:

| Incident | What caught it |
|---|---|
| 1, 3, 4 | re-running a measurement and comparing it to the page |
| 5 | reading generators, and rendering the demo page |
| 6 | checking a margin against its own confidence interval |
| 7 | re-measuring an inherited claim while writing this document |
| 8 | a runtime self-check at service startup, which the suite did not replicate |
| 9 | an audit script written because two targets' results differed implausibly |
| 10 | verifying the result of an instruction instead of assuming it |
| 11 | upgrading the lock and noticing what version 1 had frozen |

Four practices came out of that.

**1. Assert structure, not output, wherever the structure is what the code claims.** The fold
tiling assertion caught the defect on all 14 parametrisations; the no-NaN assertion could not
have caught it at all. A behavioural assertion tests one run's luck when the failure mode
produces plausible values.

**2. Make coverage match the defect's shape, or state the coverage.** The 40-order stride was
0.04% coverage reporting PASS with total confidence against two defects it could never have
reached. The self-check now runs over every order in the historical population, and the service
refuses to start rather than serve a drifted store — 8.2 seconds of the 15.5-second cold start
is that check, and it is worth it.

**3. Generate every number that appears twice.** The headline block is produced from the lock by
`scripts/91_headline_block.py` and injected into README.md and ARCHITECTURE.md between markers;
`tests/test_headline_block.py` compares both copies **byte-for-byte** against what the generator
produces from the current lock. A tolerance there would defeat the purpose, because the failure
mode is a stale transcription and a stale transcription is usually still numerically plausible.
The same rule now applies inside generators: `08_policy.py::_cell()` reads the list its table is
built from rather than recomputing, so a sentence cannot drift from the table beside it.

**4. Re-run rather than re-read.** Incidents 1 and 3 were both invisible to diffing — one because
the harm was usually 2e-7, the other because regeneration reproduced the literals exactly. The
only thing that surfaced either was executing the pipeline and comparing the result to the page.
That is now what the Tier-1 lock is for, and its exclusions are in the assertion rather than only
in the docstring, because incident 4 showed that a documented exclusion is not an exclusion.

None of this makes the suite a proof. **585 tests pass and the fold defect would have been live
under every one of them.** What changed is the class of question the checks are pointed at: from
"does this produce a plausible answer?" to "does this satisfy the contract it claims?" — which is
the only one of the two that a plausible wrong answer cannot pass.
