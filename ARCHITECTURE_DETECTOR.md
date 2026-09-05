# Architecture: the detector

The engineering decisions behind the scoring path, each stated as the decision, the
alternative that was considered, and why the alternative was rejected. A reader should be
able to disagree with any of them and still see that it was made deliberately.

This document describes what shipped. The original design is
[ARCHITECTURE.md](ARCHITECTURE.md), written before any of this existed; where the two
disagree, this document says so and the code wins. The action path is
[ARCHITECTURE_RESPONDER.md](ARCHITECTURE_RESPONDER.md). The incidents referenced
throughout are in [BUILD_CHALLENGES.md](BUILD_CHALLENGES.md).

---

## 1. Data

### The panel

Olist's Brazilian e-commerce export: 99,441 raw orders, of which 99,433 survive a 30-day
maturation filter against a snapshot of `2018-10-17 17:30:18`. Every generated report
records the SHA-256 of each source CSV it read, so a report can be tied to the bytes it
was computed from.

**Decision: derive the snapshot from observed events only.** The snapshot deliberately
excludes `order_estimated_delivery_date`.

*Alternative considered:* use the maximum of every date column, including the estimate.
*Rejected because* the estimate is forward-looking. Including it pushes the snapshot past
the end of the observed data and silently weakens the maturation filter — orders would be
declared mature on the strength of a delivery date that had not happened.

Maturation removes **8 orders**, all 8 carrying a positive label — unresolved orders near the
boundary, not observed failures. The count is reported because dropping them silently would
move prevalence.

### The proxy label

`label_b` = entered fulfillment and never reached `delivered`.

**This is a proxy and is described as one throughout.** Olist does not record
return-to-origin as an outcome. The closest observable is an order a carrier accepted that the
export never marks delivered — which includes terminal carrier losses, thefts and
administrative write-offs, and excludes any RTO eventually recorded as a delivery. **No number
in this repo should be read as an RTO rate.**

**Decision: boleto is the COD analogue.** Deferred payment, no capture at checkout.

*Alternative considered:* treat every order as COD.
*Rejected because* the loss class being modelled depends on the payment not having been
taken. Boleto is the only Olist payment type with that property, and the boleto subset is
19,784 orders — enough to be a population rather than a slice.

### The risk set, and why conditioning

The primary population is `order_delivered_carrier_date IS NOT NULL` — orders a carrier
actually accepted. This removes **1,775 of 99,433 matured orders (1.785%)**, 225 of them in
the test window, broken down by status in
[`eval/label_targets.md`](eval/label_targets.md) §1 — 609 `unavailable`, 542 `canceled`, 314
`invoiced`, 301 `processing`, and 2 `delivered` rows carrying no carrier date, listed as a
data quirk rather than silently dropped.

**Decision: never-shipped orders are removed from the population, not scored as
negatives.**

*Alternative considered:* keep them and label them 0, giving a larger and simpler
population.
*Rejected because* return-to-origin is a post-shipment event by definition. An order
cancelled in the warehouse cannot return to origin — it never survived to the point where
the risk exists. Scoring it as a negative puts rows in the denominator that were never at
risk, which understates prevalence and rewards the model for assigning low risk to
outcomes it neither observes nor can act on.

---

## 2. The estimand, stated formally

> **P(fails | ships)** — the probability that an order fails to reach `delivered`, given
> that it was handed to a carrier.

Not P(fails). The distinction is load-bearing and it is priced, not waved at.

**Conditioning costs three things, and all three are on the record:**

1. **The deployed score is conditional.** Unconditional risk is `P(ships) × P(fails |
   ships)`. This repo does not model `P(ships)`, so a consumer needing the unconditional
   quantity must supply that factor themselves. `docs/API.md` states this on the `risk`
   field of the response contract, not only in an evaluation report.
2. **Selection into the risk set is not random.** Orders leave for reasons — seller-side
   cancellation, stock unavailability — that plausibly correlate with the covariates. This
   is conditioning on a post-treatment variable.
3. **A causal reading is barred.** The predictive claim on the subpopulation survives,
   because that is what the score is used for. Reading the model's structure as a
   statement about what causes RTO does not.

**The column that defines the risk set is a post-checkout field.** It is used to build the
risk set and the primary label, and for nothing else, and §5 below is how that exclusivity
is proved rather than asserted.

### The secondary target, and a warning about it

`label_a` — any non-delivered order, on all matured orders — is computed on every run as a
benchmark and never substituted for the primary.

| | Primary | Secondary |
|---|---|---|
| Label | `label_b` | `label_a` |
| Population | risk set (shipped) | all matured |
| Estimand | **P(fails \| ships)** | P(fails) |
| Test rows | 19,662 | 19,887 |
| Test positives | **154** | 379 |
| Test prevalence | **0.783%** | 1.906% |

**The secondary target's headline AP of 0.2938 is an artifact and must not be quoted as a
result.** It is inflated by the `order_items` join leak analysed in §5; restricted to orders
that join to items it is **0.0190** against a 1.420% prevalence — a gap of +0.0048 against
its own MDD of 0.027, so it no longer clears its threshold. The primary target is unaffected:
the risk set contains exactly one order without item rows, because an order that reached a
carrier had items. **The lock stores only the leaked 0.2938 figure**, which is a real defect
in the lock's coverage and is flagged in §13.

---

## 3. Splits

### Temporal, three ways, with a pinned boundary

Chronological by `order_purchase_timestamp`. Test is the most recent 20%; validation is
the 90 days immediately preceding it. The boundary is `2018-05-24 16:58:49`, the purchase
timestamp at row position 79,546 of 99,433 matured rows sorted ascending.

**Decision: an observed timestamp, not a round date.**

*Alternative considered:* cut at a calendar boundary such as `2018-06-01`.
*Rejected because* a round date is a number that has to be maintained. Deriving the
boundary from a row position makes it a function of the data, and
`tests/test_split.py::test_split_boundary_is_an_observed_timestamp` asserts it is a
timestamp that actually appears in the panel.

Eight further tests in `tests/test_three_way_split.py` pin coverage, disjointness,
chronological ordering, validation/test separation, the declared validation length, the
committed boundary, positives in every split, and stability across calls.

### Group leakage: measured and pinned, not asserted away

ARCHITECTURE.md §3.7 asks for an assertion that no `customer_unique_id` straddles train
and test. **That assertion is false on this panel**, and the honest response was to measure
the quantity and pin it rather than delete the test.

| Population | Straddling customers |
|---|---:|
| All matured | **472** |
| Risk set | **461** |

`tests/test_split.py:110` and `:124` assert those exact counts, so the number cannot drift
unnoticed, and `data/COLUMN_WHITELIST.md` records the discrepancy against the design.

**Why this is not feature leakage here.** Under point-in-time construction (§4) a repeat
customer's test-period order sees only that customer's own strictly-prior history. Nothing
from the test window flows into a training feature. **It would be leakage under k-fold
out-of-fold target encoding** — which is the reason §4 rejects OOF encoding, and the two
decisions are the same decision seen twice.

---

## 4. Point-in-time features

35 columns over 97,658 risk-set rows.

**The rule:** for the row at time *t*, every history-derived value is computed from rows
with `order_purchase_timestamp < t`. Strictly less than — two orders sharing a timestamp
must not see each other.

### Expanding windows, not out-of-fold encoding

**Decision: point-in-time expanding windows for every history feature.**

*Alternative considered:* k-fold out-of-fold target encoding, the standard treatment for
high-cardinality categoricals.
*Rejected because* OOF encoding is designed for random splits. Under a temporal split it
lets December inform June's encoding, which is exactly the leakage the chronological split
exists to prevent — and, as §3 notes, it is the mechanism that would make those 472
straddling customers harmful.

Smoothing is explicit: pincode, seller and route encodings pull toward the global prior with
weight 50, customer encodings with weight 20. That pull has a consequence §11 reports rather
than presenting as a pure win.

### History encodings target `label_a`, not the primary target

**Decision: aggregate the broader label in history features.**

*Alternative considered:* encode `label_b`, the target actually being predicted.
*Rejected because* `label_b` is defined by `order_delivered_carrier_date`. Encoding it
would make a feature depend on that column transitively, and the exclusivity invariant —
no feature reads the carrier date, directly or indirectly — would become a matter of
tracing rather than a fact. Using `label_a` keeps the invariant absolute. The cost is that
the encoding targets a broader outcome than the model predicts, which is stated in
`eval/feature_report.md` §1.

### Verified by truncation invariance, not by inspection

The matrix is built on the full panel, built again truncated to `ts < cut`, and the surviving
rows required to be identical — a feature reading data at or after its own timestamp changes
when that data is removed. Cut: `2018-03-02 23:53:22.800000`. PASS on the full matrix and on
the `order`, `customer`, `pincode` and `availability` groups independently.

### Ties, and the primitive that got them wrong

A naive `cumsum().shift(1)` lets the second order in a tie block read the first, which is a
same-instant read rather than a prior one. Every primitive in `features/pit.py` resolves
the value as of the *start* of the tie block instead.

The first implementation used `transform("first")`, and **`GroupBy.first` skips nulls**, so
**534 of 97,658 orders** read a same-instant order as "the previous order". Fixed by gathering
the value at each block's first *position* rather than its first *non-null* value
(`features/pit.py::_block_first`). That was not a rounding change — NaN and 0.0 are not close
substitutes to a tree booster — and it moved test AP from 0.0203 to 0.0171, ROC-AUC from
0.6384 to 0.6450, boosting rounds from 33 to 50, and the probability ceiling to 6.856%. Every
downstream artifact was regenerated.

### One definition, two evaluation strategies

Serving cannot re-derive expanding windows from 97k rows per request. `features/store.py`
builds a `HistoryStore` at startup that indexes the historical population by group and
timestamp; a query binary-searches it. **The feature formulas are untouched** — this is one
definition with two evaluation strategies, not a second implementation, and the difference
between those two framings is the whole safety argument.

The guarantee is enforced twice:

- **At startup**, `ScoringService._assert_store_matches_rowscan` scores **every** order in
  the historical population through the store and requires all 35 features to be
  bit-identical to the row-scan. Not a sample. The service refuses to start rather than
  serve a drifted store, because a drift would raise nowhere else — the model would
  happily score the wrong number.
- **In the suite**, the API-vs-batch parity assertion requires 1e-12 agreement with the
  batch predictions.

That startup check used to walk a **fixed 40-order stride**, and it passed for months
against two genuinely defective implementations, because neither defect's rows intersected
the stride. See [BUILD_CHALLENGES.md](BUILD_CHALLENGES.md) incident 2.

---

## 5. Leakage defence

### The whitelist

`data/COLUMN_WHITELIST.md` splits every source column into checkout-safe, post-checkout and
label-only. `tests/test_leakage.py:40` asserts the label-only and post-checkout sets are
disjoint and `:44` asserts no checkout-safe column appears in a forbidden list, so the two
halves cannot contradict each other.

### The perturbation test, and why the name check is not the guarantee

**Decision: prove exclusivity behaviourally.**

*Alternative considered:* assert `order_delivered_carrier_date` is absent from the feature
matrix by name.
*Rejected because* **a name check is provably insufficient.** It proves the column is not
carried through. It cannot prove no feature was silently *derived* from it — a perfectly
legal column, absent from every forbidden list, can still encode post-outcome information.

So the column is perturbed instead. `tests/test_leakage.py` blanks
`order_delivered_carrier_date` to `NaT`, rebuilds everything, and asserts four things:

| Test | Assertion |
|---|---|
| `test_carrier_date_is_used_for_label_b` (:109) | `label_b` positives collapse to **0** — if the label did not consume the column, blanking it would change nothing |
| `test_carrier_date_defines_the_risk_set` (:119) | the risk set becomes **empty** |
| `test_carrier_date_does_not_reach_the_feature_matrix` (:125) | `pd.testing.assert_frame_equal` on the checkout frame — any feature derived from the column, however indirectly, shows up here as a difference |
| `test_label_a_does_not_depend_on_carrier_date` (:140) | `label_a` positive count is **unchanged** — the perturbation is specific, not a blanket corruption |

The column demonstrably moves exactly what it should and nothing else.

### The request blocklist is a courtesy, and a test proves it

`api/contract.py` enumerates twelve post-outcome fields and `/score` returns `422` on any
payload carrying one. That is a good client-facing error. **It is not the leakage
guarantee**, and the repo proves it rather than claiming it:

`tests/test_whitelist_is_load_bearing.py:82` monkeypatches `FORBIDDEN_PAYLOAD_FIELDS` to
the empty set, posts a payload carrying every one of those twelve fields with
outcome-revealing values, and asserts the returned risk equals the risk for the same
payload with those fields stripped, to an absolute tolerance of **1e-12**. The typed
request model and the checkout whitelist are what keep post-outcome data out of the model;
the enumerable blocklist is a convenience at the edge. `:97` separately asserts the 422
still fires when the courtesy check is enabled, so demoting it did not delete it.

### The class of defect neither check can see

**Join cardinality.** 767 matured orders have no `order_items` rows and 100% of them are
`label_a` positives, at 33.65× the panel rate. Every column involved is on the whitelist
and genuinely knowable at checkout; what leaks is the *absence* of joined rows, which is a
consequence of the outcome. A name check cannot see it and the perturbation check inspects
one column's values rather than a join's cardinality.

That is why `scripts/06_join_audit.py` exists and why
[`eval/join_cardinality_audit.md`](eval/join_cardinality_audit.md) audits all eight joins
against a 3× flag threshold. It found a **second, independent** leak the same way:
`order_reviews` misses 768 matured orders whose `label_b` rate is 11.068%, **9.31× the
panel rate — and this one reaches the primary target.** Stated plainly in the report: the
guardrails in this repo are real, and they did not catch this. A third kind of check did.

---

## 6. Model

LightGBM, binary objective, **natural class distribution and no resampling**, early-stopped
on average precision at round 50 of 2000.

**Decision: no `scale_pos_weight`, no resampling.**

*Alternative considered:* reweight or oversample at 0.78% prevalence.
*Rejected because* the policy layer consumes **probabilities**, not ranks. Reweighting buys
recall and wrecks calibration, and a miscalibrated probability entering an Elkan threshold is
wrong in a way no ranking metric would reveal. Retained as an ablation, not a default.

**Decision: early-stop on average precision**, not AUC or logloss — stopping on a different
metric than the one reported optimises for something nobody reads.

Parameters sit inside pre-stated bands rather than being tuned: `num_leaves` 31 (band 15–31),
`max_depth` 5 (band 4–6), `min_child_samples` 75 (band 50–100), `lambda_l2` 1.0,
`feature_fraction` 0.7, `learning_rate` 0.05. Determinism is pinned with
`deterministic=True`, `force_row_wise=True` and `num_threads=4` — the thread count too,
because `deterministic` is only honoured alongside a forced tree-building direction.

### Monotone constraints, and the four jobs they do

Three constraints ship:

| Feature | Constraint | Spec statement |
|---|---:|---|
| `pincode_failure_rate_smoothed` | +1 | pincode failure rate ↑ → risk ↑ |
| `cust_prior_failures` | +1 | prior failures ↑ → risk ↑ |
| `cust_prior_boleto_ratio` | **+1** | prepaid ratio ↑ → risk ↓ — **inverted** |

The four jobs: they encode domain knowledge the sample is too small to learn reliably; they
regularise at ~1% prevalence; they make the SHAP output non-embarrassing, so no
merchant-facing string can say "this pincode's higher failure rate reduced risk"; and they
are what makes the counterfactual endpoint safe to offer at all, since
`models/counterfactual.py` refuses to counterfactual any feature outside
`MONOTONE_CONSTRAINTS`.

**The sign inversion is the interesting part.** The spec constrains `prepaid_ratio`; this
matrix carries `cust_prior_boleto_ratio`, its complement, so the same statement takes **+1**,
not −1. A sign error here is worse than no constraint: it would force a backwards relationship
and the SHAP strings would confidently tell a merchant the opposite of the truth — the exact
failure mode the constraints exist to prevent. Recorded in `INVERTED_FROM_SPEC` and asserted
in `tests/test_model.py`.

`component_size` is omitted because the network/ring group is not in this matrix. Constraints
on features that exist but are not shipped live in a separate `UNSHIPPED_CONSTRAINTS` map, so
the required map never claims anything about a feature the model does not carry.

**Decision: monotone constraints rather than an EBM.**

*Alternative considered:* an Explainable Boosting Machine, which is glass-box by
construction.
*Rejected because* the ablation cannot resolve at this test size — 154 positives will not
separate two models whose AP differs by less than the MDD — so the comparison would produce
a preference dressed as a finding. The constraints capture most of the interpretability
guarantee in five lines. It is recorded in the deferred register as the **first item to
re-add if a larger test set becomes available**, which is the difference between a decision
and an omission.

### The column-order contract

LightGBM addresses features positionally, and **so do monotone constraints**. A frame with
the right columns in the wrong order produces silently wrong predictions — every value a real
number in a plausible range, nothing raised — and applies the pincode constraint to whichever
feature now sits at that index. The trained order is recorded in the bundle and `predict`
requires an exact match. **It does not reorder to fit**: reordering papers over a caller that
has lost track of its own schema, and the next such bug would be silent again.

---

## 7. Calibration

Platt, two parameters, `A = -0.703441`, `B = 1.477981`, fit on the last 30 days of the
validation window — 7,380 orders, 71 positives.

**Decision: Platt, not isotonic, not beta.**

*Alternative considered:* isotonic regression, the usual first choice for calibration.
*Rejected because* isotonic fits a free monotone step function and needs far more than 154
positives before it stops memorising the calibration set. Beta calibration adds a third
parameter to the same problem. Two parameters are what this sample supports, and Peduzzi's
10-events-per-parameter floor puts the admissibility bar at 20 positives against the 71
available.

Platt's target smoothing is used rather than hard 0/1 labels — at 71 positives a
well-separated score can drive the fitted slope to infinity. Uniform-mass binning is the
**evaluation scheme**, not a second remap: at 0.78% prevalence equal-width bins put nearly
every row in the first bin, and a bin-average remap stacked on Platt could break monotonicity
in the raw score, which this pipeline asserts.

### The probability ceiling, stated first

| | Calibrated P(fails \| ships) |
|---|---:|
| Minimum | 0.36150% |
| Median | 1.00149% |
| 99th percentile | 2.3795% |
| **Maximum** | **6.8559%** |
| Test prevalence | 0.7832% |

This is reported before anything else in `eval/calibration.md` because **a policy layer
built against a threshold this model cannot reach is inert.** Any Elkan `p*` above 6.856%
selects nothing. Since `p* = c_FP / (c_FP + c_FN)`, a threshold inside the reachable range
requires `c_FN / c_FP ≥ 13.59`. That is a hard constraint on the policy layer, not a
presentational note, and §9 measures the policy against it.

### The window: a comparison, not a selection, and the evidence that it is

**Decision: the 30-day window, chosen on prevalence-drift grounds and pre-registered.**

`eval/calibration_window.md` was written *before* any Brier number existed, and says so.
The argument: prevalence rises monotonically as the window widens backwards — 0.962% →
1.223% → 1.435% against a test-window base rate of 0.783% — and a Platt map's intercept
absorbs the base rate of the data it was fit on. Fit at 1.435% and apply at 0.783% and every
probability the map emits is inflated by that 1.83× prior, in a way no amount of ranking
quality corrects.

The design says to select the window by Brier score on the validation window. **Taken
literally that instruction is structurally biased**, and the bias is stated before the
numbers: validation *is* 90 days, so the 90-day candidate is the whole evaluation set and
is evaluated at home. Cross-fitting removes the in-sample advantage but not the base-rate
alignment, which is the part that matters for a map whose intercept absorbs the base rate.
So the comparison is run on a fixed recent slice — the last 30 days of validation, identical
for every candidate — with out-of-fold predictions.

| Window | Brier on slice | Brier on whole validation |
|---:|---:|---:|
| **30d** | **0.0095359** | 0.0141342 |
| 60d | 0.0095436 | 0.0141118 |
| 90d | 0.0095617 | **0.0141051** |

**The two columns disagree**, and that disagreement is the bias made visible: the
whole-window column ranks the candidates in order of length.

**And then the margin is checked, which is the part that matters:**

| Comparison | ΔBrier vs 30d | 95% CI | Resolves |
|---|---:|---|---|
| 60d − 30d | +0.0000077 | [−0.0000206, +0.0000331] | **no** |
| 90d − 30d | +0.0000259 | [−0.0000161, +0.0000618] | **no** |

**No gap excludes zero.** The 30d/60d margin is 7.7e-6 against a confidence interval
roughly seven times wider that straddles zero. A proper scoring rule that cannot separate
the options has not made a selection, so **this is a comparison, not a selection**, and
calling it a selection would overstate what was measured.

`CALIBRATION_WINDOW_DAYS = 30` is a fixed choice hardcoded in five places, and nothing
downstream reads that table. **The comparison exists to show the choice is not
load-bearing** — that the system would report materially the same thing at 60 or 90 days —
not to make it. That is the evidence that it is a comparison: no code path consumes its
output.

Checked after the fact against test, which is not an input to the selection: the 30-day
window is best on test Brier (0.0077613) and least over-predicted in the top decile (1.31×
against 1.93× and 2.13×). Had the literal reading of the design been followed, 90d would
have shipped — worse on both. **That is a finding about the design's rule, not a result to
celebrate.**

### Calibration quality

| | Value |
|---|---:|
| Brier, Platt-calibrated | 0.0077613 |
| Brier, constant at test prevalence (oracle) | 0.0077710 |
| smECE (σ = 0.005) | 0.002510 |
| Top-decile calibration gap | 0.004197 |

Bandwidth is stated as a parameter because a binned ECE can be driven toward zero by
choosing enough bins; the smooth version replaces that choice with one explicit number. The
constant-at-prevalence row is the reference that matters: at 0.78% prevalence a model
predicting the base rate for everyone already achieves a very low Brier, so **Brier
improvements here are small in absolute terms by construction** and the reliability curve
carries more information than the scalar.

Top-decile calibration is the headline rather than global ECE, because global ECE is
dominated by the mass near zero, which is not where the system acts.

**One caveat carried forward:** validation carries both the early-stopping signal and the
Platt fit, so the calibration map is fit on data the model early-stopped against. That is a
mild optimism in the calibration numbers, not in the ranking numbers. Stated rather than
hidden.

---

## 8. Explainability

TreeSHAP, `tree_path_dependent`, explainer built once at construction.

**TreeSHAP is exact for a tree ensemble.** It is not a sampling estimator and not a local
surrogate, so there is no approximation error to report and no convergence to check.
Coverage is 100% because there is one tree model.

### Additivity, verified

```
sum_j φ_j(x) + base_value == model margin(x)
```

**Max reconstruction error across 19,662 orders: 7.105e-15** — floating point, not method
error. Independently cross-checked against LightGBM's own `pred_contrib=True`, a separate
implementation of the same algorithm in a separate codebase: the two agree to **exactly
0.0**.

This is a **wiring check, not an accuracy check**. A failure would mean the explainer is
pointed at the wrong output space, the wrong feature order or the wrong iteration count —
the same class of bug as fitting Platt on probabilities instead of log-odds — and it would
silently corrupt every reason string rather than raising.

### A deviation from the design, stated

The design specifies **interventional** perturbation. It is **not available for this
model**: `shap` refuses interventional TreeSHAP on any model containing native categorical
splits, and `product_category` is one — the highest-gain feature in the matrix.

`tree_path_dependent` is also exact for the tree; the two differ in the reference
distribution. The practical consequence is that attribution is spread across correlated
features somewhat differently than an interventional reference would spread it. **The route
to interventional is open and costed**: applying the smoothed point-in-time encoding to
`product_category` makes the matrix fully numeric and unblocks it, at the cost of a retrain
that moves every number downstream. Recorded as a live decision rather than resolved
silently.

### SHAP is computed in margin space; the policy decides in probability space

Attributions are in the raw log-odds margin — the space the trees are additive in, and the
only space where the additivity identity holds. Platt is strictly monotone, so the
*ordering* is identical in both spaces. But **the reason ranking is an attribution in margin
space while the tier is a decision in probability space, and they can disagree at the
margin**: an order can sit just under a tier threshold while showing a large positive
attribution. Neither is wrong; they answer different questions. Stated because a reviewer
will ask.

### The reason mapping is a designed component

The strings do not fall out of the model. There is an attribution space, a grouping rule
and a template set, each of which could have been chosen differently, and the pipeline
diagram draws the mapping as its own stage to make that visible.

**Decision: bucket orders by summed positive SHAP attribution.**

*Alternative considered:* the pre-SHAP stand-in — dominant feature group by standardised
deviation from the training median — which was labelled provisional when it shipped.
*Rejected because* it buckets by how *unusual* an order looks rather than by what moved its
score. The two disagree on **47.7% of orders**, and the disagreement is not cosmetic: it
changes the treated set from 26 orders to 34 (10 added, 2 dropped, net +8), because the
effectiveness matrix is indexed by reason and a changed bucket changes an order's Elkan
threshold. The stand-in is retained in `policy/effectiveness.py::dominant_reason` so the
comparison stays reproducible, and is no longer the production path.

**That is the mechanism by which explanation feeds the decision, rather than decorating
it.**

---

## 9. Cost policy

### Per-order Elkan thresholds

```
p*(o, t) = c_FP(o, t) / (c_FP(o, t) + c_FN(o, t))
```

computed **per order and per tier**, not once globally.

**No labels enter `policy/elkan.py`.** That is not a convention — `apply_policy` and every
threshold function take costs and a calibrated probability, and there is no argument to
pass a label to. `expected_cost_at_threshold` does take `y`, and is an evaluation function
never called while choosing a threshold. The assertion "the policy never sees test labels
when selecting thresholds" is therefore structural rather than procedural.

### The impression / triggered decomposition

`c_FP` is split into a cost paid on every treated order regardless of response
(**impression**) and a cost paid only when the customer reacts (**triggered**).

That decomposition is what makes the tiers behave differently rather than being three names
for one lever. `prepaid_only` has an impression cost of roughly zero and a large triggered
cost — the 15% abandonment probability when COD is removed. `defer` has a real BRL 3
impression cost in analyst time, paid whether or not the order is released. One scalar `c_FP`
would make the two indistinguishable.

**Sixteen constants feed the cost model, every one tagged**: 2 `OBSERVED`, 13 `ASSUMPTION`,
1 `GUESS`. The guess is `fatigue_allowance` at BRL 0.50 per message — brand annoyance and
the unsubscribe analogue — and it is labelled a guess because there is no measurement behind
it and none is available offline. It is included rather than set to zero because setting it
to zero asserts that messaging customers is free, which is false, and the sensitivity sweep
is the honest treatment.

### The measured payoff of per-order costs

The claim that scalar costs are wrong is testable, and the test is decisive:

| Cost model | Threshold | Orders treated |
|---|---|---:|
| Flat — `confirm` at median costs | 6.944% | 0 |
| Flat — `prepaid_only` at median costs | 18.400% | 0 |
| Flat — `defer` at median costs | 32.161% | 0 |
| **Flat — total** | | **0** |
| **Nested — per-order p\*** | varies per order | **34** |

**A flat cost model treats nothing at all**, because its single threshold sits above the
6.856% ceiling for every tier. On this panel that is the difference between a policy that
exists and one that is completely inert.

### The feasibility statement, made before the result

| Tier | Median p\* | Median c_FN/c_FP | Orders assigned |
|---|---:|---:|---:|
| `confirm` | 7.404% | 12.51 | 12 |
| `prepaid_only` | 16.030% | 5.24 | 22 |
| `defer` | 31.659% | 2.16 | **0** |
| `allow` | — | — | 19,628 |

**Every tier's median threshold sits above the ceiling.** For the median order no tier is
worth firing at any risk this model can express. The 34 that do fire are those whose
individual cost ratio is unusually favourable — high freight relative to margin.

`defer` never fires, and that is reported rather than tuned away. The gap is stated so a
reader can judge it against their own numbers: `confirm` fires for the median order if
`c_FN` rises or `c_FP` falls by 1.09×, `prepaid_only` by 2.59×, `defer` by 6.29×. **The
cost constants were set before this table was computed.** An inert policy correctly
reported is a result; a policy tuned until it fires is not.

### The action ladder

`allow → confirm → prepaid_only → defer`. **Nobody is blocked.** The worst outcome for a
false positive is being asked to pay upfront — a materially different fairness profile from
a system that refuses service, and a design choice claimed explicitly rather than emerging.

`tests/test_policy.py:196` asserts the tier set is exactly those four and that neither
`block` nor `reject` is in it. `tests/test_policy.py` also asserts that higher risk never
*reduces* the chosen tier's severity, which is the monotonicity property a ladder has to
have to be a ladder.

### Treatability

**The central correction: the highest-risk orders are not the orders where intervention
helps most.** A fake or competitor-harassment order is high risk and has near-zero uplift
from a confirmation SMS — you pay and get nothing. An address missing a landmark is moderate
risk and high uplift from an address-repair prompt. So effectiveness is a function of *why*
the order is risky.

| Reason | `confirm` | `prepaid_only` | `defer` |
|---|---:|---:|---:|
| `order_composition` | 0.35 | 0.25 | 0.40 |
| `pincode` | 0.15 | 0.45 | 0.50 |
| `customer_history` | 0.10 | 0.60 | 0.65 |
| `availability` | 0.20 | 0.30 | 0.35 |

**The structure is the claim. Every value is an assumption**, and they cannot be measured
offline because the counterfactual does not exist in observational data. They are swept
rather than presented as known.

### The sensitivity sweep, and what it reveals about identifiability

Sweeping `c_rto` ×{0.5, 1.0, 1.5} against effectiveness ×{0.5, 1.0, 1.5} gives treated counts
**symmetric across the diagonal**: ×0.5/×1.0 and ×1.0/×0.5 both treat 3; ×1.0/×1.5 and
×1.5/×1.0 both treat 144. Since `c_FN = e × c_rto − impression`, the policy responds to the
*product* and barely to either alone. **The two axes are not separately identifiable** — nine
cells give five distinct behaviours, and telling them apart requires a randomised rollout.

**More treatment is also not better.** At ×1.5/×1.5 the policy treats 629 orders and the
saving turns negative. The optimum is interior, and a policy tuned to spend its whole
intervention budget would walk straight past it.

---

## 10. Evaluation

### MDD, computed first and used as a reporting rule

Minimum detectable AP difference at 154 test positives: **0.043** on the primary target,
0.027 on the secondary. Printed ahead of any metric, with the rule stated: any AP difference
below 0.043 is reported as "not resolvable at this sample size" — not as a win, and not
rerun on the secondary target until it resolves.

### And then overturned, on the record

The MDD was then used as a *test*, and that was wrong. `eval/model_report.md` §1 concluded
the model could not be shown to beat the prevalence baseline, by comparing a point estimate
(+0.0093) against a planning heuristic (0.043).

**That comparison is strictly weaker than testing the difference**, for three reasons:

1. **It discards the pairing.** Both models are scored on the same test rows, so they share
   every source of variance arising from which orders landed in the test window. Resampling
   the difference cancels that shared variance.
2. **It assumes ρ rather than measuring it.** The MDD was quoted at ρ = 0.8. The bootstrap
   needs no such assumption.
3. **The prevalence baseline is not an independent model.** Its AP *is* the prevalence of
   the sample, so it moves with the resample's positive count. Pairing captures that; an MDD
   comparison treats it as a fixed constant.

The correction was made **at the generator**, so the report now states the correction rather
than the error, and the reasoning that produced the over-cautious call stays visible. The
error was in the conservative direction — it under-claimed — which is still an error.

### What resolves

| Test | Result |
|---|---|
| Permutation null vs random ranking | **p = 0.00060**; observed AP 0.0171 above the null 99th percentile of 0.0126 |
| Paired bootstrap, AP vs prevalence baseline | **+0.0093, 95% CI [+0.0034, +0.0279]**, excludes zero; Δ ≤ 0 in 0.0000 of 10,000 resamples |
| DeLong on ROC-AUC | **0.6450, 95% CI [0.6050, 0.6850]**, p = 1.18e-12 |

**The permutation null is reported first, and it is the stronger claim.** A randomly ranked
model does not score the prevalence — it scores slightly *above* it, because average precision
averages precision-at-rank over the positives and precision-at-rank is upward-biased at the
small ranks where a positive can land early by luck. Mean AP of a random ranker on these
labels is 0.008314 against a prevalence of 0.007832: an upward bias of **+0.000482** the
textbook baseline misses, measured rather than assumed away.

How much the baseline choice matters, over 200 independent noise draws at n = 4,000:

| Procedure | Rejection rate under the null | Nominal |
|---|---:|---:|
| Paired bootstrap vs **prevalence baseline** | **0.145** | 0.05 |
| **Permutation null** (random ranking) | **0.055** | 0.05 |

The permutation null is calibrated. **The prevalence-baseline comparison rejects 2.9× as
often as it should**, and the null mean sat above the prevalence in 100% of draws — the bias
is structural, not sampling noise. So the bootstrap result is the weaker of the two, and the
permutation p-value carries the claim. Here they agree and the margin is not close, so the
conclusion does not depend on which is used.

Those four figures were literals under a heading reading "Measured, not asserted." until
they were computed; see [BUILD_CHALLENGES.md](BUILD_CHALLENGES.md) incident 3.

DeLong's midrank estimator is used because it is exact under ties, which matters here: a
gradient-boosted model assigns identical scores to every row landing in the same combination
of leaves.

### What does not resolve, stated as plainly

- **The encodings.** Full matrix leads the order-fields-only baseline by +0.0066 AP, 95% CI
  [−0.0010, +0.0249], which includes zero. This test size cannot separate a small
  contribution from none.
- **The policy's saving.** −145.52 BRL against doing nothing, paired 95% CI [−595.77,
  +34.83], worse than doing nothing in 26.05% of resamples.
- **Drift.** `eval/drift_slices.md` splits the test window into three and finds apparent
  differences, and states that at ~126 positives per slice they are within noise. **It is
  also a secondary-target result and says so three times**, because a primary-target drift
  claim cannot be built from it.

**Separation from a prevalence baseline is a low bar and is not a claim that the model is
useful.** At the top-5% budget precision is 1.42% and recall 9.09%. The policy layer decides
whether that clears the cost of intervening, and §9 is that layer.

---

## 11. Fairness

The design asks for metro / tier-2 / tier-3. **Olist is Brazilian and carries no Indian
region tier**, so two substitutes are reported with their derivations stated: pincode
observation density measured **on the training split only** (a bucket boundary drawn with
knowledge of the test window would not be a proxy for anything), and customer state, pooled
below 300 test orders. Neither is a region tier and both are labelled substitutes.

Intervention rate is treated as a separate question from precision and recall: those are
model questions, this is a policy question.

| Bucket | n | Positives | Prevalence | **Intervention rate** | 95% CI |
|---|---:|---:|---:|---:|---|
| sparse (<10) | 13,437 | 107 | 0.796% | **0.127%** | [0.067, 0.193] |
| medium (10–49) | 5,996 | 47 | 0.784% | **0.267%** | [0.150, 0.400] |
| dense (≥50) | 229 | 0 | 0.000% | **0.437%** | [0.000, 1.310] |
| panel | 19,662 | 154 | 0.783% | 0.173% | |

**The finding runs the opposite way to the concern.** The worry the design is written
against is that sparse or rural pincodes get downgraded disproportionately. On this panel
sparse pincodes carry **69.5% of the test window's failures but receive only 50.0% of the
interventions**, and are pushed less far up the ladder: 41% of sparse treatments reach
prepaid-only or above, against 88% for medium. Every bucket's CI spans the panel rate, so
none of it resolves, and that is stated in the report before the interpretation.

**The mechanism is the smoothing**, and this is the pincode ablation's real result: pulling
a thin pincode's encoded failure rate toward the global prior is exactly what stops the
encoding becoming a blacklist — and the same pull keeps sparse orders below the threshold at
which intervening pays. **The safeguard against over-treating under-observed regions is
mechanically what under-protects them.**

That is a genuine trade rather than a bug, and not obviously the wrong one: the failure mode
the smoothing prevents is a systematic penalty on customers in rarely-served areas, which is
worse than declining to intervene on their behalf. But the flattering reading — "no evidence
of over-treatment in sparse regions" — would be hiding it.

The ladder is broken out per bucket, because an aggregate intervention rate cannot test the
"nobody is blocked" claim: a bucket treated at the same rate but pushed further up the
ladder is a different outcome. Nobody in any bucket is blocked, because there is no block
tier.

---

## 12. Serving

### The contract

`POST /score` takes a **Razorpay webhook event envelope** — entities nested under
`payload.<name>.entity`, amounts as integers in centavos, `created_at` in Unix seconds.
Integration is a webhook subscription rather than a bespoke mapping.

`currency` must be `BRL`. The endpoint rejects anything else rather than converting behind
the caller's back, because a payload in another currency would be scored against magnitudes
that mean something different and the cost model would silently price the wrong one. A float
amount is rejected as a rounding bug waiting to happen.

The response carries `risk`, `tier`, three templated `reasons`, `model_version`,
`threshold_used`, `features_missing`, a `cost_breakdown` tagged
`basis: assumed_cost_constants`, and a `decision_id`. `model_version` is a hash of the
booster, the calibrator parameters and the feature order, so any retrain changes it
automatically and every response can be traced to the model that produced it.

`threshold_used` on an `allow` decision reports the cheapest action tier's threshold — the
bar the order failed to clear — rather than being null, so the field always answers the same
question.

**The raw body is read before validation**, because the leakage gate needs to see fields a
typed model would drop. A schema that ignores unknown keys cannot reject a post-outcome
field, since rejecting it requires noticing it.

### Idempotency

`responder/ingest.py` derives an event key from the `x-razorpay-event-id` header when present
and from a canonical SHA-256 of the sorted, separator-normalised body otherwise, recording
which derivation was used in the decision's `state_reason`. The key is the primary key of an
append-only `inbox` table protected by `BEFORE UPDATE`/`BEFORE DELETE` triggers, and a
repeated event returns the stored response verbatim rather than re-scoring.

**Decision: fall back to a body hash rather than requiring the header.** *Rejected
alternative:* refuse events without one — a sender that omits the header is a real
integration, a canonical body hash is a correct key for it, and recording the derivation lets
an operator tell the two apart in the log.

### Latency, and the honest range

**The 200ms budget is not established by this repo, and no verdict should be quoted.**

Four measurements of the same code path on the same machine:

| Source | p50 | Note |
|---|---:|---|
| `eval/TIER1_LOCK.json` | 322.1ms | earlier capture, still in the lock |
| `eval/latency.md` | 125.3ms | current committed artifact, verdict PASS |
| `eval/responder/write_gate.md`, baseline | 64.862ms | different harness, same endpoint |
| `eval/responder/write_gate.md`, post-wiring | 105.3ms | worst-pass p99 143.152ms |

`scripts/90_capture_tier1_lock.py` records the diagnosis in its own docstring: five captures
produced p50 between roughly 90ms and 325ms on unchanged code at the same SHA, and **the
instability lives between process invocations rather than within them** — one run's three
passes agreed to within 1.3% of each other at ~315ms while another sat at ~121ms. A
worst-of-three protocol averages over the wrong axis, and a single run cannot establish which
regime the host is in.

The write gate is the sharpest evidence, because it checked: its pre-committed rule required
the measured baseline p99 to sit within ±15% of the locked 374.5ms, the measurement came back
at 72.9ms, and it recorded **"Comparability: STOP — host is not comparable to the locked
baseline"** rather than proceeding. That is the protocol working.

**What is stable and worth claiming** is the shape rather than the magnitude. Feature
construction is 92.6% of the request. **SHAP is not the long pole** — the design expected it
to be, and it measures 4.00ms at p50, 3.2% of the request, because TreeSHAP on a single row
of a 50-tree ensemble is a handful of tree walks and the explainer is built once at startup.
The expectation was reasonable and the measurement contradicts it.

Cold start is reported on its own line rather than folded into p50, because a judge running
the container experiences it once: 15.5s, of which the full-population store self-check is
8.2s. **The model is trained at startup rather than loaded** — `artifacts/models/` holds a
saved booster and the service does not read it — which is the honest description of this
build and the single largest thing a production image would change.

---

## 13. The Tier-1 lock

`eval/TIER1_LOCK.json` freezes what the evaluation pipeline produces at a given commit, so a
later change that moves a headline number fails a test instead of shipping quietly.

**The capture script is allowed to do nothing except read and serialise.** It modifies no
training script, feature definition, calibrator, cost constant or generator. If capturing the
lock could change the thing being locked, the lock would be worthless.

**Two sources, and a fixed rule for choosing.** Numbers come from the committed reports in
`eval/`, parsed at run time — and **only from generated tables and generated f-string lines,
never from hand-written prose**, because the prose of at least one generated report has been
confirmed to carry figures its own table contradicted. The two things `eval/` does not
contain — the per-order Elkan threshold state and the model-version fingerprint — come from a
live pipeline run. Where a number exists in both places it is parsed *and* cross-checked, and
a disagreement raises rather than locking a fiction. Row lookups require exactly one matching
row, so a report reformatted such that a label matches twice fails loudly.

### The three integrity digests

Compared by exact string equality, never float tolerance.

- `feature_order_sha256` — the trained column order, kept separate from `model_version` so
  that when the fingerprint moves you can tell whether the contract moved with it.
- `cost_constants_sha256` — name, value, unit and basis for every assumption, sorted.
  **Rationale prose is excluded on purpose**: editing a justification must not look like
  editing a cost.
- `elkan_thresholds_sha256` — the load-bearing one. Since `policy/elkan.py` computes
  thresholds from costs and effectiveness only, hashing `PolicyResult.thresholds` alone would
  **not** move when the calibrator changes, so it covers the whole per-order threshold *policy
  state*: order sequence, calibrated probability, `p*(o, t)` per tier, and selected tier.

### The seven cross-checks

Six reconcile the parsed half against the live half: primary test rows, primary test
positives, treated orders, treated positives, the action-ladder total against the test-row
count, and SHAP additivity.

The seventh ties the two *reports* to each other rather than each to the live half:
`paired_bootstrap_delta` must equal `average_precision − test_prevalence`, so
`significance.md` and `model_report.md` cannot drift into describing different runs. Its
tolerance is **1.5e-4**, set by the reports' own four-decimal rendering rather than by taste
— 1e-6 is unreachable and would fail on correct data, while the drift it exists to catch, the
hardcoded `+0.0125` that used to sit in `significance.md` §1, is 3.2e-3 out, more than twenty
times the tolerance.

### What the lock does not cover — and where it is currently wrong

**It does not cover the secondary target's correction.** The lock stores
`secondary.average_precision: 0.2938`, the join-artifact-inflated figure, and does not carry
the corrected 0.0190. A reader of the lock alone gets a number the repo has already
overturned. That is the same defect class the lock's own version-2 upgrade was written to
fix for the MDD, and it has not been fixed for the secondary AP.

**Its latency block is stale.** The lock holds 322.1 / 374.5 / 326.4 ms; `eval/latency.md`
was regenerated one commit later and now yields 125.3 / 140.6 / 125.4. Running the lock's own
parser against the committed markdown today returns the latter. The capture script's docstring
says `metadata.captured_at`, `metadata.git_sha` and `latency.*` are "written for the record
and excluded from comparison" — but `tests/test_tier1_lock.py` excluded only the first, so the
generator's stated contract and the assertion enforcing it disagreed, and re-capturing failed
on two fields the generator says are not evidence of anything changing. **That has been fixed:
`UNLOCKED_FIELDS` now implements all three exclusions**, and what the lock exists for — the
evaluation numbers and the three digests — is still compared exactly at 1e-9.

**It does not cover anything not in `eval/`.** The responder artifacts, the write gate, the
break-even table and the replay are outside the lock entirely.

**And the test is opt-in.** `test_tier1_numbers_unchanged` is gated behind
`RUN_TIER1_LOCK=1` because it regenerates the pipeline. It is the one skipped test in the
default suite of 585.

---

## 14. Limitations, and the deferred register

### Limitations

Beyond the proxy label (§1), the Brazilian panel (§1), the batch/serving gap (below), the
latency instability (§12) and the fairness trade (§11):

**The batch/serving selection gap is the most consequential and it is not closed.** The 34
orders the Elkan policy treats are selected on the batch feature matrix, which carries
customer-history, pincode-history, seller and geolocation joins. A webhook payload carries
five fields, so `/score` reports `features_missing: 4` on almost every request — 4 for 32 of
the 34, and 7 for the other two.

Measured by scoring all 34 through the live endpoint:

| | Batch path | Payload path |
|---|---|---|
| Actionable | 34 | **24** (18 `prepaid_only`, 6 `confirm`) |
| `allow` | 0 | **10** |
| Calibrated risk range | 0.81%–5.25% | 1.15%–8.11% |

**Neither path dominates.** Disagreement runs in both directions — one order moves from
`prepaid_only` on the batch path to `confirm` on the payload path — and the payload path's
risk is often *higher*, not lower: one order goes from 4.24% to 8.11%. Roughly seven
decisions in ten survive the transition.

The gap is real and it is not reconciled by any number in this repo. It was found while
building the demo, when hardcoding a "high-risk" batch order showed nothing useful — and the
version of it recorded in a source comment turned out, on re-measurement, to overstate it
substantially. See [BUILD_CHALLENGES.md](BUILD_CHALLENGES.md) incident 7.

**Two positives is not a sample.** Every statement about the treated set rests on 2 failures
in 34 orders. The Wilson interval on that precision is [1.63%, 19.09%], and the break-even
CI's upper bound runs to 3.48e+15% because a bootstrap resample can drive a two-item
denominator to near zero. Reported as computed rather than trimmed.

**`customer_state` is not in the shipped matrix**, so `eval/responder/responder_replay.md`'s
region-proxy table reports a single row reading `unknown | 34`. The real regional breakdown
is `eval/fairness.md`; the replay's version is vacuous and should be read as such.

**No authentication.** No webhook signature verification, no rate limit, no tenant isolation
beyond an `account_id` that is recorded and never checked.

**Single-instance store.** One SQLite file, WAL requested, with a process-level
`threading.Lock` as the single-writer mechanism — correct for one process, wrong for two.

### The deferred register

Every item considered and dropped, with the reason. This is evidence of a filter operating.

| Item | Reason |
|---|---|
| LightGBM/TabPFN router | Olist's ~3% repeat rate leaves the dense branch unevaluable; would put the centrepiece on synthetic data |
| TabPFN | torch + checkpoint ≈ 1–2GB, contradicting the container argument used to reject the address parser; p99 unverified against the budget |
| **EBM ablation** | Cannot resolve at this test size; monotone constraints capture most of the guarantee in five lines. **First item to re-add if a larger test set becomes available** |
| Address parser | Wrong tool — needs a feature vector, not structured fields; latency, explainability, a 2GB data file |
| K-fold OOF target encoding | Leaks under a temporal split; PIT expanding windows instead (§4) |
| Address regex/vocab features | Olist carries no address strings |
| Network / ring layer | Olist carries no device identifiers or phone numbers |
| Tier-3 synthetic generator | Never built. `simulation/` is a different thing — an estimator-validation harness on real covariates |
| Interventional TreeSHAP | Blocked by native categorical splits; route open and costed (§8) |
| `has_item_rows` as a feature | Not usable on Olist — it is the join-cardinality leak (§5) |

---

## Cross-references

[README.md](README.md) · [ARCHITECTURE_RESPONDER.md](ARCHITECTURE_RESPONDER.md) ·
[BUILD_CHALLENGES.md](BUILD_CHALLENGES.md) · [ARCHITECTURE.md](ARCHITECTURE.md), the original
design · and the sixteen generated reports in [`eval/`](eval/) that every figure above is
read from.
