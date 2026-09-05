# Model card — COD return-to-origin risk detector

Per ARCHITECTURE.md §10. Numbers here are reproduced by `make all`; each links to the
artifact that generated it.

| | |
|---|---|
| **Model** | LightGBM binary classifier, 50 trees, monotonic constraints on 3 features |
| **Version** | `rto-label_b-6c0f21786683` — a fingerprint of the model's predictions, stable across platforms |
| **Target** | `label_b`: order entered fulfillment and was never delivered |
| **Estimand** | **P(fails \| ships)** — conditional on the order reaching a carrier |
| **Output** | Calibrated probability, action tier, top-3 risk reasons |
| **Training data** | Olist Brazilian E-Commerce, 2016-09 → 2018-09 |
| **Currency** | **BRL** throughout. No conversion is applied |

---

## Intended use

A **pre-shipment** risk signal for COD/deferred-payment orders on a marketplace, scored
at checkout and consumed by a bounded intervention ladder: `allow → confirm →
prepaid_only → defer`.

The model ranks orders by the probability that, **given the parcel ships**, it will not
reach the customer. The policy layer converts that into an action by comparing per-order
expected costs (§7).

**Appropriate uses**

- Prioritising which orders receive a confirmation prompt or a prepayment request
- Estimating the expected cost of an intervention policy before running it
- As a research artifact: the honest-metrics methodology is the transferable part

**This is a research submission, not a deployed system.** It is evaluated on a Brazilian
e-commerce panel with a proxy label, an assumed cost model, and a policy whose saving
does not resolve statistically. Every one of those is stated below.

## Out-of-scope use

- **Any decision that blocks, refuses, or penalises a customer.** The maximum action is a
  prepayment request. There is no block tier and adding one would invalidate the fairness
  argument in §7.5.
- **Credit, insurance, employment, housing, or any regulated decision.** Nothing here was
  designed or validated for those and the training data has no bearing on them.
- **Individual-level claims about a person.** The model scores an *order*. A high score
  is not a statement that a customer is dishonest; the largest single driver is the
  delivery pincode's historical failure rate, which is a property of geography and
  logistics, not of the person.
- **Deployment on a non-Brazilian market without retraining.** Every cost constant is BRL
  and the pincode encoding is learned on Brazilian CEP prefixes. The API rejects non-BRL
  payloads rather than converting.
- **Unconditional risk.** The output is conditional on shipment. Unconditional risk is
  P(ships) × P(fails \| ships), and P(ships) is not modelled.
- **Automated action without a human-reviewable reason.** The reason strings exist so a
  merchant can contest a flag.

---

## Training data and known biases

**Source.** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
99,441 orders, 2016-09 to 2018-10. Fetched at runtime; SHA-256 of every file read is
recorded in each report.

| Split | Orders | Positives | Prevalence | Window |
|---|---:|---:|---:|---|
| Train | 56,667 | 722 | 1.27% | 2016-09 → 2018-02 |
| Validation | 21,329 | 306 | 1.43% | 2018-02 → 2018-05 |
| Test | 19,662 | **154** | **0.783%** | 2018-05 → 2018-09 |

Chronological, no shuffling. Validation carries both the early-stopping signal and the
calibration fit.

### Known biases

**The label is a proxy, and not for the loss named in the title.** The target is "shipped
and never delivered" on a Brazilian panel. It is **not** COD return-to-origin. Boleto is
the closest structural analog to COD and holds only 61 test positives, so it is reported
as a secondary population and no metrics table is built on it.

**Geography dominates.** `pincode_failure_rate_smoothed` is the largest history-derived
contributor. That is a delivery-infrastructure signal, and delivery infrastructure is not
evenly distributed — so the model will systematically score orders to under-served
regions higher. Two mitigations are in place and one measurement is missing:

- *In place:* the encoding is smoothed with 50 pseudo-observations at the global rate, so
  a pincode with a handful of orders cannot take an extreme value. This is the mechanism
  that stops the model degenerating into a blanket pincode blacklist.
- *In place:* the action ladder never blocks. The worst outcome for a false positive is
  being asked to prepay.
- **Measured:** the §10 fairness tables are built ([`eval/fairness.md`](eval/fairness.md)).
  They do not say what the mitigations were hoped to buy. Sparse-pincode orders are
  treated at 0.127% against 0.267% for medium — under-served, not
  over-flagged, though the gap does not resolve statistically — and 1 state with an
  above-panel failure rate receives no interventions at all. The claim "rural orders are
  not downgraded disproportionately" is now *measured and holds*; the claim that they are
  adequately protected does not.

**Temporal drift is real.** Validation prevalence is 1.435% against 0.783% in test. The
calibration window was selected to minimise that mismatch, and the top decile is still
over-predicted by 1.31×.

**Customer history is 97% empty.** Only ~3% of customers place more than one order, so
every expanding-window feature is null or zero for almost every order. Two of the three
monotonic constraints sit on features that never split and carry *exactly zero*
attribution.

**Seller concentration is mild but present.** The top 10 sellers account for 17.24% of
failing order–seller pairs against 12.41% of volume.

### No protected or proxy attributes

**No feature encodes or proxies name, religion, gender, or language.**

The feature matrix is enumerated in `eval/feature_report.md` — 35 columns across order
composition, customer history, pincode, and availability. It contains no name, no gender,
no religion, no language, no age, no ethnicity, and no free-text field from which any of
them could be inferred. The customer identity used is an opaque hash
(`customer_unique_id`); the only demographic-adjacent field is the postal prefix, which is
present because delivery risk is genuinely geographic.

The column whitelist in `data/COLUMN_WHITELIST.md` is machine-parsed and enforced: a
column absent from it is rejected at load, not ignored.

**Nineteen more features were built and are not in this model.** Seller history, route
(customer state, seller state, origin-destination encoding, great-circle distance),
parcel structure, and rolling order density — built, passed every gate, and rejected by
a paired bootstrap whose 95% interval on ΔAP includes zero
([`eval/feature_expansion.md`](eval/feature_expansion.md)). They matter to this section
because they are the geography features someone reading the list above would ask for,
and the result is instructive: adding `customer_state` made it the single
most-attributed feature in the model **and made the model worse**. Naming geography
explicitly did not add information.

This is a claim about the feature list, and it is worth being precise about its status:
**Olist carries no name, religion, gender, language, age or ethnicity field, so there is
nothing to correlate a feature against. The claim is a design claim, verifiable by
reading the feature list, and not a measured one.** No experiment in this repo could
have falsified it.

What *is* measured is which features carry geography implicitly
([`eval/fairness.md`](eval/fairness.md) §4). The answer is not the obvious one: the
strongest carrier of customer state is **`order_freight`** at 12.1% of the entropy
in state — shipping cost is a function of distance — and it outranks every feature named
for geography. A model with the pincode group removed would still see it.

The converse was also tested. Admitting `customer_state` *by name* — the most direct
geographic feature available — did not improve the model
([`eval/feature_expansion.md`](eval/feature_expansion.md)). Both results point the same
way: on this panel, geography is already in the model through cost and distance, and the
named field adds a place to overfit rather than a new signal.

This verifies the feature list, **not** disparate impact. A model can contain no
protected attribute and still produce disparate outcomes through a correlated one, which
is exactly what geography is here. The measurement that tests outcomes is the
intervention-rate table above, and it found a real disparity.

---

## Metrics

> **Section 5.1 metrics are on observed outcomes. Section 5.2 metrics are on data we
> generated and are therefore an upper bound.**
>
> This repo contains **no Section 5.2**. Nothing here derives from generated data: the
> synthetic generator was never built. Every number below is on observed Olist outcomes.

### Primary target — `label_b`, P(fails \| ships), risk set

154 test positives, 0.783% prevalence, 19,662 test orders.
Source: [`eval/model_report.md`](eval/model_report.md), [`eval/significance.md`](eval/significance.md).

| Metric | Value |
|---|---:|
| Average precision | **0.0171** |
| Prevalence baseline | 0.0078 |
| Permutation test vs random ranking | **p = 0.00060** |
| Paired bootstrap vs prevalence, 95% CI | **[+0.0034, +0.0279]** |
| ROC-AUC (not the headline) | 0.6450, CI [0.6050, 0.6850] |
| Precision@1% / recall@1% | 3.05% / 3.90% |
| Precision@5% / recall@5% | 1.42% / 9.09% |

**The model separates from chance.** It does not follow that it is useful — see the cost
table.

### Calibration

Platt, 30-day window. Source: [`eval/calibration.md`](eval/calibration.md).

| Metric | Value |
|---|---:|
| Brier (calibrated) | 0.0077613 |
| smECE (σ = 0.005) | 0.002510 |
| Top-decile predicted / observed | 1.79% / 1.37% (**1.31×**) |
| **Maximum calibrated probability** | **6.86%** |

The ceiling is load-bearing: any Elkan threshold above 6.86% selects nothing, so a policy
built against one is inert.

### Cost policy

BRL, test window. Source: [`eval/policy.md`](eval/policy.md).

| Policy | Treated | Cost | Paired Δ vs nothing | 95% CI |
|---|---:|---:|---:|---|
| Nothing | 0 | 8,613.13 | — | — |
| Everything | 19,662 | 87,881.68 | +79,268.55 | [+77,406, +81,225] |
| Hand rule | 1,872 | 40,839.15 | +32,226.02 | [+30,374, +34,160] |
| **Model + cost policy** | 34 | 8,467.61 | **−145.52** | **[−595.77, +34.83]** |

**The model policy's saving does not resolve.** The paired CI includes zero and the
policy was worse than doing nothing in 26.1% of resamples.

### Metrics by segment

§10 asks for performance broken out by region tier. **Olist is Brazilian and has no
Indian metro/tier-2/tier-3 field**, so two derived substitutes are reported instead, with
their derivations stated: pincode observation density (training orders per zip prefix)
and customer state. Full tables, bootstrap CIs and the action-ladder composition per
bucket: [`eval/fairness.md`](eval/fairness.md).

**The intervention-rate finding runs opposite to the concern §10 is written against.**

| Density bucket | n | Positives | Prevalence | Intervention rate |
|---|---:|---:|---:|---:|
| sparse (<10 training orders/prefix) | 13,437 | 107 | 0.796% | **0.127%** |
| medium (10–49) | 5,996 | 47 | 0.784% | **0.267%** |
| dense (≥50) | 229 | 0 | 0.000% | 0.437% |

Sparse pincodes are treated **least**, not most, though the bootstrap CI on the
sparse-vs-medium gap spans the panel rate, so this does not resolve statistically. They
carry **69.5%** of the test window's failures and receive **50.0%** of the
interventions. They are also escalated less far up the ladder.

At state level, **1 state with an above-panel failure rate receives zero
interventions** — DF fails at 0.847%, 1.1×
the panel rate, across 472 orders, and the policy acts on none of them.

**The mechanism is not a bug.** The smoothing that prevents a pincode blacklist also
keeps thin prefixes below the threshold at which intervening pays, and the per-order
Elkan threshold declines to protect regions whose orders are individually cheap to lose.
Any cost-optimal policy does this. Mitigating it needs an intervention floor or a
fairness constraint in the objective — a product decision, and one this repo does not
make silently.

**Pincode ablation (§9 item 13):** removing every pincode-derived feature costs
**0.0072 AP**, 95% CI [+0.0004, +0.0253], which excludes zero — **42% of the model**.
The features cannot be dropped as a liability reduction, because they are carrying the
model. Its predictive value and its principal fairness exposure are the same feature
group.

**Smoothing, verified:** with m = 50 pseudo-observations, a prefix with fewer than 10
prior orders deviates from the global prior by 0.1893% on average, against an
algebraic ceiling of k/(k+50). An extreme value for a rarely-seen pincode is
arithmetically unreachable, not merely discouraged.

Also available, from earlier steps:

- **By calibrated-probability band** — [`eval/calibration.md`](eval/calibration.md) §6.
- **By risk reason** — [`eval/reasons.md`](eval/reasons.md) §6.
- **By label definition** — [`eval/label_composition.md`](eval/label_composition.md) §4.

---

## Limitations

### Responder is built but unevaluated

The bounded responder is **[BUILT-UNEVAL]** (T1.1–T6.1). Its replay reports
mechanism behaviour over the complete Tier-1 test denominator, not intervention
effect. The reason-by-tier effectiveness matrix is assumed, Olist has no contact
details, and Tier 2 validates the estimator only on semi-synthetic response with
known oracle outcomes (T4.1, T5.1).

Full list in [README.md](README.md#limitations). The ones that bear on responsible use:

1. **The saving does not resolve.** At 34 treated orders the policy cannot be shown to
   pay for itself. Deploying on the strength of the point estimate would be acting on
   noise.
2. **Every cost is an assumption** applied to Brazilian orders, and one — the message
   fatigue allowance — is an outright guess, tagged `GUESS` in `policy/constants.py`.
3. **Effectiveness is unmeasurable offline.** The reason × tier matrix is assumed. The two
   sensitivity axes are not separately identifiable: the policy responds to their product.
4. **The fairness measurement now exists and is unflattering.** Sparse pincodes are
   under-intervened relative to medium ones (though that gap does not resolve
   statistically) and one high-failure state receives no interventions at all; the
   pincode features that create that exposure carry 42% of the model's average
   precision, so they cannot be dropped without dropping the model.
5. **The proxy label is not RTO**, and roughly 60% of the broader label's positives never
   reached a carrier at all.
6. **Latency is reported as a range, not a verdict** — across five runs of unchanged code at the same commit, p50 landed between ~90ms and ~325ms and p99 between ~100ms and ~375ms; the budget is met when the host is quiet and missed when it is not. The variation is between invocations rather than within them (one run's passes sat at 314.1 / 315.8 / 318.2ms, another's at 120.7 / 122.0 / 326.6ms), so a single measurement cannot establish which regime the host is in and no pass/fail claim is made. See `eval/latency.md`. The order-of-magnitude improvement from ~3250ms is far larger than that noise band; it came from precomputing the store's per-group sums, a change made for float exactness, not speed.
7. **The address and ring feature groups do not exist.** The production feature contract
   describes them; no code implements them.

---

## Contact and provenance

Reproduce with `make all` from a clean checkout. Deterministic: two runs produce identical
numbers. Dataset SHA-256s are recorded in every report, and 266 tests pin every figure
that appears in an `eval/` artifact.
