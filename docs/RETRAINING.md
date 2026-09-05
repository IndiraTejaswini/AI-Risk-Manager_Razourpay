# Retraining spec

Extracted verbatim from `ARCHITECTURE.md` §11.4, which named this file before it existed.
The substance below is unchanged from that section; only the surrounding structure and the
cross-references are new.

---

## Cadence

> Calibrator monthly, model quarterly, smECE alarm threshold. v2 retrain activates the
> address and ring extractors once merchant data exists (§4.2).

---

## What each item refers to

Pointers to the parts of this repo the cadence above is about. **No cadence figure here is
new** — this section says where the named things live, not what the schedule should be.

**Calibrator, monthly.** The Platt map fitted on the trailing validation window
(`models/calibration.py`, `CALIBRATION_WINDOW_DAYS = 30`). It is the component most
exposed to prevalence drift: `eval/calibration.md` §5 shows the top decile over-predicted
because the fit window's base rate sits above the test window's, and Platt's intercept
carries that forward. Refitting the calibrator is cheap — it needs scores and labels, not
a retrain.

`eval/drift_slices.md` measures chronological slices on the **secondary** benchmark target
with its prevalence baseline drawn on the figure. It is a diagnostic for the benchmark only:
the result does not transfer to the primary target, and with approximately 126 positives per
slice the current measurement is not resolvable beyond noise.

**Model, quarterly.** The LightGBM booster (`models/train.py`). A retrain changes
`metadata.model_version`, which is a fingerprint of the scoring path rather than of the
serialised bytes (`api/service.py::_version`), so it moves when and only when scores move.

**smECE alarm threshold.** `eval/calibration.md` §4 reports smECE at a stated bandwidth
(σ = 0.005) rather than a binned ECE, because a binned ECE can be driven toward zero by
choosing enough bins. The current value is locked in `eval/TIER1_LOCK.json` under
`targets.primary.performance.smece`. **The alarm threshold itself is not set in this
repo** — §11.4 names it as a thing to have, and it is not one of the figures this build
measured.

**v2 extractors.** The address and ring feature groups are `[DESIGN]`, not built. See
`ARCHITECTURE.md` §4.2 and §0 — `tests/test_feature_builder.py::test_address_and_ring_groups_are_absent`
asserts no column from either group reaches the shipped matrix.

---

## What is not specified here

Stated plainly rather than left to be discovered:

- **No trigger metric or threshold value** for the smECE alarm.
- **No drift detector.** Nothing in this repo watches the input distribution.
- **No rollback procedure** beyond the fact that `model_version` on every response makes
  the serving model identifiable after the fact.

These are gaps in the spec as written, not omissions from this extraction.
