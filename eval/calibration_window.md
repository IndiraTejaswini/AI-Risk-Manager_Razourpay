# Calibration window: provisional selection

**Decision record.** No code changes accompany this file. The selection here is
**provisional** and is confirmed or overturned in Step 4 by Brier score on the validation
window, which is the mechanism ARCHITECTURE.md §6 specifies.

Evidence: [`eval/feature_report.md`](feature_report.md) §3, produced by
`scripts/03_feature_matrix.py`. Candidates are trailing windows ending at the test
boundary `2018-05-24 16:58:49`, evaluated on the primary target `label_b` over the risk
set.

---

## Selected: 30 days

Selected on **prevalence-drift grounds**, not on sample size.

| Window | Orders | Positives | Prevalence | Ratio vs test | Events/param | Selected |
|---:|---:|---:|---:|---:|---:|---|
| **30d** | 7,380 | **71** | **0.962%** | **1.23×** | 35.5 | **✓** |
| 60d | 14,148 | 173 | 1.223% | 1.56× | 86.5 | |
| 90d | 21,329 | 306 | 1.435% | 1.83× | 153.0 | |

Test window prevalence: **0.783%**.

---

## Why not sample size

**No candidate was excluded on positive count.** Platt scaling fits two parameters — a
slope and an intercept. Against Peduzzi's 10-events-per-parameter floor for a stable
logistic fit, the admissibility bar is 20 positives. The thinnest candidate carries 71,
which is 35.5 events per parameter. All three clear it comfortably.

So the exclusion rule that Step 1 was asked to apply did not fire, and saying so is the
finding: **sample size is not the binding constraint on this panel.** Reporting "none
excluded" is more useful than reporting a rule that happened to have nothing to do.

## Why prevalence drift decides it

Prevalence rises monotonically as the window widens backwards — 0.962% → 1.223% →
1.435%. The failure rate was higher earlier in the panel and declining toward the
boundary. Every candidate therefore sits above the 0.783% base rate of the window the map
will actually be applied to, and the wider the window the worse the mismatch.

> **A Platt map fit at one base rate and applied at another is miscalibrated by
> construction.**

Platt fits `P = 1 / (1 + exp(A·f + B))`. The intercept `B` absorbs the base rate of the
data it was fit on. Fit at 1.435% and apply at 0.783% and the map carries that 1.83×
prior with it — every probability it emits is inflated, systematically, in a way no
amount of ranking quality corrects. The policy layer in §7 consumes probabilities, not
ranks, so this lands directly on the decision threshold rather than being a cosmetic
concern.

The 30-day window minimises that mismatch at 1.23×, and it still carries enough positives
to fit two parameters several times over. That is the trade the selection makes: give up
positives that were surplus to the fit's requirements in exchange for the closest
available base rate.

## What would overturn this

The 30-day window is the closest match on prevalence but the noisiest fit — 71 positives
against 306. If the variance cost of the shorter window exceeds the bias cost of the base
rate mismatch, Brier will say so, because Brier is a proper scoring rule and penalises
both.

**Step 4 runs that comparison and the result stands regardless of this record.** If Brier
selects 60d or 90d, that supersedes this file and the reason is recorded there — the
selection is not re-argued from prevalence alone. This document exists so the prevalence
evidence is on record *before* the Brier number is known, and cannot be retrofitted to
whatever Step 4 returns.

## Caveat carried forward

Validation carries both the early-stopping signal (§5.2) and the Platt fit (§6), which is
what §6 specifies. It does mean the calibration map is fit on data the model early-stopped
against. Stated rather than hidden; it is a mild optimism in the calibration numbers, not
in the ranking numbers.

---

## Resolution (Step 4)

Appended by `scripts/07_calibrate.py`. The text above is unchanged: it records what was argued before any Brier number existed.

**Agrees.** Cross-fitted Brier on the validation window is lowest at **30 days** - but the candidates are statistically indistinguishable, so this is a **comparison, not a selection**. `CALIBRATION_WINDOW_DAYS = 30` is a fixed choice hardcoded in five places; nothing downstream reads this table. The comparison exists to show the choice is not load-bearing, not to make it. See [`calibration.md`](calibration.md) §2 for the margin and its confidence interval.

| Window | Cross-fitted Brier | Prevalence | Ratio vs test |
|---:|---:|---:|---:|
| 30d **←** | 0.0095359 | 0.962% | 1.23× |
| 60d | 0.0095436 | 1.223% | 1.56× |
| 90d | 0.0095617 | 1.435% | 1.83× |

Full reasoning, including why the cross-fitted comparison is used rather than a naive in-sample Brier, is in [`calibration.md`](calibration.md) §2.
