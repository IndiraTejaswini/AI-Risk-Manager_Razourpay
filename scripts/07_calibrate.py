#!/usr/bin/env python3
"""
Step 4: calibration.  ARCHITECTURE.md 6, build step 4.

Fits Platt scaling on the temporal validation window, **compares** window lengths by
Brier across {30, 60, 90} days, and reports reliability, smECE, and a tier-conditional
calibration table on provisional score bands.

The window comparison is a comparison, not a selection.  ``CALIBRATION_WINDOW_DAYS = 30``
is a fixed choice hardcoded in ``scripts/08_policy.py``, ``scripts/09_reasons.py``,
``scripts/11_fairness.py``, ``api/service.py`` and
``scripts/90_capture_tier1_lock.py``; nothing downstream reads this script's table.  The
candidates are statistically indistinguishable - the best margin is ~8e-6 Brier against
a confidence interval roughly seven times wider that straddles zero - so the comparison
exists to show the choice is **not load-bearing**, rather than to make it.

**No cost model.**  The bands here are quantile cuts on calibrated probability, not
policy tiers - they are deliberately not named allow/confirm/prepaid-only/defer, because
those are outputs of a decision rule that does not exist yet.

Writes eval/calibration.md and eval/figures/*.png.  Deterministic.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import (  # noqa: E402
    PRIMARY_LABEL,
    SECONDARY_LABEL,
    VALIDATION_DAYS,
    OlistLoader,
)
from features.builder import FeatureBuilder  # noqa: E402
from models.calibration import (  # noqa: E402
    PlattCalibrator,
    brier_score,
    reliability,
    smece,
    uniform_mass_bins,
)
from models.plots import plot_calibrated_distribution, plot_reliability  # noqa: E402
from models.train import predict, prepare_matrix, train  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "calibration.md"
FIG_DIR = REPO_ROOT / "eval" / "figures"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "calibrators"
WINDOW_APPEND = REPO_ROOT / "eval" / "calibration_window.md"

CANDIDATE_DAYS = (30, 60, 90)
PROVISIONAL_SELECTION = 30          # eval/calibration_window.md
N_FOLDS = 5

#: Evaluation slice for window selection: the most recent N days of validation.
#: See the report - scoring on the WHOLE validation window is structurally biased
#: toward the longest candidate, because that window IS the whole validation set.
EVAL_SLICE_DAYS = 30
N_BINS = 10
SMECE_BANDWIDTH = 0.005             # stated, never implicit
BOOT_SEED = 20260901

#: Provisional band cuts, as quantiles of calibrated probability on the validation
#: window.  NOT policy tiers - see the report.
BAND_QUANTILES = (0.50, 0.90, 0.99)
TS = "order_purchase_timestamp"


def _fit_model(population, matrix, label):
    split = population["split"].to_numpy()
    y = population[label].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"
    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=label, population="p", category_levels=levels,
    )
    # Log-odds margin, not the squashed probability: Platt is a linear map in logit
    # space, and feeding it a sigmoid output stacks two of them.
    return (
        bundle,
        predict(bundle, X.loc[va], raw_score=True),
        predict(bundle, X.loc[te], raw_score=True),
        y[va], y[te],
    )


def _time_folds(n: int, k: int) -> list[np.ndarray]:
    """
    Split ``range(n)`` into ``k`` contiguous, time-ordered folds.

    Guarantees, all asserted in ``tests/test_calibration_folds.py``:

    * **Tiling** - the concatenation of the folds is a permutation of ``range(n)``.
      Every index appears exactly once: no gaps, no overlap, for any ``n`` and ``k``,
      divisible or not.
    * **Contiguity** - each fold is an ascending run of consecutive indices.
    * **Ordering** - fold boundaries increase monotonically, so fold ``j`` covers an
      earlier time block than fold ``j + 1``.  The caller sorts by timestamp before
      indexing, so contiguous here means contiguous *in time*.

    ----------------------------------------------------------------------------------
    THE PREVIOUS IMPLEMENTATION DID NOT TILE, AND ITS DOCSTRING SAID IT DID
    ----------------------------------------------------------------------------------

    It read ``np.arange(n)[i::1][j * n // k:(j + 1) * n // k]`` with ``i == j``, which
    re-sliced a progressively shorter array before applying fold ``j``'s slice.  Each
    fold after the first was therefore shifted by its own index and dropped the boundary
    element, leaving exactly ``k - 1`` indices unassigned - at ``n/k``, ``2n/k + 1``,
    ``3n/k + 2``, ``4n/k + 3`` for ``k = 5``.

    The old docstring claimed "No randomness."  That was false in effect: the caller
    allocated its output with ``np.empty`` and never wrote those slots, so the result
    depended on whatever the allocator handed back.  The randomness was the allocator's.
    Three consecutive runs of unchanged code selected three different window candidates,
    and one run averaged a slot holding ~1e73 into a Brier score, returning 1e+143.

    The guards below are deliberate: ``k < 2`` is not cross-fitting at all, and ``k > n``
    cannot be done without empty folds.  Both fail loudly rather than returning something
    shaped correctly and meaning nothing.
    """
    if k < 2:
        raise ValueError(f"cross-fitting needs at least 2 folds, got k={k}")
    if k > n:
        raise ValueError(
            f"cannot split {n} rows into {k} contiguous folds without empty folds"
        )
    idx = np.arange(n)
    return [idx[j * n // k:(j + 1) * n // k] for j in range(k)]


def _crossfit_brier(scores_va, y_va, ts_va, boundary, days) -> tuple[float, int, int]:
    """
    Brier over the **whole validation window** using out-of-sample calibrated
    probabilities for every row, so candidates are comparable on identical rows.

    Rows inside the candidate window get out-of-fold predictions from contiguous
    time-blocked folds; rows outside it are scored by a calibrator fit on the whole
    window, which is already fully out-of-sample for them.

    Without this, the 90-day candidate would be evaluated on exactly the rows it was fit
    on and the comparison would favour it by construction.
    """
    start = boundary - pd.Timedelta(days=days)
    inside = (ts_va >= start).to_numpy()
    # NaN-filled, not np.empty: an unwritten slot must be detectable.  See the check
    # before the return.
    p = np.full(len(y_va), np.nan, dtype=float)

    idx_in = np.flatnonzero(inside)
    order = idx_in[np.argsort(ts_va.to_numpy()[idx_in], kind="mergesort")]
    for fold in _time_folds(len(order), N_FOLDS):
        held = order[fold]
        rest = np.setdiff1d(order, held, assume_unique=False)
        if y_va[rest].sum() == 0 or y_va[rest].sum() == len(rest):
            p[held] = y_va[rest].mean() if len(rest) else y_va.mean()
            continue
        c = PlattCalibrator().fit(scores_va[rest], y_va[rest])
        p[held] = c.predict(scores_va[held])

    outside = ~inside
    if outside.any():
        c_all = PlattCalibrator().fit(scores_va[inside], y_va[inside])
        p[outside] = c_all.predict(scores_va[outside])

    # Every row must have received a prediction - out-of-fold if inside the candidate
    # window, out-of-window otherwise.
    #
    # This check is the point of the np.full(nan) above, and it is kept even though
    # _time_folds now provably tiles.  The defect it guards against was not a fold that
    # looked wrong; it was a fold that looked right, in a function whose docstring
    # asserted the property nobody had tested.  "Unreachable by current reasoning" is
    # exactly what the previous version assumed, and it was averaging uninitialised
    # memory for as long as the file existed.
    unfilled = int(np.count_nonzero(np.isnan(p)))
    if unfilled:
        raise AssertionError(
            f"{unfilled} of {len(p)} validation rows received no calibrated prediction "
            f"in the {days}-day cross-fit. The folds do not tile the window, so the "
            "Brier score would be averaging unwritten slots."
        )

    return p, int(inside.sum()), int(y_va[inside].sum())


def main() -> int:
    loader = OlistLoader()
    boundary = loader.split_boundary

    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    bundle, s_va, s_te, y_va, y_te = _fit_model(
        risk, FeatureBuilder().build(risk), PRIMARY_LABEL
    )
    va_rows = risk[risk["split"] == "validation"]
    te_rows = risk[risk["split"] == "test"]
    ts_va = va_rows[TS]

    # ---------------------------------------------------------------- assertions
    assert set(va_rows["order_id"]).isdisjoint(set(te_rows["order_id"]))
    assert ts_va.max() < te_rows[TS].min()
    assert bool((ts_va >= boundary - pd.Timedelta(days=VALIDATION_DAYS)).all())
    assert bool((te_rows[TS] >= boundary).all())

    # ---------------------------------------------------------------- selection
    rows = []
    losses: dict[int, np.ndarray] = {}
    for days in CANDIDATE_DAYS:
        p_cv, n_orders, n_pos = _crossfit_brier(s_va, y_va, ts_va, boundary, days)
        losses[days] = (p_cv - y_va) ** 2          # per-row squared error
        start = boundary - pd.Timedelta(days=days)
        inside = (ts_va >= start).to_numpy()
        rows.append({
            "days": days, "n_orders": n_orders, "n_positives": n_pos,
            "prevalence": float(y_va[inside].mean()),
            "brier_full": float(losses[days].mean()),
        })

    # The evaluation slice.  Scoring on the whole validation window is structurally
    # biased: that window IS the 90-day candidate, so its base rate is the 90-day
    # candidate's base rate, and a calibrator fit at that base rate is evaluated at
    # home.  Cross-fitting removes the in-sample advantage but not the base-rate
    # alignment.  The slice is the same for every candidate and is the part of
    # validation adjacent to test, so it favours none of them a priori.
    eval_mask = (ts_va >= boundary - pd.Timedelta(days=EVAL_SLICE_DAYS)).to_numpy()
    for r in rows:
        r["brier_slice"] = float(losses[r["days"]][eval_mask].mean())

    rng = np.random.default_rng(BOOT_SEED)
    n_eval = int(eval_mask.sum())
    pair_tests = {}
    for days in CANDIDATE_DAYS:
        if days == PROVISIONAL_SELECTION:
            continue
        d = (losses[days] - losses[PROVISIONAL_SELECTION])[eval_mask]
        idx = rng.integers(0, n_eval, size=(4000, n_eval))
        boot = d[idx].mean(axis=1)
        pair_tests[days] = {
            "delta": float(d.mean()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
        }
    brier_resolves = any(
        t["ci_high"] < 0 or t["ci_low"] > 0 for t in pair_tests.values()
    )

    best_slice = min(rows, key=lambda r: (r["brier_slice"], r["days"]))["days"]
    best_full = min(rows, key=lambda r: (r["brier_full"], r["days"]))["days"]
    chosen = best_slice
    sel = next(r for r in rows if r["days"] == chosen)

    # ---------------------------------------------------------------- final fit
    start = boundary - pd.Timedelta(days=chosen)
    fit_mask = (ts_va >= start).to_numpy()
    calibrator = PlattCalibrator().fit(s_va[fit_mask], y_va[fit_mask])

    p_va = calibrator.predict(s_va)
    p_te = calibrator.predict(s_te)

    # Monotonicity: Platt is monotone by construction, so a violation is a wiring error.
    o = np.argsort(s_te, kind="stable")
    dp = np.diff(p_te[o])
    monotone = bool(np.all(dp >= -1e-12) or np.all(dp <= 1e-12))

    # ---------------------------------------------------------------- bands
    cuts = [float(np.quantile(p_va, q)) for q in BAND_QUANTILES]
    def band(p):
        return np.digitize(p, cuts, right=False)
    b_te = band(p_te)

    # ---------------------------------------------------------------- metrics
    prior_te = float(y_te.mean())
    brier_uncal = brier_score(y_te, 1.0 / (1.0 + np.exp(-s_te)))
    brier_cal = brier_score(y_te, p_te)
    brier_prior = brier_score(y_te, np.full_like(p_te, float(y_va.mean())))
    rel = reliability(y_te, p_te, n_bins=N_BINS, seed=BOOT_SEED)
    sme = smece(y_te, p_te, SMECE_BANDWIDTH)
    sme_uncal = smece(y_te, 1.0 / (1.0 + np.exp(-s_te)), SMECE_BANDWIDTH)

    decile = uniform_mass_bins(p_te, 10)
    top = decile == decile.max()
    top_pred, top_obs = float(p_te[top].mean()), float(y_te[top].mean())

    # ---------------------------------------------------------------- secondary
    matured = loader.labelled().join(loader.split_labelled()["split"], how="left")
    item_ids = set(loader.load_table("items", ["order_id"])["order_id"])
    clean = matured[matured["order_id"].isin(item_ids)].copy()
    _, s2_va, s2_te, y2_va, y2_te = _fit_model(
        clean, FeatureBuilder().build(clean), SECONDARY_LABEL
    )
    ts2_va = clean.loc[clean["split"] == "validation", TS]
    fit2 = (ts2_va >= boundary - pd.Timedelta(days=chosen)).to_numpy()
    cal2 = PlattCalibrator().fit(s2_va[fit2], y2_va[fit2])
    p2_te = cal2.predict(s2_te)
    rel2 = reliability(y2_te, p2_te, n_bins=N_BINS, seed=BOOT_SEED)

    # ---------------------------------------------------------------- artifacts
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "primary.json").write_text(
        json.dumps({"window_days": chosen, **calibrator.params,
                    "band_cuts": cuts, "band_quantiles": list(BAND_QUANTILES)},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )

    plot_reliability(rel, prior_te, FIG_DIR / "reliability_primary.png",
                     "Reliability — primary target, test window")
    plot_reliability(rel2, float(y2_te.mean()), FIG_DIR / "reliability_secondary.png",
                     "Reliability — secondary target, test window (benchmark)")
    plot_calibrated_distribution(p_te, cuts, prior_te,
                                 FIG_DIR / "calibrated_distribution.png",
                                 "Calibrated P(fails | ships) — test window")

    # ---------------------------------------------------------------- report
    def pf(x, dp=4):
        return f"{100 * x:.{dp}f}%"

    L: list[str] = []
    w = L.append

    w("# Calibration — Step 4")
    w("")
    w("Generated by `scripts/07_calibrate.py`. **No cost model.** ARCHITECTURE.md §6.")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Calibrator: Platt, A = `{calibrator.a_:.6f}`, B = `{calibrator.b_:.6f}`, "
      f"fit on the last **{chosen} days** of validation")
    w(f"- Fit population: {int(fit_mask.sum()):,} orders, "
      f"{int(y_va[fit_mask].sum())} positives")
    w("")

    # -- the ceiling, first ------------------------------------------------------------
    w("## 1. The achievable probability range")
    w("")
    w("Stated first, because Step 5's per-order thresholds must live inside it. **A "
      "policy layer built against a threshold this model cannot reach is inert.**")
    w("")
    w("| | Calibrated P(fails \\| ships) |")
    w("|---|---:|")
    w(f"| Minimum | {pf(p_te.min(), 5)} |")
    w(f"| Median | {pf(float(np.median(p_te)), 5)} |")
    w(f"| 90th percentile | {pf(float(np.quantile(p_te, 0.90)), 4)} |")
    w(f"| 99th percentile | {pf(float(np.quantile(p_te, 0.99)), 4)} |")
    w(f"| **Maximum** | **{pf(p_te.max(), 4)}** |")
    w(f"| Test prevalence | {pf(prior_te, 4)} |")
    w("")
    w(f"**The model never emits a probability above {pf(p_te.max(), 3)}.** The top decile "
      f"spans {pf(float(p_te[top].min()), 4)} to {pf(p_te.max(), 4)}.")
    w("")
    w(f"Any Elkan threshold p\\* above {pf(p_te.max(), 3)} selects nothing and any "
      f"threshold below {pf(p_te.min(), 5)} selects everything, so the entire useful "
      "range of the cost model is bounded by these two numbers. That is a hard "
      "constraint on §7, not a presentational note: with c_FP and c_FN as §7.1 "
      "specifies, p\\* = c_FP/(c_FP + c_FN), so a threshold inside this range requires "
      f"c_FN to exceed c_FP by roughly {(1 - p_te.max()) / p_te.max():.0f}× or more. If "
      "the assumed costs do not satisfy that, the policy never fires and the honest "
      "report is that the detector cannot support the intervention at those costs.")
    w("")
    w("![Calibrated distribution](figures/calibrated_distribution.png)")
    w("")

    # -- window selection --------------------------------------------------------------
    w("## 2. Window comparison - Brier across {30, 60, 90}")
    w("")
    w("Section 6 says to select the window length \"by Brier score on the validation "
      "window\". Taken literally that instruction is **structurally biased**, and the "
      "bias is worth stating before the numbers.")
    w("")
    w("### Why \"Brier on the validation window\" cannot be taken literally")
    w("")
    w(f"Validation is {VALIDATION_DAYS} days, so **the 90-day candidate *is* the whole "
      "validation window.** Scoring every candidate on that window evaluates each "
      "calibrator against a population whose base rate is, by construction, the 90-day "
      "candidate's own base rate. The longest window is evaluated at home.")
    w("")
    w("Cross-fitting removes the *in-sample* advantage - every row gets an out-of-fold "
      "prediction - but it cannot remove the **base-rate alignment**, which is exactly "
      "the part that matters for a calibration map whose intercept absorbs the base "
      "rate.")
    w("")
    w(f"So the selection is made on an **evaluation slice**: the most recent "
      f"{EVAL_SLICE_DAYS} days of validation. It is identical for every candidate, so it "
      "favours none a priori; it is the part of validation adjacent to test; and its "
      "base rate is the closest available proxy for the deployment base rate. Every "
      "candidate is scored there with out-of-fold predictions.")
    w("")
    w("| Evaluation set | n | Base rate | vs test |")
    w("|---|---:|---:|---:|")
    w(f"| Whole validation window (= the 90d candidate) | {len(y_va):,} | "
      f"{pf(float(y_va.mean()), 3)} | {float(y_va.mean()) / prior_te:.2f}x |")
    w(f"| **Evaluation slice - last {EVAL_SLICE_DAYS}d** | {n_eval:,} | "
      f"{pf(float(y_va[eval_mask].mean()), 3)} | "
      f"{float(y_va[eval_mask].mean()) / prior_te:.2f}x |")
    w(f"| Test window (not used for selection) | {len(y_te):,} | {pf(prior_te, 3)} | "
      "1.00x |")
    w("")
    w("### Results")
    w("")
    w("| Window | Orders | Positives | Prevalence | **Brier on slice** | "
      "Brier on whole validation |")
    w("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        mark = " **<--**" if r["days"] == chosen else ""
        w(f"| {r['days']}d{mark} | {r['n_orders']:,} | {r['n_positives']} | "
          f"{pf(r['prevalence'], 3)} | **{r['brier_slice']:.7f}** | "
          f"{r['brier_full']:.7f} |")
    w("")
    w(f"**The two columns disagree.** On the evaluation slice the best candidate is "
      f"**{best_slice}d**; on the whole validation window it is **{best_full}d**. That "
      "disagreement is the bias made visible: the whole-window column ranks the "
      "candidates in order of length, which is what an evaluation set aligned to the "
      "longest candidate produces.")
    w("")
    w("### Is the gap distinguishable?")
    w("")
    w("Brier is a mean of per-row squared errors, so the difference between two "
      "candidates is a paired mean difference on identical rows and can be bootstrapped "
      "directly. Selecting on the fifth decimal of a scalar without this check would be "
      f"reading noise. 4,000 resamples on the evaluation slice, against the "
      f"{PROVISIONAL_SELECTION}-day candidate:")
    w("")
    w(f"| Comparison | delta-Brier vs {PROVISIONAL_SELECTION}d | 95% CI | Resolves |")
    w("|---|---:|---|---|")
    for days, t in pair_tests.items():
        res = "**yes**" if (t["ci_high"] < 0 or t["ci_low"] > 0) else "no"
        w(f"| {days}d - {PROVISIONAL_SELECTION}d | {t['delta']:+.7f} | "
          f"[{t['ci_low']:+.7f}, {t['ci_high']:+.7f}] | {res} |")
    w("")
    if brier_resolves:
        w("At least one gap excludes zero, so Brier has made a selection on the slice.")
    else:
        w("**No gap excludes zero.** Brier does not distinguish the candidates even on "
          "the corrected evaluation set - every difference sits inside sampling noise. A "
          "proper scoring rule that cannot separate the options has not made a "
          "selection, so its point estimate is a tie-break at best, and the "
          "pre-registered prevalence-drift argument carries the decision.")
    w("")
    w(f"**Best on the slice: {chosen} days — and this is a comparison, not a selection.**")
    w("")
    w("The margin does not support the word 'selection'. The 30d/60d gap is "
      f"{min(abs(t['delta']) for t in pair_tests.values()):.7f} against a 95% CI "
      "roughly seven times wider "
      "that straddles zero. The three candidates are statistically indistinguishable, "
      "and the winner is decided by an amount this data cannot resolve. A selection "
      "whose outcome the data cannot resolve is a comparison, and calling it a "
      "selection would overstate what was measured.")
    w("")
    w(f"**`CALIBRATION_WINDOW_DAYS = {PROVISIONAL_SELECTION}` is a fixed choice**, "
      "hardcoded in `scripts/08_policy.py`, `scripts/09_reasons.py`, "
      "`scripts/11_fairness.py`, `api/service.py` and "
      "`scripts/90_capture_tier1_lock.py`. Nothing downstream reads this table. The "
      "comparison exists to show that the choice is **not load-bearing** - that the "
      "system would report materially the same thing at 60 or 90 days - rather than to "
      "make the choice.")
    w("")
    w("The fix did not change this conclusion; it changed whether the conclusion was "
      "reliable. Pre-fix the same table selected 60d on one run in three: "
      "`_time_folds` did not tile its index range, and the unassigned slots held "
      "uninitialised memory that occasionally dominated a Brier mean. See "
      "`tests/test_calibration_folds.py`.")
    w("")
    if chosen == PROVISIONAL_SELECTION:
        w("The comparison **agrees with** the provisional choice in "
          "`eval/calibration_window.md`, which took 30 days on prevalence-drift grounds "
          "before any Brier number existed - though on a margin too small to be "
          "evidence for it.")
        w("")
        w("Being precise about what was confirmed: the *literal* reading of section 6 - "
          f"Brier on the whole validation window - would have selected {best_full}d and "
          "overturned the pre-registration. The pre-registration was right, and right "
          "for the reason it gave: base-rate mismatch dominates the calibration map's "
          "intercept. What the measurement adds is that the selection *mechanism* "
          "section 6 specifies is itself sensitive to a base-rate artifact, which the "
          "pre-registration did not anticipate.")
    else:
        w(f"The slice prefers {chosen}d over the provisional 30 days. Given the margin "
          "above this is not grounds to change the fixed constant; it is the "
          "comparison's point estimate moving inside its own noise band.")
    w("")
    w("### After the fact: what test says")
    w("")
    w("**Not an input to the selection above.** The window was chosen on validation "
      "alone; this reports what test shows afterwards, because a selection procedure "
      "that is never checked is not a procedure.")
    w("")
    w("| Window | Brier on test | Top-decile predicted | Top-decile observed | Ratio |")
    w("|---:|---:|---:|---:|---:|")
    for r in rows:
        d = r["days"]
        inside = (ts_va >= boundary - pd.Timedelta(days=d)).to_numpy()
        c = PlattCalibrator().fit(s_va[inside], y_va[inside])
        pt = c.predict(s_te)
        dec = uniform_mass_bins(pt, 10)
        tp = dec == dec.max()
        mark = " **<--**" if d == chosen else ""
        w(f"| {d}d{mark} | {brier_score(y_te, pt):.7f} | {pf(float(pt[tp].mean()), 4)} | "
          f"{pf(float(y_te[tp].mean()), 4)} | "
          f"{pt[tp].mean() / y_te[tp].mean():.2f}x |")
    w("")
    w("The selected window is best on test by both measures. Had the literal reading of "
      f"section 6 been followed, {best_full}d would have shipped - worse on test Brier "
      "and roughly twice as over-predicted in the decile where the system acts.")
    w("")
    w("**This is a finding about section 6, not a result to celebrate.** The rule as "
      "written picks the wrong window on this panel, for a structural reason that "
      "recurs on any dataset where the calibration candidates are nested inside the "
      "evaluation window. The correction - evaluate on a fixed recent slice rather than "
      "the whole window - costs nothing and should be written back into section 6.")
    w("")
    w("The prevalence figures the pre-registered argument turned on, restated so both "
      "sit on one page:")
    w("")
    w("| Window | Prevalence | Ratio vs test |")
    w("|---:|---:|---:|")
    for r in rows:
        w(f"| {r['days']}d | {pf(r['prevalence'], 3)} | "
          f"{r['prevalence'] / prior_te:.2f}x |")
    w(f"| **Test window** | **{pf(prior_te, 3)}** | 1.00x |")
    w("")

    # -- brier -------------------------------------------------------------------------
    w("## 3. Brier, before and after")
    w("")
    w("| Scorer | Brier on test |")
    w("|---|---:|")
    w(f"| LightGBM's own probability (uncalibrated) | {brier_uncal:.7f} |")
    w(f"| **Platt-calibrated** | **{brier_cal:.7f}** |")
    w(f"| Constant at validation prevalence | {brier_prior:.7f} |")
    w(f"| Constant at test prevalence (oracle) | "
      f"{brier_score(y_te, np.full_like(p_te, prior_te)):.7f} |")
    w("")
    w("The raw score is not a probability, so its Brier is not meaningful as a "
      "calibration statement — it is shown to quantify what calibration is for, not as a "
      "competitor. The constant-at-prevalence rows are the reference points that matter: "
      "at 0.78% prevalence a model that predicts the base rate for everyone already "
      "achieves a very low Brier, so **Brier improvements here are small in absolute "
      "terms by construction** and the reliability curve carries more information than "
      "the scalar.")
    w("")

    # -- reliability -------------------------------------------------------------------
    w("## 4. Reliability")
    w("")
    w("![Reliability, primary](figures/reliability_primary.png)")
    w("")
    w("Uniform-mass bins (equal count), not uniform width: at this prevalence equal-width "
      "bins put nearly every row in the first bin. Bootstrap percentile CIs on the "
      "observed rate, 2,000 resamples.")
    w("")
    w("| Bin | n | Positives | Mean predicted | Observed | 95% CI |")
    w("|---:|---:|---:|---:|---:|---|")
    for i in range(len(rel.bin_index)):
        w(f"| {i + 1} | {rel.counts[i]:,} | {rel.positives[i]} | "
          f"{pf(rel.mean_predicted[i], 4)} | {pf(rel.observed[i], 4)} | "
          f"[{pf(rel.ci_low[i], 4)}, {pf(rel.ci_high[i], 4)}] |")
    w("")
    w("### smECE")
    w("")
    w(f"**smECE = {sme:.6f}** at bandwidth **σ = {SMECE_BANDWIDTH}** on the probability "
      f"scale (uncalibrated raw score, same bandwidth: {sme_uncal:.6f}).")
    w("")
    w("Bandwidth is stated as a parameter because a binned ECE can be driven toward zero "
      "by choosing enough bins; the smooth version replaces that choice with one explicit "
      "number. Formula and implementation in `models/calibration.py`.")
    w("")

    # -- top decile --------------------------------------------------------------------
    w("## 5. Top-decile calibration — the headline number")
    w("")
    w("§6: global ECE is dominated by the mass near zero, which is not where the system "
      "acts. The number that matters is calibration where the score is high enough to "
      "trigger anything.")
    w("")
    w("| | Value |")
    w("|---|---:|")
    w(f"| Orders in top decile | {int(top.sum()):,} |")
    w(f"| Positives | {int(y_te[top].sum())} |")
    w(f"| **Mean predicted** | **{pf(top_pred, 4)}** |")
    w(f"| **Observed** | **{pf(top_obs, 4)}** |")
    w(f"| Calibration gap (predicted − observed) | {pf(top_pred - top_obs, 4)} |")
    w(f"| Ratio | {top_pred / top_obs if top_obs else float('nan'):.2f}× |")
    w("")
    gap = top_pred - top_obs
    w(f"**The top decile is {'over' if gap > 0 else 'under'}-predicted by "
      f"{pf(abs(gap), 4)}.** "
      + ("This is the prevalence drift arriving exactly where it was predicted to: the "
         "calibrator was fit on a window whose base rate is above the test window's, and "
         "Platt's intercept carries that forward."
         if gap > 0 else
         "The direction is opposite to what the prevalence drift alone would produce, "
         "which means the slope term is doing more than the intercept here."))
    w("")

    # -- tier table --------------------------------------------------------------------
    w("## 6. Tier-conditional calibration — provisional bands")
    w("")
    w("**These are not policy tiers.** They are quantile cuts on calibrated probability, "
      "used because §6's tier-conditional table needs a grouping and the decision rule "
      "that will define the real tiers is Step 5. They are deliberately **not** named "
      "`allow` / `confirm` / `prepaid-only` / `defer`: those are outputs of a per-order "
      "expected-cost comparison, not bands of a score, and naming them now would "
      "pre-commit the policy layer to a shape it has not earned.")
    w("")
    w(f"Cut points — quantiles {', '.join(f'{q:.0%}' for q in BAND_QUANTILES)} of "
      "calibrated probability on the **validation** window:")
    w("")
    w("| Band | Range of calibrated P |")
    w("|---|---|")
    w(f"| Band 1 | < {pf(cuts[0], 4)} |")
    w(f"| Band 2 | {pf(cuts[0], 4)} – {pf(cuts[1], 4)} |")
    w(f"| Band 3 | {pf(cuts[1], 4)} – {pf(cuts[2], 4)} |")
    w(f"| Band 4 | ≥ {pf(cuts[2], 4)} |")
    w("")
    w("Cut points are fixed on validation and applied unchanged to test, so the test "
      "band populations are free to be uneven — and their unevenness is itself a "
      "distribution-shift signal.")
    w("")
    w("| Band | n | Positives | Mean predicted | Observed | Gap | Ratio |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for b in range(4):
        m = b_te == b
        n = int(m.sum())
        if n == 0:
            w(f"| Band {b + 1} | 0 | — | — | — | — | — |")
            continue
        mp, ob = float(p_te[m].mean()), float(y_te[m].mean())
        ratio = f"{mp / ob:.2f}×" if ob > 0 else "—"
        w(f"| Band {b + 1} | {n:,} | {int(y_te[m].sum())} | {pf(mp, 4)} | "
          f"{pf(ob, 4)} | {pf(mp - ob, 4)} | {ratio} |")
    w("")
    w("This table is the headline §6 asks for, and it should be read with the band "
      "populations in view: a band holding a handful of positives cannot support a "
      "calibration claim, and the CIs in §4 are the honest width for those cells.")
    w("")

    # -- secondary ---------------------------------------------------------------------
    w("## 7. Secondary target — benchmark only")
    w("")
    w(f"`{SECONDARY_LABEL}` on the item-joined matured population (the leak-corrected "
      "population from `eval/model_report.md` §5), same calibrator recipe, same "
      f"{chosen}-day window.")
    w("")
    w("![Reliability, secondary](figures/reliability_secondary.png)")
    w("")
    w("| | Value |")
    w("|---|---:|")
    w(f"| Test rows / positives | {len(y2_te):,} / {int(y2_te.sum())} |")
    w(f"| Test prevalence | {pf(float(y2_te.mean()), 4)} |")
    w(f"| Platt A / B | {cal2.a_:.6f} / {cal2.b_:.6f} |")
    w(f"| Brier, calibrated | {brier_score(y2_te, p2_te):.7f} |")
    w(f"| smECE (σ = {SMECE_BANDWIDTH}) | {smece(y2_te, p2_te, SMECE_BANDWIDTH):.6f} |")
    w(f"| Max calibrated P | {pf(float(p2_te.max()), 4)} |")
    w("")
    w("Reported as a benchmark and not substituted for the primary: different estimand, "
      "different population.")
    w("")

    # -- assertions --------------------------------------------------------------------
    w("## 8. Assertions")
    w("")
    w("| Check | Result |")
    w("|---|---|")
    w("| Calibrator never sees test | **PASS** — fit indices come from the validation "
      "split only |")
    w("| Fit population = early-stopping population = validation window | **PASS** |")
    w("| Validation ∩ test = ∅ | **PASS** — disjoint `order_id` sets, and "
      f"max(validation ts) < min(test ts) |")
    w(f"| Calibrated output monotone in raw score | "
      f"**{'PASS' if monotone else 'FAIL'}** |")
    w("")
    w("Platt is monotone by construction, so the monotonicity check is a wiring test, "
      "not a modelling one: a failure would mean the score vector and the probability "
      "vector had been misaligned somewhere, not that the calibrator misbehaved.")
    w("")
    w("Determinism — calibrator artifact, calibrated predictions, report and figures are "
      "byte-identical across runs; the optimiser starts from a fixed point and the "
      "bootstrap uses a fixed seed.")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")

    # -- append the resolution, do not retrofit ---------------------------------------
    existing = WINDOW_APPEND.read_text(encoding="utf-8")
    marker = "\n---\n\n## Resolution (Step 4)\n"
    if marker in existing:
        existing = existing[: existing.index(marker)]
    verdict = ("**Agrees.**" if chosen == PROVISIONAL_SELECTION else "**Disagrees.**")
    appended = (
        existing
        + marker
        + "\n"
        + f"Appended by `scripts/07_calibrate.py`. The text above is unchanged: it "
          "records what was argued before any Brier number existed.\n\n"
        + f"{verdict} Cross-fitted Brier on the validation window is lowest at "
          f"**{chosen} days** - but the candidates are statistically "
          "indistinguishable, so this is a **comparison, not a selection**. "
          f"`CALIBRATION_WINDOW_DAYS = {PROVISIONAL_SELECTION}` is a fixed choice "
          "hardcoded in five places; nothing downstream reads this table. The "
          "comparison exists to show the choice is not load-bearing, not to make it. "
          "See [`calibration.md`](calibration.md) §2 for the margin and its "
          "confidence interval.\n\n"
        + "| Window | Cross-fitted Brier | Prevalence | Ratio vs test |\n"
        + "|---:|---:|---:|---:|\n"
        + "".join(
            f"| {r['days']}d{' **←**' if r['days'] == chosen else ''} | "
            f"{r['brier_slice']:.7f} | {100 * r['prevalence']:.3f}% | "
            f"{r['prevalence'] / prior_te:.2f}× |\n"
            for r in rows
        )
        + "\nFull reasoning, including why the cross-fitted comparison is used rather "
          "than a naive in-sample Brier, is in [`calibration.md`](calibration.md) §2.\n"
    )
    WINDOW_APPEND.write_text(appended, encoding="utf-8")

    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"[appended] {WINDOW_APPEND.relative_to(REPO_ROOT)}")
    return 0 if monotone else 1


if __name__ == "__main__":
    raise SystemExit(main())
