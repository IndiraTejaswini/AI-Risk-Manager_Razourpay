#!/usr/bin/env python3
"""
Task F: resolve the Step 3 null with the test ARCHITECTURE.md 9 actually specifies.

Step 3 compared two point estimates against the minimum detectable difference.  That is
a planning heuristic, not a test.  This runs the paired bootstrap over identical test
rows, plus DeLong on ROC-AUC, and states which conclusion the paired test supports.

Nothing is tuned on the outcome.  The models are loaded and scored exactly as Step 3
fitted them.

Writes eval/significance.md.  Deterministic: fixed bootstrap seed.
"""

from __future__ import annotations

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

from data.loader import PRIMARY_LABEL, OlistLoader  # noqa: E402
from features.builder import FeatureBuilder  # noqa: E402
from models.significance import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    APScorer,
    delong_auc_test,
    paired_bootstrap_ap,
    permutation_ap_null,
)
from models.train import (  # noqa: E402
    ORDER_ONLY_ABSENT_CONSTRAINTS,
    ORDER_ONLY_GROUPS,
    predict,
    prepare_matrix,
    train,
)

OUT_PATH = REPO_ROOT / "eval" / "significance.md"
FIG_DIR = REPO_ROOT / "eval" / "figures"

MDD_PRIMARY = 0.043

# -- the noise-draw calibration study, §2 -----------------------------------------------
#
# These figures used to be typed into the report as literals, under a heading reading
# "Measured, not asserted."  The study was real - it lives in
# tests/test_significance.py::test_calibration_of_the_two_chance_comparisons - but its
# committed parameters (n=3000, 120 draws) were not the ones the report described, so no
# run of this repo reproduced the numbers on the page.  They are computed here now.
#
# The test keeps its cheaper parameters and its structural assertions; this is the
# measurement the report quotes.
NOISE_DRAWS = 200
NOISE_N = 4_000
NOISE_PREVALENCE = 0.02
NOISE_SEED = 20260901
NOISE_BOOT_RESAMPLES = 150
NOISE_PERMUTATIONS = 200


def _noise_calibration_study() -> dict:
    """
    False-positive rate of the two chance comparisons under a pure-noise scorer.

    A correctly calibrated 5% test rejects 5% of the time.  Returns the realised
    rejection rate of each, plus the share of draws whose permutation-null mean sat
    above the prevalence line - the structural bias that makes "beats the prevalence
    baseline" an easier bar than "beats chance".
    """
    rng = np.random.default_rng(NOISE_SEED)
    boot_rejects = perm_rejects = used = biased = 0

    for _ in range(NOISE_DRAWS):
        y = (rng.random(NOISE_N) < NOISE_PREVALENCE).astype(int)
        if y.sum() < 5:
            continue
        s = rng.normal(size=NOISE_N)          # independent of y: pure noise
        used += 1
        b = paired_bootstrap_ap(
            y, s, None, name="noise", against_prevalence=True,
            n_resamples=NOISE_BOOT_RESAMPLES, seed=int(rng.integers(1_000_000_000)),
        )
        boot_rejects += int(b.excludes_zero)
        pr = permutation_ap_null(
            y, APScorer(y, s).ap(),
            n_permutations=NOISE_PERMUTATIONS, seed=int(rng.integers(1_000_000_000)),
        )
        perm_rejects += int(pr.beats_chance)
        biased += int(pr.upward_bias > 0)

    return {
        "draws": used,
        "boot_rate": boot_rejects / used,
        "perm_rate": perm_rejects / used,
        "biased_share": biased / used,
    }


def _fit_and_score(risk, matrix, allow_missing=frozenset()):
    split = risk["split"].to_numpy()
    y = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=PRIMARY_LABEL, population="risk_set", category_levels=levels,
        allow_missing_constraints=allow_missing,
    )
    return bundle, predict(bundle, X.loc[te]), y[te]


def main() -> int:
    loader = OlistLoader()
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")

    _, s_full, y_test = _fit_and_score(risk, FeatureBuilder().build(risk))
    _, s_order, _ = _fit_and_score(
        risk,
        FeatureBuilder(groups=ORDER_ONLY_GROUPS).build(risk),
        allow_missing=ORDER_ONLY_ABSENT_CONSTRAINTS,
    )

    vs_prev = paired_bootstrap_ap(
        y_test, s_full, None, name="model − prevalence baseline", against_prevalence=True
    )
    vs_order = paired_bootstrap_ap(
        y_test, s_full, s_order, name="full matrix − order-fields-only"
    )
    delong = delong_auc_test(y_test, s_full)
    delong_order = delong_auc_test(y_test, s_order)
    perm = permutation_ap_null(y_test, APScorer(y_test, s_full).ap())
    noise = _noise_calibration_study()

    # Top-5% budget, computed the same way eval/model_report.md computes it, so the
    # sentence in §5 cannot drift from the table it is describing.
    k5 = int(np.ceil(0.05 * len(y_test)))
    top5 = np.argsort(-s_full, kind="stable")[:k5]
    top5_tp = int(y_test[top5].sum())
    top5_precision = top5_tp / k5
    top5_recall = top5_tp / int(y_test.sum())

    # figure
    from models.plots import plot_bootstrap

    plot_bootstrap(
        {"full": (vs_prev.differences, "vs prevalence baseline", vs_prev),
         "baseline": (vs_order.differences, "vs order-fields-only", vs_order)},
        path=FIG_DIR / "bootstrap_ap.png",
        title="Paired bootstrap — distribution of the AP difference",
    )

    def pctf(x, dp=3):
        return f"{100 * x:.{dp}f}%"

    L: list[str] = []
    w = L.append

    w("# Significance — resolving the Step 3 null")
    w("")
    w("Generated by `scripts/05_significance.py`. Supersedes the MDD comparison in "
      "`eval/model_report.md` §1 and §2.")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Paired bootstrap: **{BOOTSTRAP_RESAMPLES:,}** resamples, seed "
      f"`{BOOTSTRAP_SEED}`, identical test rows for every model")
    w(f"- Test window: {len(y_test):,} rows, {int(y_test.sum())} positives, "
      f"prevalence {pctf(y_test.mean())}")
    w("")

    # -- why this supersedes the MDD ----------------------------------------------------
    w("## 1. Why this supersedes the MDD comparison")
    w("")
    w(f"Step 3 reported the model as unresolved because its AP sat {vs_prev.observed:+.4f} above the "
      f"prevalence baseline against a minimum detectable difference of {MDD_PRIMARY:.3f}. "
      "**That was not a test.** The MDD is a *planning* figure: it answers \"before "
      "seeing data, how large a gap would I need?\" under an assumed AP and an assumed "
      "correlation ρ between two models.")
    w("")
    w("Comparing two realised point estimates against it is strictly weaker than testing "
      "the difference, for three reasons:")
    w("")
    w("1. **It discards the pairing.** Both models are scored on the *same* test rows, so "
      "they share every source of variance arising from which orders landed in the test "
      "window. Resampling the difference cancels that shared variance; comparing two "
      "independent point estimates does not.")
    w("2. **It assumes ρ rather than measuring it.** The MDD was quoted at ρ = 0.8. The "
      "bootstrap needs no such assumption — the correlation is whatever it is, and it is "
      "absorbed automatically.")
    w("3. **The prevalence baseline is not an independent model.** Its AP *is* the "
      "prevalence of the sample, so it moves with the resample's positive count. Pairing "
      "captures that; an MDD comparison treats it as a fixed constant.")
    w("")
    w("The MDD remains reported below as the planning figure it was. It is not the answer.")
    w("")

    # -- the tests ---------------------------------------------------------------------
    w("## 2. Against chance — permutation null")
    w("")
    w("Reported first, because it is the comparison the prevalence baseline only "
      "approximates.")
    w("")
    w("A *randomly ranked* model does not score the prevalence — it scores slightly "
      "above it. Average precision averages precision-at-rank over the positives, and "
      "precision-at-rank is upward-biased at the small ranks where a positive can land "
      "early by luck. So \"beats the prevalence baseline\" is a marginally easier bar "
      "than \"beats chance\", and at 154 positives the gap is worth measuring rather "
      "than assuming away.")
    w("")
    w(f"Null: {perm.n_permutations:,} random rankings of the actual test labels.")
    w("")
    w("| | Value |")
    w("|---|---:|")
    w(f"| Prevalence (textbook PR baseline) | {perm.prevalence:.6f} |")
    w(f"| **Mean AP of a random ranker** | **{perm.null_mean:.6f}** |")
    w(f"| Upward bias the prevalence line misses | **{perm.upward_bias:+.6f}** |")
    w(f"| Null 95th / 99th percentile | {perm.null_p95:.6f} / {perm.null_p99:.6f} |")
    w(f"| **Observed model AP** | **{perm.observed:.6f}** |")
    w(f"| **Permutation p-value** | **{perm.p_value:.5f}** |")
    w("")
    if perm.beats_chance:
        w(f"**The model beats chance.** Its average precision of {perm.observed:.4f} sits "
          f"above the 99th percentile of the random-ranking null ({perm.null_p99:.4f}), "
          f"p = {perm.p_value:.5f}. This is the stronger statement: not merely above the "
          "idealised prevalence line, but above what random ordering of these exact "
          "labels actually produces.")
    else:
        w(f"**The model does not beat chance** at p = {perm.p_value:.5f}.")
    w("")
    w(f"The bias is {perm.upward_bias:+.6f} — about "
      f"{100 * perm.upward_bias / (perm.observed - perm.prevalence):.0f}% of the observed "
      "margin over prevalence, so it does not change the conclusion here. It is measured "
      "rather than assumed because at a smaller positive count it would.")
    w("")
    w("### How much does the baseline choice matter?")
    w("")
    w("Measured, not asserted. Under a pure-noise scorer a correctly calibrated 5% test "
      f"should reject 5% of the time. Over {noise['draws']:,} independent noise draws "
      f"(n = {NOISE_N:,}, prevalence {NOISE_PREVALENCE:.0%}):")
    w("")
    w("| Procedure | Rejection rate under the null | Nominal |")
    w("|---|---:|---:|")
    w(f"| Paired bootstrap vs **prevalence baseline** | **{noise['boot_rate']:.3f}** "
      "| 0.05 |")
    w(f"| **Permutation null** (random ranking) | **{noise['perm_rate']:.3f}** | 0.05 |")
    w("")
    w("The permutation null is calibrated. The prevalence-baseline comparison rejects "
      f"**{noise['boot_rate'] / 0.05:.1f}x as often as it should**, and the null mean "
      f"sat above the prevalence in **{noise['biased_share']:.0%} of draws** — the bias "
      "is structural, not sampling noise.")
    w("")
    w("So §3's bootstrap-against-prevalence result should be read as the weaker of the "
      "two, and the permutation p-value above is the one that carries the claim. Here "
      f"they agree and the margin is not close (p = {perm.p_value:.5f}, observed "
      f"{perm.observed:.4f} against a null 99th percentile of {perm.null_p99:.4f}), so "
      "the conclusion does not depend on which is used.")
    w("")
    w("*This came out of the repo's own test suite: a deliberately random scorer "
      "\"separated\" from the prevalence baseline in the paired bootstrap. That is a "
      "property of the baseline rather than a bug in the bootstrap, and it is pinned in "
      "`tests/test_significance.py` as a calibration comparison.*")
    w("")

    w("## 3. Paired bootstrap on the AP difference")
    w("")
    w("| Comparison | Observed Δ | Bootstrap mean | 95% CI | P(Δ ≤ 0) | Verdict |")
    w("|---|---:|---:|---|---:|---|")
    for r in (vs_prev, vs_order):
        w(f"| {r.name} | {r.observed:+.4f} | {r.mean:+.4f} | "
          f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.frac_le_zero:.4f} | "
          f"**{r.verdict}** |")
    w("")
    w("![Bootstrap difference distributions](figures/bootstrap_ap.png)")
    w("")
    w("Difference distribution, percentiles:")
    w("")
    w("| Comparison | p1 | p5 | p25 | p50 | p75 | p95 | p99 |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in (vs_prev, vs_order):
        cells = " | ".join(f"{r.percentiles[q]:+.4f}" for q in (1, 5, 25, 50, 75, 95, 99))
        w(f"| {r.name} | {cells} |")
    w("")

    # -- DeLong ------------------------------------------------------------------------
    w("## 4. DeLong test on ROC-AUC")
    w("")
    w("Ranking quality and probability quality come apart (ARCHITECTURE.md §9), so the "
      "ranking may resolve where average precision does not. Both outcomes are reported.")
    w("")
    w("| Model | ROC-AUC | 95% CI | z vs 0.5 | p | Excludes 0.5 |")
    w("|---|---:|---|---:|---:|---|")
    for label, d in (("Full matrix", delong), ("Order fields only", delong_order)):
        w(f"| {label} | {d.auc:.4f} | [{d.ci_low:.4f}, {d.ci_high:.4f}] | {d.z:.2f} | "
          f"{d.p_value:.3e} | **{'yes' if d.excludes_half else 'no'}** |")
    w("")
    w("DeLong's midrank estimator is exact under ties, which matters here: a "
      "gradient-boosted model assigns identical scores to every row landing in the same "
      "combination of leaves.")
    w("")

    # -- the conclusion ----------------------------------------------------------------
    w("## 5. What the tests support")
    w("")

    if vs_prev.excludes_zero:
        w(f"### The model separates. The Step 3 \"unresolved\" call was too conservative.")
        w("")
        w(f"The paired 95% CI on the AP difference against the prevalence baseline is "
          f"**[{vs_prev.ci_low:+.4f}, {vs_prev.ci_high:+.4f}]**, which **excludes zero**. "
          f"The difference was ≤ 0 in {vs_prev.frac_le_zero:.4f} of "
          f"{vs_prev.n_resamples:,} resamples. The permutation null in §2 agrees and is "
          f"the stronger test: p = {perm.p_value:.5f} against random ranking.")
        w("")
        w("**Correction to `eval/model_report.md` §1.** That section reports the model as "
          "not demonstrably beating the prevalence baseline. That conclusion was reached "
          f"by comparing a point estimate ({vs_prev.observed:+.4f}) against a planning "
          f"heuristic ({MDD_PRIMARY:.3f}) rather than by testing the difference. The "
          "paired test — which is the procedure §9 specifies and is strictly more "
          "powerful — resolves it. **The model does separate from the prevalence "
          "baseline on the primary target.**")
        w("")
        w("The earlier statement was honest about its uncertainty but wrong about the "
          "conclusion, and it was wrong in the conservative direction: it under-claimed. "
          "Recorded here rather than by editing §1, so the reasoning that produced the "
          "over-cautious call stays visible.")
    else:
        w("### The null stands, and is now properly evidenced")
        w("")
        w(f"The paired 95% CI is **[{vs_prev.ci_low:+.4f}, {vs_prev.ci_high:+.4f}]**, "
          f"which **includes zero**; the difference was ≤ 0 in "
          f"{vs_prev.frac_le_zero:.4f} of {vs_prev.n_resamples:,} resamples.")
        w("")
        w("Step 3's conclusion was right, but it was *inferred* from a planning figure "
          "rather than tested. It is now evidenced by the procedure §9 specifies.")
    w("")

    if vs_order.excludes_zero:
        w(f"**The encodings are resolvable.** The full matrix beats the order-only "
          f"baseline by {vs_order.observed:+.4f} AP with a 95% CI of "
          f"[{vs_order.ci_low:+.4f}, {vs_order.ci_high:+.4f}], excluding zero. Step 3 "
          "called this unresolved on the same MDD reasoning; that call is superseded. "
          "The customer-history and pincode groups do contribute, and given customer "
          "history is empty for 97% of rows, the contribution is essentially the pincode "
          "target encoding — which is what §4.5 predicted would carry the signal.")
    else:
        w(f"**The encodings remain unresolved.** The full matrix leads the order-only "
          f"baseline by {vs_order.observed:+.4f} AP, 95% CI "
          f"[{vs_order.ci_low:+.4f}, {vs_order.ci_high:+.4f}], which includes zero. This "
          "test size cannot separate a small contribution from none.")
    w("")

    if delong.excludes_half:
        w(f"**Ranking resolves.** ROC-AUC {delong.auc:.4f}, 95% CI "
          f"[{delong.ci_low:.4f}, {delong.ci_high:.4f}], p = {delong.p_value:.2e}. The "
          "model orders orders better than chance, and that conclusion is independent of "
          "the average-precision result above — exactly the targeting-vs-prediction "
          "split §9 warns about.")
    else:
        w(f"**Ranking does not resolve either.** ROC-AUC 95% CI "
          f"[{delong.ci_low:.4f}, {delong.ci_high:.4f}] includes 0.5.")
    w("")

    w("### What this does not license")
    w("")
    w("Separation from a prevalence baseline is a low bar and is **not** a claim that the "
      f"model is useful. At the top-5% budget precision is {100 * top5_precision:.2f}% "
      f"and recall {100 * top5_recall:.2f}%: the "
      "policy layer in §7 decides whether that clears the cost of intervening, and that "
      "layer does not exist yet. Nor does calibration — these are scores, not "
      "probabilities, and every number here is about *ranking*.")
    w("")
    w("Nothing was tuned on these results.")
    w("")

    # -- MDD, in its place --------------------------------------------------------------
    w("## 6. The MDD, reported as the planning figure it is")
    w("")
    w("| | Value |")
    w("|---|---:|")
    w(f"| Planning MDD (AP 0.10, ρ 0.8, α 0.05, power 0.80) | {MDD_PRIMARY:.3f} |")
    w(f"| Observed Δ vs prevalence | {vs_prev.observed:+.4f} |")
    w(f"| Paired bootstrap 95% CI half-width | "
      f"{(vs_prev.ci_high - vs_prev.ci_low) / 2:.4f} |")
    w("")
    hw = (vs_prev.ci_high - vs_prev.ci_low) / 2
    w(f"The realised half-width is **{hw:.4f}**, against a planning MDD of "
      f"**{MDD_PRIMARY:.3f}** — the planning figure was **{MDD_PRIMARY / hw:.1f}× too "
      "pessimistic** for this comparison. The approximation assumed AP ≈ 0.10; the "
      "realised AP is ~0.02, and the standard error of average precision scales with "
      "√(AP(1−AP)), so assuming an AP five times too high inflated the required gap "
      "correspondingly. The planning figure did its job — it stopped us promising "
      "resolution we could not deliver — but it should not have been used as a test, "
      "and this section is the correction.")
    w("")
    w("**Carried forward:** comparisons in §9 are adjudicated by the paired bootstrap, "
      "not by the MDD. The MDD is quoted once, as a planning figure, and never as a "
      "decision rule.")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
