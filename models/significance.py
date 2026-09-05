"""
Paired significance testing.  ARCHITECTURE.md 9, "paired bootstrap over identical test
rows for every model comparison".

Why paired, and why this supersedes the MDD comparison
------------------------------------------------------

The minimum detectable difference in eval/positive_counts.md is a **planning** figure:
it answers "before seeing data, how big a gap would I need?" under an assumed AP and an
assumed correlation between two models.  Comparing two realised point estimates against
it is not a test - it discards the pairing, assumes rho rather than measuring it, and is
strictly less powerful than resampling the difference directly.

Two models scored on the same test rows share every source of variance that comes from
*which rows landed in the test window*.  Resampling the rows and recomputing the
**difference** cancels that shared variance.  What remains is the variance of the
difference itself, which is the quantity the question is actually about.

Implementation note: the bootstrap resamples rows as multinomial counts over the
original index, so both models see the identical resample.  Each model's test rows are
sorted by score once, tie groups are located once, and each replicate is a weighted
cumulative sum over those fixed groups - so 10,000 replicates cost two sorts, not
20,000.  The tie handling matches ``sklearn.average_precision_score`` exactly and is
asserted against it in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "APScorer",
    "BootstrapResult",
    "DeLongResult",
    "paired_bootstrap_ap",
    "delong_auc_test",
    "permutation_ap_null",
    "PermutationResult",
]

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260901


class APScorer:
    """
    Precomputed structure for evaluating average precision under row weights.

    Sorting and tie-grouping depend only on the scores, not on the resample, so they are
    done once.  ``ap(weights)`` is then a handful of vector operations.
    """

    def __init__(self, y_true: np.ndarray, scores: np.ndarray) -> None:
        y = np.asarray(y_true).astype(np.float64)
        s = np.asarray(scores, dtype=np.float64)

        order = np.argsort(-s, kind="stable")
        self._order = order
        self._y = y[order]
        s_sorted = s[order]

        # Tie groups: average precision steps once per distinct score, not once per row.
        boundaries = np.flatnonzero(np.diff(s_sorted)) + 1
        self._starts = np.concatenate(([0], boundaries))

    def ap(self, weights: np.ndarray | None = None) -> float:
        w = (
            np.ones_like(self._y)
            if weights is None
            else np.asarray(weights, dtype=np.float64)[self._order]
        )
        pos_g = np.add.reduceat(w * self._y, self._starts)
        all_g = np.add.reduceat(w, self._starts)

        cum_pos = np.cumsum(pos_g)
        cum_all = np.cumsum(all_g)
        total_pos = cum_pos[-1]
        if total_pos <= 0:
            return float("nan")

        precision = cum_pos / np.maximum(cum_all, 1e-12)
        recall = cum_pos / total_pos
        d_recall = np.diff(np.concatenate(([0.0], recall)))
        return float(np.sum(d_recall * precision))

    def prevalence(self, weights: np.ndarray | None = None) -> float:
        w = (
            np.ones_like(self._y)
            if weights is None
            else np.asarray(weights, dtype=np.float64)[self._order]
        )
        total = w.sum()
        return float((w * self._y).sum() / total) if total else float("nan")


@dataclass(frozen=True)
class BootstrapResult:
    name: str
    observed: float
    mean: float
    ci_low: float
    ci_high: float
    frac_le_zero: float
    n_resamples: int
    percentiles: dict[int, float]
    differences: np.ndarray

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0

    @property
    def verdict(self) -> str:
        if self.excludes_zero:
            return "separates"
        return "does not separate"


def paired_bootstrap_ap(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray | None,
    *,
    name: str,
    against_prevalence: bool = False,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """
    Bootstrap the difference ``AP(a) - AP(b)`` over identical resampled test rows.

    ``against_prevalence=True`` compares model ``a`` against the prevalence baseline -
    the horizontal PR line whose average precision *is* the prevalence of the sample.
    Recomputing that prevalence per replicate is what makes the comparison paired: both
    quantities move together as the resample's positive count moves.
    """
    y = np.asarray(y_true).astype(np.float64)
    n = len(y)

    scorer_a = APScorer(y, scores_a)
    scorer_b = None if against_prevalence else APScorer(y, scores_b)

    observed = scorer_a.ap() - (
        scorer_a.prevalence() if against_prevalence else scorer_b.ap()
    )

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    p = np.full(n, 1.0 / n)

    for i in range(n_resamples):
        w = rng.multinomial(n, p).astype(np.float64)
        a = scorer_a.ap(w)
        b = scorer_a.prevalence(w) if against_prevalence else scorer_b.ap(w)
        diffs[i] = a - b

    finite = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return BootstrapResult(
        name=name,
        observed=float(observed),
        mean=float(finite.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        frac_le_zero=float((finite <= 0).mean()),
        n_resamples=len(finite),
        percentiles={
            q: float(np.percentile(finite, q)) for q in (1, 5, 25, 50, 75, 95, 99)
        },
        differences=finite,
    )


# --------------------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = t
    return out


@dataclass(frozen=True)
class DeLongResult:
    auc: float
    variance: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float

    @property
    def excludes_half(self) -> bool:
        return self.ci_low > 0.5


def delong_auc_test(
    y_true: np.ndarray, scores: np.ndarray, null_auc: float = 0.5
) -> DeLongResult:
    """
    DeLong's variance estimate for a single ROC-AUC, and a test against ``null_auc``.

    DeLong's estimator uses the midrank structural components, so it is exact for tied
    scores - which matters here, because a gradient-boosted model assigns identical
    scores to every row landing in the same combination of leaves.

    Ranking quality and probability quality come apart (ARCHITECTURE.md 9), so this can
    resolve where the average-precision comparison does not.  Both outcomes are reported.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)

    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present")

    tx, ty = _midrank(pos), _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))

    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n

    se = float(np.sqrt(var))
    z = (auc - null_auc) / se if se > 0 else float("inf")
    return DeLongResult(
        auc=float(auc),
        variance=float(var),
        ci_low=float(auc - 1.959963984540054 * se),
        ci_high=float(auc + 1.959963984540054 * se),
        z=float(z),
        p_value=float(2 * stats.norm.sf(abs(z))),
    )


# --------------------------------------------------------------------------------------
# Permutation null - the honest "better than chance" comparison
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PermutationResult:
    observed: float
    prevalence: float
    null_mean: float
    null_p95: float
    null_p99: float
    p_value: float
    n_permutations: int
    upward_bias: float
    null: np.ndarray

    @property
    def beats_chance(self) -> bool:
        return self.p_value < 0.05


def permutation_ap_null(
    y_true: np.ndarray,
    observed_ap: float,
    *,
    n_permutations: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> PermutationResult:
    """
    Null distribution of average precision under a **random ranking** of these labels.

    Why this and not the prevalence baseline
    ----------------------------------------

    The prevalence line is the textbook PR baseline, and its average precision is exactly
    the prevalence.  But a *randomly ranked* model does not score the prevalence - it
    scores slightly above it, because average precision averages precision-at-rank over
    the positives and precision-at-rank is upward-biased at the small ranks where a
    positive can land early by luck.  The bias shrinks as the positive count grows.

    So "beats the prevalence baseline" is a slightly easier bar than "beats chance", and
    at a small positive count the difference can matter.  This measures the gap
    (:attr:`upward_bias`) and tests against the harder, correct null.

    Permuting the labels is exactly a random ranking, and with distinct scores there are
    no ties, so average precision has the closed form used here - no sort per draw.
    """
    y = np.asarray(y_true).astype(np.float64)
    n = len(y)
    n_pos = float(y.sum())
    if n_pos == 0:
        raise ValueError("permutation null needs at least one positive")

    k = np.arange(1, n + 1, dtype=np.float64)
    rng = np.random.default_rng(seed)

    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        yp = rng.permutation(y)
        null[i] = float((yp * np.cumsum(yp) / k).sum() / n_pos)

    prevalence = float(y.mean())
    # +1 in numerator and denominator: the observed value is itself one draw from the
    # null under H0, so a p-value of exactly zero is not attainable.
    p = float((np.count_nonzero(null >= observed_ap) + 1) / (n_permutations + 1))
    return PermutationResult(
        observed=float(observed_ap),
        prevalence=prevalence,
        null_mean=float(null.mean()),
        null_p95=float(np.percentile(null, 95)),
        null_p99=float(np.percentile(null, 99)),
        p_value=p,
        n_permutations=n_permutations,
        upward_bias=float(null.mean() - prevalence),
        null=null,
    )
