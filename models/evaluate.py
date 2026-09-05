"""
Evaluation metrics and figures.  ARCHITECTURE.md 9 items 2, 3, 4.

Primary metric is average precision (``average_precision_score``), not trapezoidal
PR-AUC: the trapezoid interpolates between operating points that are not achievable, and
overstates a curve with few positives - which is exactly this setting.

ROC is computed and plotted because omitting it invites the question, and it is labelled
as not the headline everywhere it appears.  At 0.78% prevalence ROC-AUC is dominated by
the negatives and looks flattering regardless of whether the model is useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

__all__ = ["Metrics", "evaluate", "precision_recall_at_k", "OPERATING_BUDGET"]

#: The operating point, as an intervention budget rather than a probability threshold.
#: A threshold cannot be chosen honestly before the cost model exists - the per-order
#: Elkan threshold is step 5 - so the operating point here is "treat the top 5% of
#: scores", a stated budget, and it is labelled provisional wherever it is reported.
OPERATING_BUDGET = 0.05

K_BUDGETS = (0.01, 0.05)


@dataclass(frozen=True)
class Metrics:
    n: int
    n_positives: int
    prevalence: float
    average_precision: float
    roc_auc: float
    at_k: dict[float, dict[str, float]]
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def lift(self) -> float:
        """Average precision relative to the prevalence baseline."""
        return self.average_precision / self.prevalence if self.prevalence else float("nan")

    @property
    def ap_over_baseline(self) -> float:
        return self.average_precision - self.prevalence

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")


def precision_recall_at_k(
    y_true: np.ndarray, scores: np.ndarray, k: float
) -> dict[str, float]:
    """
    Precision and recall over the top ``k`` fraction by score.

    Ties are broken by taking exactly ``ceil(k * n)`` rows after a stable descending
    sort, so the budget is honoured exactly and the result is deterministic.
    """
    n = len(scores)
    take = max(1, int(np.ceil(k * n)))
    order = np.argsort(-scores, kind="stable")[:take]
    selected = y_true[order]
    n_pos = int(y_true.sum())
    tp = int(selected.sum())
    return {
        "k": k,
        "n_selected": take,
        "tp": tp,
        "precision": tp / take,
        "recall": tp / n_pos if n_pos else float("nan"),
        "lift": (tp / take) / (n_pos / n) if n_pos else float("nan"),
    }


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    n = len(y_true)
    n_pos = int(y_true.sum())
    prevalence = n_pos / n if n else float("nan")

    take = max(1, int(np.ceil(OPERATING_BUDGET * n)))
    threshold = float(np.sort(scores)[::-1][take - 1])
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    return Metrics(
        n=n,
        n_positives=n_pos,
        prevalence=prevalence,
        average_precision=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        at_k={k: precision_recall_at_k(y_true, scores, k) for k in K_BUDGETS},
        threshold=threshold,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def pr_points(y_true: np.ndarray, scores: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return recall, precision


def roc_points(y_true: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr
