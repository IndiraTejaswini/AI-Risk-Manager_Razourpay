"""
Calibration.  ARCHITECTURE.md 6, build step 4.

    LightGBM raw score
       → Platt scaling (2-parameter), fit on the temporal validation window
       → uniform-mass binning of the fitted function's outputs
       → calibrated P(failure) on a common scale

**What the second line means here.** Platt's output *is* the calibrated probability.  The
uniform-mass binning is the **evaluation scheme** - the bins the reliability curve, the
tier table and the ECE are computed over.  It is specified as uniform-mass rather than
uniform-width because at 0.78% prevalence equal-width bins put essentially every row in
the first bin and the curve says nothing.  Binning is not a second remap: a bin-average
remap stacked on Platt could break monotonicity, and monotonicity in the raw score is a
property this pipeline asserts.

**Not isotonic, not beta.**  Isotonic fits a free monotone step function and needs far
more positives than 154 before it stops memorising the calibration set; beta adds a third
parameter to the same problem.  Platt's two parameters are what this sample supports.

Platt's target smoothing is used, not hard 0/1 labels:

    t+ = (N+ + 1) / (N+ + 2),    t- = 1 / (N- + 2)

which is the original method's regularisation against a separable fit.  At 71 positives
in the 30-day window that is not a detail - with hard labels a well-separated score can
drive the fitted slope to infinity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "PlattCalibrator",
    "brier_score",
    "uniform_mass_bins",
    "reliability",
    "smece",
    "CalibrationReport",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, avoiding overflow for large |z|."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class PlattCalibrator:
    """
    Two-parameter Platt scaling:  P = 1 / (1 + exp(A·f + B)).

    Monotone in ``f`` by construction - strictly decreasing if A > 0, increasing if
    A < 0 - so a calibrated output that does not move monotonically with the raw score
    indicates a wiring error, not a modelling one.
    """

    def __init__(self) -> None:
        self.a_: float | None = None
        self.b_: float | None = None
        self.n_pos_: int = 0
        self.n_neg_: int = 0

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        f = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y).astype(int)

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Platt scaling needs both classes present")

        # Platt's smoothed targets.
        hi = (n_pos + 1.0) / (n_pos + 2.0)
        lo = 1.0 / (n_neg + 2.0)
        t = np.where(y == 1, hi, lo)

        def objective(params):
            """
            NLL of the smoothed targets under P = sigmoid(-z), z = A·f + B.

                -Σ [ t·log P + (1-t)·log(1-P) ]
              =  Σ [ log(1+e^z) + (t-1)·z ]

            ``logaddexp(0, z)`` is log(1+e^z) evaluated without overflow.
            """
            a, b = params
            z = a * f + b
            return float(np.sum(np.logaddexp(0.0, z) + (t - 1.0) * z))

        def gradient(params):
            # d/dz [ log(1+e^z) + (t-1)z ] = sigmoid(z) + t - 1
            a, b = params
            z = a * f + b
            d = _sigmoid(z) + t - 1.0
            return np.array([float(np.sum(d * f)), float(np.sum(d))])

        # Deterministic start, deterministic optimiser.
        start = np.array([0.0, np.log((n_neg + 1.0) / (n_pos + 1.0))])
        res = minimize(objective, start, jac=gradient, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-12})

        self.a_, self.b_ = float(res.x[0]), float(res.x[1])
        self.n_pos_, self.n_neg_ = n_pos, n_neg
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.a_ is None:
            raise RuntimeError("calibrator is not fitted")
        z = self.a_ * np.asarray(scores, dtype=np.float64) + self.b_
        return _sigmoid(-z)

    @property
    def params(self) -> dict:
        return {"A": self.a_, "B": self.b_, "n_pos": self.n_pos_, "n_neg": self.n_neg_}


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def uniform_mass_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Bin assignment by equal mass, not equal width.

    At this prevalence equal-width bins are useless - almost every row lands in the
    lowest bin and the reliability curve degenerates to one point.  Ties in the score
    can make bins uneven; that is reported rather than forced.
    """
    p = np.asarray(p, dtype=float)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)


@dataclass(frozen=True)
class CalibrationReport:
    bin_index: np.ndarray
    mean_predicted: np.ndarray
    observed: np.ndarray
    counts: np.ndarray
    positives: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def reliability(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
    n_boot: int = 2000,
    seed: int = 20260901,
) -> CalibrationReport:
    """Uniform-mass reliability curve with percentile bootstrap CIs on the observed rate."""
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    b = uniform_mass_bins(p, n_bins)
    ids = np.unique(b)

    mean_pred, obs, counts, pos = [], [], [], []
    lo, hi = [], []
    rng = np.random.default_rng(seed)

    for i in ids:
        m = b == i
        yy, pp = y[m], p[m]
        n = len(yy)
        mean_pred.append(float(pp.mean()))
        obs.append(float(yy.mean()))
        counts.append(n)
        pos.append(int(yy.sum()))
        draws = rng.integers(0, n, size=(n_boot, n))
        rates = yy[draws].mean(axis=1)
        lo.append(float(np.percentile(rates, 2.5)))
        hi.append(float(np.percentile(rates, 97.5)))

    return CalibrationReport(
        bin_index=ids,
        mean_predicted=np.array(mean_pred),
        observed=np.array(obs),
        counts=np.array(counts),
        positives=np.array(pos),
        ci_low=np.array(lo),
        ci_high=np.array(hi),
    )


def smece(y: np.ndarray, p: np.ndarray, bandwidth: float) -> float:
    """
    Smooth expected calibration error at a **stated** bandwidth.

    Binned ECE depends on an arbitrary bin count and can be driven to zero by choosing
    enough bins.  The smooth version replaces binning with a Gaussian kernel regression
    of the residual (y − p) on p, and averages the absolute smoothed residual over the
    predictive distribution:

        smECE(σ) = (1/n) · Σ_i | Σ_j K_σ(p_i − p_j)(y_j − p_j) / Σ_j K_σ(p_i − p_j) |

    The bandwidth is a parameter and is reported with the number, never left implicit.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    r = y - p

    # Evaluate on a grid and interpolate: O(n·g) rather than O(n²).
    grid = np.linspace(p.min(), p.max(), 512)
    d = (grid[:, None] - p[None, :]) / bandwidth
    k = np.exp(-0.5 * d * d)
    denom = k.sum(axis=1)
    num = k @ r
    smoothed = np.where(denom > 0, num / np.maximum(denom, 1e-300), 0.0)
    at_points = np.interp(p, grid, smoothed)
    return float(np.mean(np.abs(at_points)))
