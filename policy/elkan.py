"""
Per-order Elkan thresholds and tier selection.  ARCHITECTURE.md 7.4, 7.5.

    p*(o, t) = c_FP(o, t) / (c_FP(o, t) + c_FN(o, t))

computed per order and per tier rather than once globally, with c_FP decomposed into
impression + triggered (7.2).

**No labels enter this module.**  Thresholds are a function of costs and effectiveness
only.  That is not a convention, it is what makes the assertion "the policy never sees
test labels when selecting thresholds" structurally true rather than a matter of
discipline: there is no argument to pass a label to.

The ladder is allow -> confirm -> prepaid_only -> defer, and nobody is blocked: the
worst outcome for a false positive is being asked to prepay (7.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from policy.costs import ACTION_TIERS, TIERS, c_rto, tier_costs
from policy.effectiveness import effectiveness_vector

__all__ = ["PolicyResult", "apply_policy", "expected_cost_at_threshold"]


@dataclass(frozen=True)
class PolicyResult:
    tier: np.ndarray                     # chosen tier per order
    thresholds: dict[str, np.ndarray]    # p*(o, t) per action tier
    c_fp: dict[str, np.ndarray]
    c_fn: dict[str, np.ndarray]
    expected_cost: dict[str, np.ndarray] # per order, per tier including "allow"
    rto_cost: np.ndarray
    reasons: np.ndarray
    fires: dict[str, np.ndarray]         # p >= p* per action tier

    @property
    def any_fires(self) -> bool:
        return bool(any(f.any() for f in self.fires.values()))


def apply_policy(
    p: np.ndarray,
    features: pd.DataFrame,
    reasons: np.ndarray,
    effectiveness_scale: float = 1.0,
    c_rto_scale: float = 1.0,
) -> PolicyResult:
    """
    Select a tier per order by minimum expected cost, and report the Elkan threshold
    each tier would need.

    Parameters
    ----------
    p:
        Calibrated P(fails | ships).  **The only place risk enters.**
    effectiveness_scale, c_rto_scale:
        Sensitivity axes.  Applied multiplicatively to the effectiveness matrix and to
        c_rto respectively.
    """
    p = np.asarray(p, dtype=float)
    value = np.nan_to_num(features["order_value"].to_numpy(dtype=float), nan=0.0)
    freight = np.nan_to_num(features["order_freight"].to_numpy(dtype=float), nan=0.0)

    rto = c_rto(value, freight) * c_rto_scale
    costs = tier_costs(features)

    thresholds: dict[str, np.ndarray] = {}
    c_fp: dict[str, np.ndarray] = {}
    c_fn: dict[str, np.ndarray] = {}
    exp: dict[str, np.ndarray] = {"allow": p * rto}
    fires: dict[str, np.ndarray] = {}

    for tier in ACTION_TIERS:
        e = effectiveness_vector(reasons, tier, effectiveness_scale)
        impression = costs[tier]["impression"]
        fp = costs[tier]["c_fp"]
        fn = e * rto - impression

        # A tier whose c_FN is non-positive can never be worth firing: acting costs more
        # than the failure it prevents, whatever the probability.  Threshold set to
        # infinity so it is excluded rather than producing a nonsensical ratio.
        denom = fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            star = np.where(fn > 0, fp / np.where(denom != 0, denom, np.nan), np.inf)
        star = np.nan_to_num(star, nan=np.inf, posinf=np.inf)

        thresholds[tier] = star
        c_fp[tier] = fp
        c_fn[tier] = fn
        exp[tier] = p * (impression + (1.0 - e) * rto) + (1.0 - p) * fp
        fires[tier] = p >= star

    stacked = np.vstack([exp[t] for t in TIERS])
    tier = np.array(TIERS)[np.argmin(stacked, axis=0)]

    return PolicyResult(
        tier=tier, thresholds=thresholds, c_fp=c_fp, c_fn=c_fn,
        expected_cost=exp, rto_cost=rto, reasons=reasons, fires=fires,
    )


def expected_cost_at_threshold(
    p: np.ndarray,
    y: np.ndarray,
    features: pd.DataFrame,
    reasons: np.ndarray,
    threshold: float,
    tier: str,
    effectiveness_scale: float = 1.0,
    c_rto_scale: float = 1.0,
) -> float:
    """
    **Realised** cost of treating every order with ``p >= threshold`` at ``tier``.

    Uses observed labels, so this is an evaluation function, not a decision function -
    it is never called while choosing a threshold.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(bool)

    value = np.nan_to_num(features["order_value"].to_numpy(dtype=float), nan=0.0)
    freight = np.nan_to_num(features["order_freight"].to_numpy(dtype=float), nan=0.0)
    rto = c_rto(value, freight) * c_rto_scale
    costs = tier_costs(features)
    e = effectiveness_vector(reasons, tier, effectiveness_scale)

    treated = p >= threshold
    impression = costs[tier]["impression"]
    fp_cost = costs[tier]["c_fp"]

    cost = np.zeros_like(rto)
    # Untreated: pay the RTO if it fails.
    cost[~treated & y] = rto[~treated & y]
    # Treated and would have failed: pay the impression, and the residual failure.
    m = treated & y
    cost[m] = impression[m] + (1.0 - e[m]) * rto[m]
    # Treated and would have been fine: pay the impression plus the conditional loss.
    m = treated & ~y
    cost[m] = fp_cost[m]
    return float(cost.sum())
