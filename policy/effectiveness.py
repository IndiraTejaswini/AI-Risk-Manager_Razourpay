"""
Intervention effectiveness.  ARCHITECTURE.md 7.3.

The central correction in 7.3: **the highest-risk orders are not the orders where
intervention helps most.**  A fake or competitor-harassment order is high risk and has
near-zero uplift from a confirmation SMS - nothing you send changes the outcome, you pay
and get nothing.  An address missing a landmark is moderate risk and high uplift from an
address-repair prompt.  So effectiveness is a function of *why* the order is risky, not a
constant.

    effectiveness[reason][tier]

Every value here is an **ASSUMPTION**.  The structure is defensible; the numbers are not
measured and cannot be measured offline, because the counterfactual does not exist in
observational data.  They are swept in the report rather than presented as known.

--------------------------------------------------------------------------------------
REASON BUCKETS - SHAP IS NOW THE SOURCE
--------------------------------------------------------------------------------------

7.3 buckets on the dominant SHAP reason, and as of step 6 that is what the pipeline
uses: ``ReasonExplainer.reason_buckets`` returns the group with the largest summed
**positive** attribution, which is how much each group actually pushed the score up.

:func:`dominant_reason` below is the pre-SHAP stand-in it replaced - the dominant feature
group by standardised deviation from the training median.  It is **retained but no longer
the production path**, because the two disagree on 47% of orders and the comparison is
reported in eval/reasons.md section 7.  Keeping it makes that comparison reproducible;
using it would mean bucketing by how unusual an order looks rather than by what moved
its score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["REASONS", "EFFECTIVENESS", "dominant_reason", "effectiveness_vector"]

REASONS = ("order_composition", "pincode", "customer_history", "availability")

#: ASSUMPTION, every cell.  Swept in the report.
#:
#: The shape encodes 7.3's argument:
#:   - a confirmation prompt repairs a shaky-looking order, and does nothing for a
#:     customer with a history of failures
#:   - prepaid-only is the tier that works on a serial refuser, because it removes the
#:     mechanism rather than asking about it
#:   - a pincode with an elevated failure rate is a delivery-geography problem; asking
#:     the customer to confirm does not move it much, removing COD does
EFFECTIVENESS: dict[str, dict[str, float]] = {
    "order_composition": {"confirm": 0.35, "prepaid_only": 0.25, "defer": 0.40},
    "pincode":           {"confirm": 0.15, "prepaid_only": 0.45, "defer": 0.50},
    "customer_history":  {"confirm": 0.10, "prepaid_only": 0.60, "defer": 0.65},
    "availability":      {"confirm": 0.20, "prepaid_only": 0.30, "defer": 0.35},
}

#: Representative feature per group, used for the pre-SHAP bucketing.
_REPRESENTATIVE = {
    "order_composition": "freight_ratio",
    "pincode": "pincode_failure_rate_smoothed",
    "customer_history": "cust_prior_failure_rate",
    "availability": "n_missing_features",
}


def dominant_reason(
    features: pd.DataFrame, reference: pd.DataFrame | None = None
) -> np.ndarray:
    """
    **Superseded by SHAP.**  Retained so eval/reasons.md section 7 can reproduce the
    comparison between this stand-in and the attribution-based buckets; not the
    production path.  Use ``ReasonExplainer.reason_buckets``.

    Bucket each order by the feature group on which it looks most unusual.

    Standardisation uses the **reference** frame's median and IQR - the training split -
    so the bucketing does not depend on the population being scored.  Ties break toward
    the earlier group in :data:`REASONS`, deterministically.
    """
    ref = features if reference is None else reference
    scores = np.zeros((len(features), len(REASONS)), dtype=float)

    for j, reason in enumerate(REASONS):
        col = _REPRESENTATIVE[reason]
        if col not in features.columns:
            continue
        v = features[col].to_numpy(dtype=float)
        r = ref[col].to_numpy(dtype=float)
        med = np.nanmedian(r)
        q75, q25 = np.nanpercentile(r, 75), np.nanpercentile(r, 25)
        scale = (q75 - q25) or 1.0
        scores[:, j] = np.nan_to_num((v - med) / scale, nan=0.0)

    return np.array(REASONS)[np.argmax(scores, axis=1)]


def effectiveness_vector(
    reasons: np.ndarray, tier: str, scale: float = 1.0
) -> np.ndarray:
    """
    Per-order effectiveness for ``tier``, given each order's reason bucket.

    ``scale`` multiplies every cell, capped at 1.0 - the sweep axis.  A scale of 0
    means the intervention never works, which is the pessimistic end of 7.3's range and
    the one that should make the policy stand down.
    """
    table = {r: min(1.0, EFFECTIVENESS[r][tier] * scale) for r in REASONS}
    return np.array([table[r] for r in reasons], dtype=float)
