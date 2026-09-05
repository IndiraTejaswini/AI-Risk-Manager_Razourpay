"""Expose the per-order policy costs carried by a score response."""

from __future__ import annotations

from typing import Any
import pandas as pd

from policy.costs import ACTION_TIERS, tier_costs

__all__ = ["explain_cost"]


def explain_cost(policy_result: Any, features: pd.DataFrame, index: int = 0) -> dict[str, Any]:
    """
    Return the cost terms that produce the response's ``threshold_used``.

    ``PolicyResult`` already contains the cost calculation.  The impression term is
    looked up from the same cost table so the two friction terms remain separately
    visible while the algebra stays identical to ``policy.elkan``.
    """
    selected = str(policy_result.tier[index])
    if selected == "allow":
        selected = min(
            ACTION_TIERS,
            key=lambda tier: float(policy_result.thresholds[tier][index]),
        )

    impression = float(tier_costs(features)[selected]["impression"][index])
    expected_rto_loss = float(policy_result.c_fn[selected][index] + impression)
    expected_triggered_cost = float(policy_result.c_fp[selected][index] - impression)

    return {
        "expected_rto_loss": expected_rto_loss,
        "impression_cost": impression,
        "expected_triggered_cost": expected_triggered_cost,
        "currency": "BRL",
        "basis": "assumed_cost_constants",
    }
