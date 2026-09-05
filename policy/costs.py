"""
Per-order cost model.  ARCHITECTURE.md 7.1, 7.2.

Nested, not flat.  A 50 BRL order and a 1,500 BRL order do not face the same
intervention bar, so every term is a function of the order and the optimal threshold
becomes a per-order quantity rather than a constant.

--------------------------------------------------------------------------------------
IMPRESSION VS TRIGGERED  (7.2)
--------------------------------------------------------------------------------------

An **impression cost** is paid on every treated order regardless of what the customer
does.  A **triggered cost** is paid only on response.  Collapsing them is a modelling
error, and the tier where it matters most is prepaid-only:

    confirm        impression = message send + fatigue allowance
                   triggered  = P(drop-off at confirm) x margin
    prepaid_only   impression = 0            <- a checkout configuration, not a message
                   triggered  = P(abandon | COD removed) x margin
    defer          impression = manual review cost
                   triggered  = P(abandon | deferred) x margin

Prepaid-only's cost is **entirely conditional**.  With no impression term its false-
positive cost is smaller than confirm's per treated order despite a much larger
conditional loss, which moves its threshold substantially - the point 7.2 makes.

--------------------------------------------------------------------------------------
THE COST MATRIX
--------------------------------------------------------------------------------------

For tier t on order o with effectiveness e:

    C(act, order fails)   = impression(t) + (1 - e) * c_rto(o)
    C(act, order is fine) = impression(t) + triggered(t, o) * ltv(o)
    C(allow, order fails) = c_rto(o)
    C(allow, fine)        = 0

so, in Elkan's terms,

    c_FP(o, t) = impression(t) + triggered(t, o) * ltv(o)
    c_FN(o, t) = e * c_rto(o) - impression(t)

The triggered term is charged only against a *good* order: a customer who abandons an
order that was going to fail anyway is the intervention working, and counting that as a
cost would double-count against the effectiveness term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from policy.constants import get

__all__ = ["TIERS", "ACTION_TIERS", "c_rto", "margin", "ltv_multiplier", "tier_costs"]

#: The action ladder, ARCHITECTURE.md 7.5.  Nobody is blocked - the worst outcome for a
#: false positive is being asked to prepay.
TIERS = ("allow", "confirm", "prepaid_only", "defer")
ACTION_TIERS = ("confirm", "prepaid_only", "defer")


def c_rto(order_value: np.ndarray, forward_freight: np.ndarray) -> np.ndarray:
    """
    Cost of a return-to-origin.

        forward shipping + reverse shipping + handling
        + blocked inventory value x expected days
    """
    reverse = forward_freight * get("reverse_shipping_multiple")
    blocked = (
        order_value
        * get("inventory_holding_rate_daily")
        * get("rto_blocked_days")
    )
    return forward_freight + reverse + get("handling_cost") + blocked


def margin(order_value: np.ndarray) -> np.ndarray:
    return order_value * get("margin_rate")


def ltv_multiplier(prior_orders: np.ndarray) -> np.ndarray:
    """
    Friction applied to a loyal customer costs future orders too (7.1).

    On this panel 97% of customers have no prior order, so the multiplier is 1.0 for
    almost every row - the mechanism is implemented and correct, and it is nearly inert
    on Olist for the same reason the customer-history features are.
    """
    m = 1.0 + get("ltv_multiplier_per_prior_order") * np.nan_to_num(prior_orders, nan=0.0)
    return np.minimum(m, get("ltv_multiplier_cap"))


def tier_costs(features: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """
    Per-order impression and triggered costs for each action tier.

    Returns ``{tier: {"impression": ..., "triggered": ..., "c_fp": ...}}`` where
    ``c_fp`` is the full false-positive cost, impression + triggered x LTV.
    """
    value = np.nan_to_num(features["order_value"].to_numpy(dtype=float), nan=0.0)
    m = margin(value)
    ltv = ltv_multiplier(features["cust_prior_orders"].to_numpy(dtype=float))

    zeros = np.zeros_like(value)
    out: dict[str, dict[str, np.ndarray]] = {}

    out["confirm"] = {
        "impression": zeros + get("message_cost") + get("fatigue_allowance"),
        "triggered": get("p_dropoff_at_confirm") * m,
    }
    out["prepaid_only"] = {
        # Near-zero by construction: a checkout configuration, not a send.
        "impression": zeros,
        "triggered": get("p_abandon_if_cod_removed") * m,
    }
    out["defer"] = {
        "impression": zeros + get("manual_review_cost"),
        "triggered": get("p_abandon_if_deferred") * m,
    }

    for tier in out:
        out[tier]["c_fp"] = out[tier]["impression"] + out[tier]["triggered"] * ltv
    return out
