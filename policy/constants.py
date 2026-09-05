"""
Every assumed constant in the policy layer, in one module.

--------------------------------------------------------------------------------------
CURRENCY - READ THIS FIRST
--------------------------------------------------------------------------------------

**Everything in this module and every figure downstream of it is in BRL (Brazilian
reais).**  Olist is a Brazilian panel; ``price`` and ``freight_value`` are BRL.

The track is Indian and ARCHITECTURE.md writes costs in rupees.  This repo does **not**
convert.  Relabelling BRL as INR would put a rupee sign on Brazilian magnitudes and make
every number in the cost table wrong by whatever the exchange rate happens to be - the
kind of thing that is invisible in a slide and obvious in a spreadsheet.

A conversion rate is recorded below as an assumption **and is deliberately not applied**.
It exists so a reader can do the arithmetic themselves and see exactly what it would
cost them.

--------------------------------------------------------------------------------------
BASIS TAGS
--------------------------------------------------------------------------------------

    OBSERVED     measured from the Olist data
    ASSUMPTION   a stated value with reasoning; not measured here
    GUESS        a number with no defensible basis, included because omitting the term
                 would be worse than estimating it badly

Only ``price`` and ``freight_value`` are OBSERVED.  Everything else that enters a cost is
ASSUMPTION or GUESS, and the report enumerates all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Assumption", "ASSUMPTIONS", "get", "CURRENCY", "BRL_TO_INR_REFERENCE"]

CURRENCY = "BRL"


@dataclass(frozen=True)
class Assumption:
    name: str
    value: float
    unit: str
    basis: str          # OBSERVED | ASSUMPTION | GUESS
    rationale: str


_REGISTRY: list[Assumption] = [
    # -- observed ----------------------------------------------------------------------
    Assumption(
        "order_value", float("nan"), "BRL", "OBSERVED",
        "sum of olist_order_items.price for the order",
    ),
    Assumption(
        "forward_freight", float("nan"), "BRL", "OBSERVED",
        "sum of olist_order_items.freight_value; the outbound leg actually charged",
    ),

    # -- c_rto -------------------------------------------------------------------------
    Assumption(
        "reverse_shipping_multiple", 1.0, "x forward freight", "ASSUMPTION",
        "the return leg is priced as the outbound leg. Reverse logistics is often "
        "dearer per parcel than forward, so this is the conservative end",
    ),
    Assumption(
        "handling_cost", 8.00, "BRL per RTO", "ASSUMPTION",
        "receiving, inspecting and restocking a returned parcel; a fixed per-parcel "
        "charge independent of order value",
    ),
    Assumption(
        "inventory_holding_rate_daily", 0.0005, "fraction of value per day", "ASSUMPTION",
        "~18% annualised carrying cost, the standard planning figure for working "
        "capital plus warehousing",
    ),
    Assumption(
        "rto_blocked_days", 20.0, "days", "ASSUMPTION",
        "round-trip time during which the unit is neither sold nor sellable: outbound "
        "transit, failed delivery attempts, return transit, restock",
    ),

    # -- margin ------------------------------------------------------------------------
    Assumption(
        "margin_rate", 0.20, "fraction of order value", "ASSUMPTION",
        "contribution margin on a marketplace order. Enters every conversion-loss term, "
        "so the whole friction side scales with it",
    ),

    # -- friction: impression ----------------------------------------------------------
    Assumption(
        "message_cost", 0.30, "BRL per message", "ASSUMPTION",
        "one WhatsApp/SMS send. Paid on every treated order regardless of response",
    ),
    Assumption(
        "fatigue_allowance", 0.50, "BRL per message", "GUESS",
        "brand annoyance and the unsubscribe analogue. There is no measurement behind "
        "this number and none is available offline. It is included because setting it "
        "to zero asserts that messaging customers is free, which is false, and the "
        "sensitivity sweep is the honest treatment of it",
    ),
    Assumption(
        "manual_review_cost", 3.00, "BRL per order", "ASSUMPTION",
        "analyst time to hold and inspect a deferred order. Impression cost of the "
        "defer tier - paid whether or not the order is released",
    ),

    # -- friction: triggered -----------------------------------------------------------
    Assumption(
        "p_dropoff_at_confirm", 0.02, "probability", "ASSUMPTION",
        "a good customer who cancels because they were asked to confirm. Triggered: "
        "paid only when it happens",
    ),
    Assumption(
        "p_abandon_if_cod_removed", 0.15, "probability", "ASSUMPTION",
        "cash-on-delivery is a payment preference, not only a payment method; removing "
        "it loses a real share of otherwise-good orders. The dominant term in the "
        "prepaid-only tier, whose impression cost is ~0",
    ),
    Assumption(
        "p_abandon_if_deferred", 0.25, "probability", "ASSUMPTION",
        "holding an order for review costs more conversions than a confirmation prompt",
    ),

    # -- LTV ---------------------------------------------------------------------------
    Assumption(
        "ltv_multiplier_per_prior_order", 0.15, "x friction per prior order", "ASSUMPTION",
        "friction applied to a loyal customer costs future orders, not just this one "
        "(section 7.1). Multiplier is 1 + 0.15 x prior orders, capped",
    ),
    Assumption(
        "ltv_multiplier_cap", 2.0, "x", "ASSUMPTION",
        "ceiling on the LTV multiplier so a single high-frequency customer cannot "
        "dominate the cost table",
    ),

    # -- currency ----------------------------------------------------------------------
    Assumption(
        "brl_to_inr_reference", 15.5, "INR per BRL", "ASSUMPTION",
        "NOT APPLIED. Approximate mid-market rate, 2025. Recorded so a reader can "
        "convert the tables themselves; every number in this repo stays in BRL",
    ),
]

ASSUMPTIONS: dict[str, Assumption] = {a.name: a for a in _REGISTRY}

BRL_TO_INR_REFERENCE = ASSUMPTIONS["brl_to_inr_reference"].value


def get(name: str) -> float:
    """Value of an assumed constant. Raises if the name is not registered."""
    if name not in ASSUMPTIONS:
        raise KeyError(
            f"{name!r} is not a registered assumption. Every constant entering a cost "
            "must be declared in policy/constants.py with a basis tag."
        )
    return ASSUMPTIONS[name].value
