"""
Razorpay-shaped request and response contract.  ARCHITECTURE.md 11.1.

The payload is modelled on Razorpay's **webhook event** envelope wrapping Orders and
Payments entities, not a generic order dict.  Half a day of work, and it is the
difference between "a return-risk model" and "a return-risk model for Razorpay
merchants": the integration is a webhook subscription rather than a bespoke mapping.

Shape follows the platform's conventions:

  * a webhook is ``{entity: "event", event: ..., contains: [...], payload: {...}}``
  * entities nest under ``payload.<name>.entity``
  * **amounts are integers in the currency's smallest unit** (paise for INR, centavos
    for BRL).  A float amount is a rounding bug waiting to happen and is rejected.
  * ``created_at`` is a Unix timestamp in seconds
  * ``notes`` is a free-form string map and is ignored by the scorer

--------------------------------------------------------------------------------------
CURRENCY
--------------------------------------------------------------------------------------

The model is trained on Brazilian orders and every cost constant is BRL (see
policy/constants.py).  A payload in another currency would be scored against magnitudes
that mean something different - a 50,000-unit order is R$500 or Rs 500 depending on the
label, and the cost model would silently price the wrong one.

So the endpoint **requires ``currency: "BRL"``** and rejects anything else with a clear
message rather than converting behind the caller's back.  That is a real limitation of a
model trained on one market, and it is surfaced at the contract rather than buried.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "ScoreRequest",
    "ScoreResponse",
    "SUPPORTED_CURRENCY",
    "FORBIDDEN_PAYLOAD_FIELDS",
    "CostBreakdown",
]

SUPPORTED_CURRENCY = "BRL"

#: Fields that carry outcome information.  A caller replaying a historical order could
#: include them without meaning to leak; the endpoint rejects the request rather than
#: ignoring them, so a client sending post-outcome data finds out immediately.
FORBIDDEN_PAYLOAD_FIELDS = frozenset({
    "order_status",
    "status",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "delivered_at",
    "shipped_at",
    "order_approved_at",
    "order_estimated_delivery_date",
    "review_score",
    "label",
    "label_a",
    "label_b",
})


class OrderEntity(BaseModel):
    """Razorpay Orders object, trimmed to the checkout-time admissible fields."""

    id: str = Field(..., examples=["order_EKwxwAgItmmXdp"])
    entity: Literal["order"] = "order"
    amount: int = Field(..., ge=0, description="Smallest currency unit (centavos)")
    currency: str = Field(..., examples=[SUPPORTED_CURRENCY])
    receipt: str | None = None
    created_at: int = Field(..., description="Unix seconds")
    notes: dict[str, str] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def _supported(cls, v: str) -> str:
        if v.upper() != SUPPORTED_CURRENCY:
            raise ValueError(
                f"currency {v!r} is not supported. This model is trained on Brazilian "
                f"orders and every cost constant is {SUPPORTED_CURRENCY}; scoring "
                "another currency would price the wrong magnitudes. No conversion is "
                "applied deliberately - see policy/constants.py."
            )
        return v.upper()


class PaymentEntity(BaseModel):
    """Razorpay Payments object. ``method`` carries the COD/deferred signal."""

    id: str | None = None
    entity: Literal["payment"] = "payment"
    amount: int = Field(..., ge=0)
    currency: str = SUPPORTED_CURRENCY
    method: Literal["card", "netbanking", "wallet", "upi", "cod", "boleto",
                    "emi", "paylater"] = "card"
    international: bool = False
    # Razorpay exposes instalment count on EMI payments; maps to payment_installments.
    emi_installments: int | None = Field(default=None, ge=1)
    notes: dict[str, str] = Field(default_factory=dict)


class LineItem(BaseModel):
    """One order line. Maps to olist_order_items."""

    product_id: str | None = None
    seller_id: str | None = None
    amount: int = Field(..., ge=0, description="Unit price, smallest currency unit")
    quantity: int = Field(default=1, ge=1)
    shipping_amount: int = Field(default=0, ge=0)
    category: str | None = Field(default=None, description="Product category slug")
    weight_g: float | None = None
    length_cm: float | None = None
    height_cm: float | None = None
    width_cm: float | None = None
    photos_qty: float | None = None


class CustomerEntity(BaseModel):
    """
    Razorpay Customers object plus the merchant's own customer key.

    ``customer_reference`` is the stable identity the history is keyed on. Absent for a
    guest checkout, which is the cold-start path and **97% of this panel's traffic**.
    """

    id: str | None = None
    entity: Literal["customer"] = "customer"
    customer_reference: str | None = None
    name: str | None = None
    contact: str | None = None
    email: str | None = None


class ShippingAddress(BaseModel):
    """Only the pincode is used. Address strings are [BUILT-UNEVAL], see 4.2."""

    zipcode: str = Field(..., examples=["14409"])
    city: str | None = None
    state: str | None = None
    country: str | None = None


class EventPayload(BaseModel):
    order: dict[Literal["entity"], OrderEntity]
    payment: dict[Literal["entity"], PaymentEntity] | None = None
    customer: dict[Literal["entity"], CustomerEntity] | None = None
    shipping_address: dict[Literal["entity"], ShippingAddress]
    line_items: list[LineItem] = Field(default_factory=list)


class ScoreRequest(BaseModel):
    """Webhook-shaped envelope. ``POST /score`` accepts exactly this."""

    entity: Literal["event"] = "event"
    account_id: str = Field(..., examples=["acc_BFQ7uQEaa7j2z7"])
    event: str = Field(default="order.created", examples=["order.created"])
    contains: list[str] = Field(default_factory=lambda: ["order", "payment"])
    created_at: int
    payload: EventPayload


class CostBreakdown(BaseModel):
    expected_rto_loss: float
    impression_cost: float
    expected_triggered_cost: float
    currency: Literal["BRL"]
    basis: Literal["assumed_cost_constants"]


class ScoreResponse(BaseModel):
    """
    ARCHITECTURE.md 11.1, exactly.

    ``risk`` is the calibrated P(fails | ships) - conditional on the order shipping, per
    the primary target. ``threshold_used`` is the per-order Elkan p* for the tier that
    was selected, or for the cheapest action tier when the decision is ``allow``.
    """

    risk: float = Field(..., ge=0.0, le=1.0)
    tier: Literal["allow", "confirm", "prepaid_only", "defer"]
    reasons: list[str]
    model_version: str
    threshold_used: float
    features_missing: int
    cost_breakdown: CostBreakdown
    decision_id: str | None = None
