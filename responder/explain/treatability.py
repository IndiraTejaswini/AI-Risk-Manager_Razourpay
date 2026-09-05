"""Assumption-labelled explanations of which intervention may help."""

from __future__ import annotations

from dataclasses import dataclass

from policy.effectiveness import EFFECTIVENESS
from responder.templates.registry import ReasonClass, Tier
from responder.trust import Tagged, Tier as TrustTier


_POLICY_REASON = {
    ReasonClass.ADDRESS_QUALITY: "order_composition",
    ReasonClass.PINCODE_HISTORY: "pincode",
    ReasonClass.CUSTOMER_HISTORY: "customer_history",
    ReasonClass.ORDER_STRUCTURE: "order_composition",
    ReasonClass.CATEGORY_SEASONALITY: "availability",
    ReasonClass.RING_SIGNAL: "customer_history",
}

_TIER_LABELS = {
    "confirm": "confirmation",
    "prepaid_only": "prepaid-only",
    "defer": "deferral",
}


@dataclass(frozen=True)
class Treatability:
    reason_class: ReasonClass
    current_tier: Tier
    best_tier: Tier
    effectiveness: Tagged
    sentence: str


def _coerce_reason(value: ReasonClass | str) -> ReasonClass:
    if isinstance(value, ReasonClass):
        return value
    try:
        return ReasonClass(value)
    except ValueError as exc:
        raise ValueError(f"unmapped treatability reason class: {value!r}") from exc


def _coerce_tier(value: Tier | str) -> Tier:
    if isinstance(value, Tier):
        return value
    try:
        return Tier(value)
    except ValueError as exc:
        raise ValueError(f"unmapped treatability tier: {value!r}") from exc


def treatability(reason_class: ReasonClass | str, tier: Tier | str) -> Treatability:
    reason = _coerce_reason(reason_class)
    current = _coerce_tier(tier)
    policy_reason = _POLICY_REASON[reason]
    values = EFFECTIVENESS[policy_reason]
    best_value = max(values.values())
    best_name = next(name for name, value in values.items() if value == best_value)
    best_tier = Tier(best_name)
    current_value = values[current.value]
    effectiveness = Tagged(
        current_value,
        tier=TrustTier.ASSUMED,
        source="effectiveness_prior",
    )

    sentence = (
        f"This order is risky mainly because of {reason.value.replace('_', ' ')}. "
        f"The {_TIER_LABELS[best_name]} lever has the highest assumed effectiveness."
    )
    cheaper = [
        (name, value)
        for name, value in values.items()
        if value < best_value and value >= best_value * 0.9
    ]
    if cheaper:
        cheaper_name, _ = max(cheaper, key=lambda item: item[1])
        sentence += (
            f" {_TIER_LABELS[cheaper_name].capitalize()} is a cheaper option "
            "with a close assumed effect."
        )
    if current_value != best_value:
        sentence += f" The current {_TIER_LABELS[current.value]} tier is not the strongest lever."
    return Treatability(reason, current, best_tier, effectiveness, sentence)


def render_treatability(reason_class: ReasonClass | str, tier: Tier | str) -> str:
    """Render only the assumption-labelled treatability sentence."""
    return treatability(reason_class, tier).sentence
