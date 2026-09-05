"""Closed, typed, deterministic message-template registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from models.explain import NO_RISK_REASON, REASON_TEMPLATES


class ReasonClass(str, Enum):
    ADDRESS_QUALITY = "address_quality"
    PINCODE_HISTORY = "pincode_history"
    CUSTOMER_HISTORY = "customer_history"
    ORDER_STRUCTURE = "order_structure"
    CATEGORY_SEASONALITY = "category_seasonality"
    RING_SIGNAL = "ring_signal"


class Tier(str, Enum):
    CONFIRM = "confirm"
    PREPAID_ONLY = "prepaid_only"
    DEFER = "defer"


@dataclass(frozen=True)
class Template:
    id: str
    version: str
    message_class: str
    field_names: tuple[str, ...]
    _render: Callable[[Mapping[str, str]], str]

    def render(self, fields: Mapping[str, str]) -> str:
        missing = set(self.field_names) - fields.keys()
        if missing:
            raise KeyError(f"missing template fields: {sorted(missing)}")
        return self._render(fields) + "\n"


_FEATURE_CLASS = {
    "pincode_failure_rate_smoothed": ReasonClass.PINCODE_HISTORY,
    "pincode_prior_orders": ReasonClass.PINCODE_HISTORY,
    "pincode_prior_failures": ReasonClass.PINCODE_HISTORY,
    "global_prior_failure_rate": ReasonClass.CATEGORY_SEASONALITY,
    "cust_prior_failures": ReasonClass.CUSTOMER_HISTORY,
    "cust_prior_failure_rate": ReasonClass.CUSTOMER_HISTORY,
    "cust_prior_orders": ReasonClass.CUSTOMER_HISTORY,
    "cust_prior_boleto_ratio": ReasonClass.CUSTOMER_HISTORY,
    "cust_prior_avg_value": ReasonClass.CUSTOMER_HISTORY,
    "cust_days_since_prior_order": ReasonClass.CUSTOMER_HISTORY,
    "product_category": ReasonClass.CATEGORY_SEASONALITY,
    "purchase_hour": ReasonClass.CATEGORY_SEASONALITY,
    "purchase_dow": ReasonClass.CATEGORY_SEASONALITY,
    "purchase_month": ReasonClass.CATEGORY_SEASONALITY,
    "purchase_is_weekend": ReasonClass.CATEGORY_SEASONALITY,
}
_FEATURE_CLASS.update(
    {key: ReasonClass.ORDER_STRUCTURE for key in REASON_TEMPLATES if key not in _FEATURE_CLASS}
)
REASON_CLASS_BY_REASON = {NO_RISK_REASON: ReasonClass.ORDER_STRUCTURE}
REASON_CLASS_BY_REASON.update({
    REASON_TEMPLATES[key]: _FEATURE_CLASS[key] for key in REASON_TEMPLATES
})


def reason_class(reason: str) -> ReasonClass:
    try:
        return REASON_CLASS_BY_REASON[reason]
    except KeyError as exc:
        raise ValueError(f"unmapped detector reason: {reason!r}") from exc


def _service(fields: Mapping[str, str]) -> str:
    return f"Please verify your delivery address at {fields['address']}."


def _prepaid(fields: Mapping[str, str]) -> str:
    return f"Please complete payment for order {fields['order_id']} before dispatch."


def _defer(fields: Mapping[str, str]) -> str:
    return f"We need one more delivery detail for order {fields['order_id']}."


def _template(reason: ReasonClass, tier: Tier) -> Template:
    render = _service if tier is Tier.CONFIRM else _prepaid if tier is Tier.PREPAID_ONLY else _defer
    fields = ("address",) if tier is Tier.CONFIRM else ("order_id",)
    message_class = "SERVICE" if tier is Tier.CONFIRM else "PROMOTIONAL"
    return Template(f"{reason.value}_{tier.value}", "v1", message_class, fields, render)


TEMPLATES: dict[tuple[ReasonClass, Tier], Template] = {
    (reason, tier): _template(reason, tier)
    for reason in ReasonClass
    for tier in Tier
}
