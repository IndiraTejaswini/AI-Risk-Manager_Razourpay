"""Runtime provenance tracking for reported values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Real


class Tier(IntEnum):
    ASSUMED = 0
    MEASURED = 2
    HUMAN = 3


class UngroundedClaim(ValueError):
    """A value with insufficient provenance was presented as measured."""


@dataclass(frozen=True)
class Tagged:
    value: float
    tier: Tier
    source: str

    def _combine(self, other: Tagged | Real, operation) -> Tagged:
        if isinstance(other, Tagged):
            return Tagged(operation(self.value, other.value), min(self.tier, other.tier),
                          f"{self.source} * {other.source}")
        return Tagged(operation(self.value, float(other)), self.tier, self.source)

    def __add__(self, other):
        return self._combine(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._combine(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._combine(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._combine(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._combine(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._combine(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._combine(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._combine(other, lambda a, b: b / a)


def render_measured(value: Tagged) -> str:
    if value.tier < Tier.MEASURED:
        raise UngroundedClaim(
            f"{value.source} is {value.tier.name}; cannot render as measured"
        )
    return f"{value.value:g}"
