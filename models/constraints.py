"""
Monotonic constraints.  ARCHITECTURE.md 5.3.

The spec lists four directional constraints:

    pincode_failure_rate  ↑  → risk ↑
    prior_failures        ↑  → risk ↑
    prepaid_ratio         ↑  → risk ↓
    component_size        ↑  → risk ↑

``component_size`` is omitted: the network/ring group is not in this feature matrix
(ARCHITECTURE.md 4.2 - Olist carries no device identifiers or phone numbers).  The
remaining three are mapped below.

--------------------------------------------------------------------------------------
THE SIGN INVERSION - READ THIS BEFORE EDITING
--------------------------------------------------------------------------------------

The spec constrains ``prepaid_ratio``.  This matrix does not carry a prepaid ratio; it
carries ``cust_prior_boleto_ratio``, which is its **complement**.  Boleto is the COD
analog on Olist - deferred payment, no card capture at checkout - so:

    prepaid_ratio = 1 - boleto_ratio

    prepaid_ratio ↑ → risk ↓        (the spec)
    boleto_ratio  ↑ → risk ↑        (the same statement about this matrix's feature)

so ``cust_prior_boleto_ratio`` takes **+1**, not -1.

A sign error here is worse than having no constraint at all: it would force the model to
learn a relationship that is backwards, and the resulting SHAP strings would confidently
tell a merchant the opposite of the truth - which is exactly the failure mode 5.3 exists
to prevent.  The inversion is recorded in :data:`INVERTED_FROM_SPEC` and asserted in
tests/test_model.py rather than left as a comment.
"""

from __future__ import annotations

__all__ = [
    "MONOTONE_CONSTRAINTS",
    "UNSHIPPED_CONSTRAINTS",
    "ALL_CONSTRAINTS",
    "INVERTED_FROM_SPEC",
    "OMITTED_FROM_SPEC",
    "ConstraintError",
    "build_constraint_vector",
]


class ConstraintError(ValueError):
    """Raised when the constraint map cannot be applied to a feature list."""


#: The named constant.  Feature name -> LightGBM monotone constraint.
#:   +1  feature increases  => predicted risk may only increase
#:   -1  feature increases  => predicted risk may only decrease
#:    0  unconstrained (the default for anything absent from this map)
#:
#: Loaded by name; never inlined at a call site.
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "pincode_failure_rate_smoothed": +1,   # spec: pincode_failure_rate ↑ → risk ↑
    "cust_prior_failures": +1,             # spec: prior_failures       ↑ → risk ↑
    "cust_prior_boleto_ratio": +1,         # spec: prepaid_ratio        ↑ → risk ↓  (INVERTED)
}

#: Constraints on features that exist but are NOT in the shipped matrix (the seller and
#: route groups - eval/feature_expansion.md).
#:
#: They are kept apart from :data:`MONOTONE_CONSTRAINTS` for one reason: that map is
#: *required*, and requiring a constraint on a feature the shipped model does not carry
#: would make every ordinary call site declare an exception.  A guard that fires on the
#: normal case is a guard people learn to silence.
#:
#: The rename protection is not lost, only relocated: tests/test_feature_expansion.py
#: asserts every key here is present and carries +1 in the expanded matrix, so renaming
#: one still fails loudly - in the test that owns that matrix.
#:
#: Same argument as the pincode rate: a historical failure rate that rises must not
#: reduce predicted risk, or the reason string contradicts itself in front of a
#: merchant.
UNSHIPPED_CONSTRAINTS: dict[str, int] = {
    "seller_failure_rate_smoothed": +1,
    "route_pair_failure_rate_smoothed": +1,
}

#: Every constraint the vector builder will apply.  Presence is optional for the
#: unshipped half; the sign is not.
ALL_CONSTRAINTS: dict[str, int] = {**MONOTONE_CONSTRAINTS, **UNSHIPPED_CONSTRAINTS}

#: Features whose sign is flipped relative to the ARCHITECTURE.md wording, with the
#: reason.  Asserted in the test suite so the inversion cannot be "corrected" by someone
#: reading 5.3 alone.
INVERTED_FROM_SPEC: dict[str, str] = {
    "cust_prior_boleto_ratio": (
        "spec constrains prepaid_ratio (-1); this matrix carries the boleto ratio, its "
        "complement (prepaid = 1 - boleto), so the constraint is +1"
    ),
}

#: Spec constraints with no corresponding column in this matrix, and why.
OMITTED_FROM_SPEC: dict[str, str] = {
    "component_size": (
        "network/ring group is not built for Olist (ARCHITECTURE.md 4.2 - no device "
        "identifiers or phone numbers in the data)"
    ),
}


def build_constraint_vector(
    feature_names: list[str] | tuple[str, ...],
    allow_missing: frozenset[str] = frozenset(),
) -> list[int]:
    """
    Return the monotone constraint vector aligned to ``feature_names``.

    Parameters
    ----------
    allow_missing:
        Constrained features the caller **knows** are absent, named explicitly.  A
        deliberately reduced matrix - the order-only ablation, say - legitimately
        carries none of them.

        This is not a switch that disables the check.  The caller has to name each
        absent feature, so a *renamed* or accidentally dropped constraint still raises,
        and adding a fourth constraint to a group the reduced matrix does include fails
        loudly rather than being silently permitted.

    Raises
    ------
    ConstraintError
        If a key in :data:`MONOTONE_CONSTRAINTS` is absent from ``feature_names`` and
        was not named in ``allow_missing``.  Silently emitting a vector with the
        constraint missing would ship an unconstrained model that still claims 5.3's
        guarantees.

        :data:`UNSHIPPED_CONSTRAINTS` is **not** required - those features are absent
        from the shipped matrix by design - but is applied wherever present.
    """
    names = list(feature_names)
    unknown = sorted(allow_missing - set(MONOTONE_CONSTRAINTS))
    if unknown:
        raise ConstraintError(
            "allow_missing names feature(s) that carry no constraint: "
            + ", ".join(unknown)
            + ". Probably a typo, or a constraint that was renamed."
        )

    missing = sorted(set(MONOTONE_CONSTRAINTS) - set(names) - allow_missing)
    if missing:
        raise ConstraintError(
            "constrained feature(s) absent from the matrix: "
            + ", ".join(missing)
            + ". The constraint would be silently dropped; rename the entry in "
            "MONOTONE_CONSTRAINTS, add the feature, or name it in allow_missing if "
            "this matrix is deliberately reduced."
        )

    vector = [ALL_CONSTRAINTS.get(name, 0) for name in names]

    if len(vector) != len(names):  # pragma: no cover - structurally impossible
        raise ConstraintError(
            f"constraint vector length {len(vector)} != feature count {len(names)}"
        )
    return vector
