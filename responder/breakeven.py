"""Measured break-even effectiveness for the replayed treated set."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.costs import ACTION_TIERS, c_rto, tier_costs
from policy.effectiveness import effectiveness_vector
from policy.elkan import apply_policy
from responder.replay import _data
from responder.trust import Tagged, Tier, render_measured

FIG_PATH = ROOT / "eval" / "responder" / "breakeven.png"
OUT_PATH = ROOT / "eval" / "responder" / "breakeven.md"


def compute_break_even(
    treated_failures: Tagged | float,
    impression_cost: Tagged | float,
    triggered_cost: Tagged | float,
    c_rto: Tagged | float,
) -> Tagged:
    """Return the observed failure-prevention fraction needed to break even."""
    if not isinstance(treated_failures, Tagged):
        treated_failures = Tagged(float(treated_failures), Tier.MEASURED, "treated_failures")
    if not isinstance(impression_cost, Tagged):
        impression_cost = Tagged(float(impression_cost), Tier.MEASURED, "impression_cost")
    if not isinstance(triggered_cost, Tagged):
        triggered_cost = Tagged(float(triggered_cost), Tier.MEASURED, "triggered_cost")
    if not isinstance(c_rto, Tagged):
        c_rto = Tagged(float(c_rto), Tier.MEASURED, "c_rto_sweep")
    denominator = treated_failures * c_rto
    if denominator.value <= 0:
        raise ValueError("treated failure RTO denominator must be positive")
    return (impression_cost + triggered_cost) / denominator


def sweep_break_even() -> dict[str, Tagged]:
    """Calculate the ±50% c_rto band from observed treated failures."""
    _, features, labels, reasons, probabilities = _data()
    policy = apply_policy(probabilities, features, reasons)
    costs = tier_costs(features)
    base_rto = c_rto(
        np.nan_to_num(features["order_value"].to_numpy(float), nan=0.0),
        np.nan_to_num(features["order_freight"].to_numpy(float), nan=0.0),
    )
    treated = np.zeros(len(labels), dtype=bool)
    impression = np.zeros(len(labels))
    triggered = np.zeros(len(labels))
    for tier in ACTION_TIERS:
        mask = policy.tier == tier
        e = effectiveness_vector(reasons, tier)
        mask &= e * base_rto > costs[tier]["impression"]
        treated |= mask
        impression[mask] = costs[tier]["impression"][mask]
        triggered[mask] = costs[tier]["triggered"][mask] * (~labels[mask].astype(bool))
    failures = treated & labels.astype(bool)
    numerator = Tagged(
        float(impression.sum() + triggered.sum()),
        Tier.MEASURED,
        "observed_impression_plus_triggered_cost",
    )
    failure_count = int(failures.sum())
    if failure_count == 0:
        raise ValueError("treated set contains no observed failures")
    average_rto = float(base_rto[failures].mean())
    return {
        "low": compute_break_even(
            Tagged(failure_count, Tier.MEASURED, "treated_failures"),
            numerator,
            Tagged(0.0, Tier.MEASURED, "zero_additional_cost"),
            Tagged(average_rto * 0.5, Tier.MEASURED, "c_rto_sweep"),
        ),
        "base": compute_break_even(
            Tagged(failure_count, Tier.MEASURED, "treated_failures"),
            numerator,
            Tagged(0.0, Tier.MEASURED, "zero_additional_cost"),
            Tagged(average_rto, Tier.MEASURED, "c_rto_sweep"),
        ),
        "high": compute_break_even(
            Tagged(failure_count, Tier.MEASURED, "treated_failures"),
            numerator,
            Tagged(0.0, Tier.MEASURED, "zero_additional_cost"),
            Tagged(average_rto * 1.5, Tier.MEASURED, "c_rto_sweep"),
        ),
    }


def main() -> int:
    values = sweep_break_even()
    labels = ["-50%", "base", "+50%"]
    points = [values[key].value for key in ("low", "base", "high")]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(labels, np.asarray(points) * 100, "o-", label="Measured break-even")
    ax.axhspan(5, 10, color="#7aa6d8", alpha=0.25, label="External OTP claim: 5–10 pp")
    ax.axhspan(20, 40, color="#d89b7a", alpha=0.25, label="External confirmation claim: 20–40%")
    ax.set_ylabel("Required failure prevention (%)")
    ax.set_title("Break-even effectiveness; external bands are unverified vendor claims")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)

    # Scale to percent before rendering, not after.  render_measured prints the value
    # it is given, so appending a literal "%" to the bare fraction reported 0.0598 as
    # "0.0598%" - a hundredfold understatement of the same quantity responder's own
    # policy_table.py prints as 5.98% with a ".2%" format spec.  The plot above has
    # always multiplied; only this table did not.  Tagged * float keeps the tier and
    # the source, so the provenance chain render_measured checks is unbroken.
    rendered = [render_measured(values[key] * 100) for key in ("low", "base", "high")]
    OUT_PATH.write_text(
        "\n".join([
            "# Break-even effectiveness",
            "",
            "Generated by `responder/breakeven.py` from observed outcomes in the primary Tier-1 test window.",
            "No effectiveness prior enters the calculation.",
            "",
            "| `c_rto` sweep | Required effectiveness |",
            "|---:|---:|",
            *[f"| {label} | {value}% |" for label, value in zip(labels, rendered)],
            "",
            "The band is compared with external, unverified vendor claims only: OTP or WhatsApp "
            "pre-dispatch confirmation (30–40% RTO reduction), IVR verification (20–30%), "
            "OTP alone (5–10 percentage points on overall RTO), and part-pay COD fees "
            "(60–70% of impulse orders). These anchors are not inputs to the arithmetic.",
            "",
            "![Break-even effectiveness](breakeven.png)",
            "",
        ]),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
