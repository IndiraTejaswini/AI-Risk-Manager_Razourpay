"""Deterministic dry-run replay over the primary risk-set test window."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import PRIMARY_LABEL
from models.calibration import PlattCalibrator
from models.explain import ReasonExplainer
from models.train import predict, prepare_matrix, train
from policy.costs import ACTION_TIERS, c_rto, tier_costs
from policy.effectiveness import EFFECTIVENESS, REASONS, effectiveness_vector
from policy.elkan import apply_policy
from api.service import ScoringService
from responder.templates.registry import ReasonClass

OUT_PATH = ROOT / "eval" / "responder" / "responder_replay.md"


def _fmt(value: float) -> str:
    return f"{value:,.2f}"


def _data():
    service = ScoringService()
    risk = service.loader.risk_set().join(service.loader.split_labelled()["split"], how="left")
    matrix = service.builder.build(risk)
    split = risk["split"].to_numpy()
    train_mask, validation_mask, test_mask = split == "train", split == "validation", split == "test"
    y = risk[PRIMARY_LABEL].astype(int).to_numpy()
    _, levels = prepare_matrix(matrix.loc[train_mask])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[train_mask], pd.Series(y[train_mask]), X.loc[validation_mask],
        pd.Series(y[validation_mask]), target=PRIMARY_LABEL, population="risk_set",
        category_levels=levels,
    )
    validation_scores = predict(bundle, X.loc[validation_mask], raw_score=True)
    validation_ts = risk.loc[validation_mask, "order_purchase_timestamp"]
    fit = (validation_ts >= service.loader.split_boundary - pd.Timedelta(days=30)).to_numpy()
    calibrator = PlattCalibrator().fit(validation_scores[fit], y[validation_mask][fit])
    probabilities = calibrator.predict(predict(bundle, X.loc[test_mask], raw_score=True))
    features = matrix.loc[test_mask].reset_index(drop=True)
    labels = y[test_mask]
    explainer = ReasonExplainer(bundle)
    reasons = explainer.reason_buckets(explainer.explain(X.loc[test_mask]))
    return service, features, labels, reasons, probabilities


def replay() -> dict[str, object]:
    """Return JSON-like deterministic replay results for all test-window orders."""
    _, features, labels, reasons, probabilities = _data()
    policy = apply_policy(probabilities, features, reasons)
    selected = policy.tier.copy()
    policy_to_class = {
        "order_composition": ReasonClass.ORDER_STRUCTURE.value,
        "pincode": ReasonClass.PINCODE_HISTORY.value,
        "customer_history": ReasonClass.CUSTOMER_HISTORY.value,
        "availability": ReasonClass.CATEGORY_SEASONALITY.value,
    }
    policy_reason = np.array([policy_to_class[str(value)] for value in reasons])
    costs = tier_costs(features)
    rto = c_rto(
        np.nan_to_num(features["order_value"].to_numpy(float), nan=0.0),
        np.nan_to_num(features["order_freight"].to_numpy(float), nan=0.0),
    )

    statuses = np.full(len(labels), "held", dtype=object)
    effective = np.zeros(len(labels), dtype=float)
    gate = np.full(len(labels), "held", dtype=object)
    for tier in ACTION_TIERS:
        mask = selected == tier
        e = effectiveness_vector(reasons, tier)
        suppress = mask & (e * rto <= costs[tier]["impression"])
        statuses[mask] = tier
        statuses[suppress] = "suppressed"
        gate[suppress] = "effectiveness_below_impression_cost"
        gate[mask & ~suppress] = "passed"
        effective[mask] = e[mask]

    treated = np.isin(statuses, ACTION_TIERS)
    failures = labels.astype(bool)
    impression = np.zeros(len(labels))
    triggered = np.zeros(len(labels))
    for tier in ACTION_TIERS:
        mask = statuses == tier
        impression[mask] = costs[tier]["impression"][mask]
        triggered[mask] = costs[tier]["triggered"][mask] * (~failures[mask])
    predicted_cost = np.where(
        treated & failures,
        impression + (1.0 - effective) * rto,
        np.where(treated, impression + triggered, np.where(failures, rto, 0.0)),
    )
    realized_cost = np.where(failures, rto, 0.0)

    by_status_reason = Counter((str(statuses[i]), policy_reason[i]) for i in range(len(labels)))
    gate_histogram = Counter(gate)
    selected_failures = int((treated & failures).sum())
    selected_count = int(treated.sum())
    positive_count = int(failures.sum())
    return {
        "rows": len(labels),
        "positive_count": positive_count,
        "base_rate": float(failures.mean()),
        "statuses": dict(sorted(Counter(statuses).items())),
        "by_status_reason": dict(sorted(by_status_reason.items())),
        "gate_histogram": dict(sorted(gate_histogram.items())),
        "impression_cost": float(impression.sum()),
        "triggered_cost": float(triggered.sum()),
        "targeting": {
            "selected": selected_count,
            "selected_failures": selected_failures,
            "precision": selected_failures / selected_count if selected_count else 0.0,
            "recall": selected_failures / positive_count if positive_count else 0.0,
        },
        "predicted_cost": float(predicted_cost.sum()),
        "realized_cost": float(realized_cost.sum()),
        "region_tiers": dict(sorted(Counter(
            str(features.iloc[i].get("customer_state", "unknown"))
            for i in range(len(labels)) if treated[i]
        ).items())),
    }


def render_report(result: dict[str, object]) -> str:
    statuses = result["statuses"]
    targeting = result["targeting"]
    lines = [
        "# Responder dry-run replay — primary Tier-1 test window",
        "",
        "Generated by `responder/replay.py`; all counts use the complete test-window denominator.",
        "",
        f"Orders: **{result['rows']:,}**; observed failures: **{result['positive_count']:,}** "
        f"({result['base_rate']:.2%}).",
        "",
        "| Outcome bucket | Orders |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:,} |" for key, value in statuses.items())
    lines.extend([
        "",
        "## Outcome buckets by tier and reason class",
        "",
        "| Bucket | Reason class | Orders |",
        "|---|---|---:|",
    ])
    lines.extend(
        f"| {bucket} | {reason} | {count:,} |"
        for (bucket, reason), count in result["by_status_reason"].items()
    )
    lines.extend([
        "",
        "## Targeting quality",
        "",
        f"Policy-selected orders: **{targeting['selected']:,}**; failures in selected set: "
        f"**{targeting['selected_failures']:,}**; precision: **{targeting['precision']:.2%}**; "
        f"recall: **{targeting['recall']:.2%}**. Base rate is the full-window rate above.",
        "",
        "## Gate histogram",
        "",
        "| Gate outcome | Orders |",
        "|---|---:|",
    ])
    lines.extend(f"| {key} | {value:,} |" for key, value in result["gate_histogram"].items())
    lines.extend([
        "",
        "## Costs",
        "",
        "Impression and triggered costs are policy assumptions and are reported over all rows.",
        "",
        f"- Impression cost (ASSUMED): BRL {_fmt(result['impression_cost'])}",
        f"- Triggered cost (ASSUMED): BRL {_fmt(result['triggered_cost'])}",
        f"- Predicted cost using effectiveness priors (ASSUMED): BRL {_fmt(result['predicted_cost'])}",
        f"- Realized observed RTO cost without counterfactual uplift (MEASURED): BRL {_fmt(result['realized_cost'])}",
        "",
        "## Intervention rate by region proxy",
        "",
        "| Region value | Treated orders |",
        "|---|---:|",
    ])
    lines.extend(f"| {key} | {value:,} |" for key, value in result["region_tiers"].items())
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    result = replay()
    OUT_PATH.write_text(render_report(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
