#!/usr/bin/env python3
"""Generate the assumption-free policy comparison table."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.service import ScoringService
from models.calibration import PlattCalibrator
from models.train import predict, prepare_matrix, train
from policy.constants import CURRENCY
from policy.costs import ACTION_TIERS, c_rto, ltv_multiplier, tier_costs
from policy.effectiveness import effectiveness_vector
from policy.elkan import apply_policy
from models.explain import ReasonExplainer

OUT_PATH = REPO_ROOT / "eval" / "policy_table.md"
HAND_ARTIFACT = REPO_ROOT / "eval" / "policy.md"
BOOT = 2_000
SEED = 20260901


def _hand_threshold() -> float:
    text = HAND_ARTIFACT.read_text(encoding="utf-8")
    match = re.search(r"`order_value > ([\d,]+\.\d+) BRL AND customer", text)
    if not match:
        raise ValueError(f"could not find hand-rule threshold in {HAND_ARTIFACT}")
    return float(match.group(1).replace(",", ""))


def _policy_cost_vector(p, y, features, reasons, tier_choice):
    value = np.nan_to_num(features["order_value"].to_numpy(dtype=float), nan=0.0)
    freight = np.nan_to_num(features["order_freight"].to_numpy(dtype=float), nan=0.0)
    rto = c_rto(value, freight)
    costs = tier_costs(features)
    y = np.asarray(y).astype(bool)
    impression = np.zeros(len(y))
    triggered = np.zeros(len(y))
    cost = np.where(y, rto, 0.0).astype(float)
    for tier in ACTION_TIERS:
        selected = tier_choice == tier
        if not selected.any():
            continue
        impression[selected] = costs[tier]["impression"][selected]
        triggered[selected] = costs[tier]["triggered"][selected]
        e = effectiveness_vector(reasons, tier)
        fp = costs[tier]["c_fp"]
        cost[selected & y] = impression[selected & y] + (1.0 - e[selected & y]) * rto[selected & y]
        cost[selected & ~y] = fp[selected & ~y]
    treated = tier_choice != "allow"
    return cost, treated, impression, triggered * np.where(~y, 1.0, 0.0)


def _row_data():
    svc = ScoringService()
    risk = svc.loader.risk_set().join(svc.loader.split_labelled()["split"], how="left")
    matrix = svc.builder.build(risk)
    split = risk["split"].to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"
    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    y_all = risk["label_b"].astype(int).to_numpy()
    bundle = train(
        X.loc[tr], pd.Series(y_all[tr]), X.loc[va], pd.Series(y_all[va]),
        target="label_b", population="risk_set", category_levels=levels,
    )
    validation_ts = risk.loc[va, "order_purchase_timestamp"]
    fit_mask = (
        validation_ts >= svc.loader.split_boundary - pd.Timedelta(days=30)
    ).to_numpy()
    calibrator = PlattCalibrator().fit(
        predict(bundle, X.loc[va], raw_score=True)[fit_mask],
        y_all[va][fit_mask],
    )
    p = calibrator.predict(predict(bundle, X.loc[te], raw_score=True))
    feats = matrix.loc[te].reset_index(drop=True)
    y = y_all[te]
    explainer = ReasonExplainer(bundle)
    reasons = explainer.reason_buckets(explainer.explain(X.loc[te]))
    result = apply_policy(p, feats, reasons)

    hand_cut = _hand_threshold()
    value = np.nan_to_num(feats["order_value"].to_numpy(dtype=float), nan=0.0)
    new_customer = np.nan_to_num(feats["cust_prior_orders"], nan=0.0) == 0
    hand_tier = np.where((value > hand_cut) & new_customer, "prepaid_only", "allow")
    everything_tier = np.full(len(y), "prepaid_only")
    return (
        y, feats, reasons,
        {
            "Intervene on nothing": np.full(len(y), "allow"),
            "Intervene on everything": everything_tier,
            "Hand-written rule": hand_tier,
            "Model + cost policy": result.tier,
        },
    )


def _row_metrics(y, features, reasons, choices, rto_scale=1.0):
    value = np.nan_to_num(features["order_value"].to_numpy(dtype=float), nan=0.0)
    freight = np.nan_to_num(features["order_freight"].to_numpy(dtype=float), nan=0.0)
    rto = c_rto(value, freight) * rto_scale
    costs = tier_costs(features)
    ltv = ltv_multiplier(
        features["cust_prior_orders"].to_numpy(dtype=float)
    )
    treated = choices != "allow"
    impression = np.zeros(len(y))
    triggered = np.zeros(len(y))
    realised = np.where(y.astype(bool), rto, 0.0)
    for tier in ACTION_TIERS:
        mask = choices == tier
        if not mask.any():
            continue
        impression[mask] = costs[tier]["impression"][mask]
        triggered[mask] = (
            costs[tier]["triggered"][mask]
            * ltv[mask]
            * np.where(~y[mask].astype(bool), 1.0, 0.0)
        )
        e = effectiveness_vector(reasons, tier)
        realised[mask & y.astype(bool)] = (
            impression[mask & y.astype(bool)]
            + (1.0 - e[mask & y.astype(bool)]) * rto[mask & y.astype(bool)]
        )
        realised[mask & ~y.astype(bool)] = (
            costs[tier]["impression"][mask & ~y.astype(bool)]
            + costs[tier]["triggered"][mask & ~y.astype(bool)]
        )
    denominator = np.where(y.astype(bool) & treated, rto, 0.0)
    return treated, y.astype(bool), impression, triggered, realised, denominator


def _fmt(value: float) -> str:
    return f"{value:,.2f}"


def main() -> int:
    y, features, reasons, policies = _row_data()
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(y), size=(BOOT, len(y)))
    rows = []
    for name, choices in policies.items():
        treated, failed, impression, triggered, realised, denominator = _row_metrics(
            y, features, reasons, choices
        )
        n_treated = int(treated.sum())
        n_failures = int((treated & failed).sum())
        imp = float(impression.sum())
        trig = float(triggered.sum())
        bands = []
        for scale in (0.5, 1.5):
            _, _, _, _, _, denom_scaled = _row_metrics(
                y, features, reasons, choices, rto_scale=scale
            )
            numerator = impression + triggered
            values = numerator[indices].sum(axis=1) / np.maximum(
                denom_scaled[indices].sum(axis=1), 1e-12
            )
            bands.append(float(np.median(values)))
        boot = (impression[indices].sum(axis=1) + triggered[indices].sum(axis=1)) / np.maximum(
            denominator[indices].sum(axis=1), 1e-12
        )
        rows.append(
            f"| {name} | {n_treated:,} | {n_failures:,} | {_fmt(imp)} | "
            f"{_fmt(trig)} | {min(bands) * 100:.2f}%–{max(bands) * 100:.2f}% | "
            f"[{np.percentile(boot, 2.5) * 100:.2f}%, "
            f"{np.percentile(boot, 97.5) * 100:.2f}%] |"
        )

    text = "\n".join([
        "# Policy table — measured costs and break-even effectiveness",
        "",
        "Generated by `eval/policy_table.py` from the risk-set test window. "
        "The hand-written rule is read from the Step 10 `eval/policy.md` artifact; "
        "its threshold is not re-fit here.",
        "",
        "> **Caption.** The deliberately absent columns are **RTO cost avoided** and "
        "**Net**. Both require multiplying by an effectiveness prior, which would turn "
        "this measured table into an assumed one. Their absence is the argument. "
        "Break-even effectiveness is the measured fraction of treated failures that "
        "must be prevented to cover impression plus expected triggered cost. The "
        "reported band sweeps `c_rto` from 50% to 150% of its stated value; the CI is "
        "a paired percentile bootstrap over the identical test rows.",
        "",
        "| Policy | Orders treated | Failures in treated set | Impression cost | "
        "Expected triggered cost | Break-even effectiveness | CI |",
        "|---|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
    ])
    OUT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
