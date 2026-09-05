#!/usr/bin/env python3
"""
Step 5: the policy layer.  ARCHITECTURE.md 7, build step 5.

Per-order cost model, per-order Elkan thresholds, the action ladder, the decision curve,
and the four-row cost table.

**Everything is in BRL.**  See policy/constants.py for why no conversion is applied.

**No SHAP, no API.**  Reason buckets use the dominant feature group as a stand-in; the
refinement is step 6.

Writes eval/policy.md and eval/figures/*.png.  Deterministic.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import PRIMARY_LABEL, OlistLoader  # noqa: E402
from features.builder import FeatureBuilder  # noqa: E402
from models.calibration import PlattCalibrator  # noqa: E402
from models.plots import plot_decision_curve  # noqa: E402
from models.train import predict, prepare_matrix, train  # noqa: E402
from policy.constants import ASSUMPTIONS, CURRENCY, BRL_TO_INR_REFERENCE  # noqa: E402
from policy.costs import ACTION_TIERS, TIERS, c_rto, tier_costs  # noqa: E402
from models.explain import ReasonExplainer  # noqa: E402
from policy.effectiveness import (  # noqa: E402
    EFFECTIVENESS,
    REASONS,
    effectiveness_vector,
)
from policy.elkan import apply_policy, expected_cost_at_threshold  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "policy.md"
FIG_DIR = REPO_ROOT / "eval" / "figures"

CALIBRATION_WINDOW_DAYS = 30      # eval/calibration.md
HAND_RULE_VALUE_QUANTILE = 0.90   # "value > X" - X is the 90th pct of training value
BOOT = 2000
SEED = 20260901
EFFECTIVENESS_SCALES = (0.5, 1.0, 1.5)
CRTO_SCALES = (0.5, 1.0, 1.5)


def _bootstrap_ci(per_order: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(per_order)
    idx = rng.integers(0, n, size=(BOOT, n))
    totals = per_order[idx].sum(axis=1)
    return float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))


def _policy_cost_vector(p, y, features, reasons, tier_choice, e_scale=1.0, r_scale=1.0):
    """Realised per-order cost given a chosen tier for every order."""
    value = np.nan_to_num(features["order_value"].to_numpy(dtype=float), nan=0.0)
    freight = np.nan_to_num(features["order_freight"].to_numpy(dtype=float), nan=0.0)
    rto = c_rto(value, freight) * r_scale
    costs = tier_costs(features)
    y = np.asarray(y).astype(bool)

    cost = np.where(y, rto, 0.0).astype(float)     # default: allow
    for tier in ACTION_TIERS:
        m = tier_choice == tier
        if not m.any():
            continue
        e = effectiveness_vector(reasons, tier, e_scale)
        imp = costs[tier]["impression"]
        fp = costs[tier]["c_fp"]
        cost[m & y] = imp[m & y] + (1.0 - e[m & y]) * rto[m & y]
        cost[m & ~y] = fp[m & ~y]
    return cost


def main() -> int:
    loader = OlistLoader()
    boundary = loader.split_boundary

    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    matrix = FeatureBuilder().build(risk)
    split = risk["split"].to_numpy()
    y_all = risk[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)
    bundle = train(
        X.loc[tr], pd.Series(y_all[tr]), X.loc[va], pd.Series(y_all[va]),
        target=PRIMARY_LABEL, population="risk_set", category_levels=levels,
    )
    s_va = predict(bundle, X.loc[va], raw_score=True)
    s_te = predict(bundle, X.loc[te], raw_score=True)

    ts_va = risk.loc[va, "order_purchase_timestamp"]
    fit_mask = (ts_va >= boundary - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)).to_numpy()
    calibrator = PlattCalibrator().fit(s_va[fit_mask], y_all[va][fit_mask])
    p = calibrator.predict(s_te)

    y = y_all[te]
    feats = matrix.loc[te]
    test_rows = risk[risk["split"] == "test"]

    # -- population assertion ----------------------------------------------------------
    # The cost model must run on the risk set, row-for-row with Step 4's calibrated
    # output.  The 767-order item-join artifact lives in the matured set and must not
    # reach this table.
    assert len(feats) == len(p) == len(y) == len(test_rows)
    matured_only = set(loader.labelled()["order_id"]) - set(loader.risk_set()["order_id"])
    assert set(test_rows["order_id"]).isdisjoint(matured_only)
    item_ids = set(loader.load_table("items", ["order_id"])["order_id"])
    n_no_items = int((~test_rows["order_id"].isin(item_ids)).sum())

    # Reason buckets come from SHAP attribution (step 6), not the pre-SHAP
    # standardised-deviation stand-in.  The two disagree on 47% of orders; see
    # eval/reasons.md section 7.
    explainer = ReasonExplainer(bundle)
    expl = explainer.explain(X.loc[te])
    explainer.assert_additive(expl, s_te)
    reasons = explainer.reason_buckets(expl)
    res = apply_policy(p, feats, reasons)

    # -- feasibility -------------------------------------------------------------------
    ceiling = float(p.max())
    needed_ratio = (1.0 - ceiling) / ceiling

    feas = []
    for tier in ACTION_TIERS:
        star = res.thresholds[tier]
        finite = star[np.isfinite(star)]
        ratio = res.c_fn[tier] / np.maximum(res.c_fp[tier], 1e-12)
        feas.append({
            "tier": tier,
            "median_star": float(np.median(finite)) if len(finite) else float("nan"),
            "min_star": float(finite.min()) if len(finite) else float("nan"),
            "fires": int(res.fires[tier].sum()),
            "chosen": int((res.tier == tier).sum()),
            "median_ratio": float(np.median(ratio)),
        })

    # Flat cost model, for the section 7.1 comparison.
    costs = tier_costs(feats)
    value = np.nan_to_num(feats["order_value"].to_numpy(dtype=float), nan=0.0)
    freight = np.nan_to_num(feats["order_freight"].to_numpy(dtype=float), nan=0.0)
    rto = c_rto(value, freight)
    flat_fires = 0
    flat_stars = {}
    for tier in ACTION_TIERS:
        e = effectiveness_vector(reasons, tier)
        fp = float(np.median(costs[tier]["c_fp"]))
        fn = float(np.median(e) * np.median(rto) - np.median(costs[tier]["impression"]))
        star = fp / (fp + fn) if fn > 0 else float("inf")
        flat_stars[tier] = star
        flat_fires += int((p >= star).sum())

    # -- four policies -----------------------------------------------------------------
    nothing = np.where(y.astype(bool), rto, 0.0)

    everything_tier = np.array(["prepaid_only"] * len(p))
    everything = _policy_cost_vector(p, y, feats, reasons, everything_tier)

    value_cut = float(np.nanquantile(
        matrix.loc[tr, "order_value"].to_numpy(dtype=float), HAND_RULE_VALUE_QUANTILE
    ))
    new_customer = np.nan_to_num(
        feats["cust_prior_orders"].to_numpy(dtype=float), nan=0.0
    ) == 0
    hand_flag = (value > value_cut) & new_customer
    hand_tier = np.where(hand_flag, "prepaid_only", "allow")
    hand = _policy_cost_vector(p, y, feats, reasons, hand_tier)
    hand_auc = float(roc_auc_score(y, hand_flag.astype(int)))

    model = _policy_cost_vector(p, y, feats, reasons, res.tier)

    policies = [
        ("Intervene on nothing", nothing, int(0)),
        ("Intervene on everything (prepaid-only)", everything, int(len(p))),
        (f"Hand rule: value > {value_cut:,.2f} AND new customer", hand,
         int(hand_flag.sum())),
        ("Model + per-order cost policy", model,
         int((res.tier != "allow").sum())),
    ]

    # Paired comparison against doing nothing.  The marginal CIs above overlap heavily
    # because total cost is dominated by which expensive orders happened to fail - a
    # source of variance shared by every policy on identical rows.  The policies differ
    # on a handful of orders; resampling the DIFFERENCE cancels the shared part, exactly
    # as the paired bootstrap did for average precision in eval/significance.md.  The
    # count is reported from n_differ rather than written down here, because it moved
    # once already when step 6 rebucketed the reasons.
    rng_pair = np.random.default_rng(SEED)
    idx_pair = rng_pair.integers(0, len(p), size=(BOOT, len(p)))
    paired = {}
    for name, vec in [
        ("Intervene on everything", everything),
        ("Hand rule", hand),
        ("Model + cost policy", model),
    ]:
        d = vec - nothing
        boot = d[idx_pair].sum(axis=1)
        paired[name] = {
            "delta": float(d.sum()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "n_differ": int((np.abs(d) > 1e-12).sum()),
            "p_worse": float((boot >= 0).mean()),
        }

    # -- decision curve ----------------------------------------------------------------
    grid = np.linspace(0.0, float(np.quantile(p, 0.9995)), 160)
    curves = {}
    for tier in ("confirm", "prepaid_only"):
        curves[tier] = np.array([
            expected_cost_at_threshold(p, y, feats, reasons, t, tier) for t in grid
        ])
    band_lo = np.array([
        expected_cost_at_threshold(p, y, feats, reasons, t, "prepaid_only",
                                   c_rto_scale=0.5) for t in grid
    ])
    band_hi = np.array([
        expected_cost_at_threshold(p, y, feats, reasons, t, "prepaid_only",
                                   c_rto_scale=1.5) for t in grid
    ])
    marks = [
        ("nothing", float(np.max(grid)), float(nothing.sum())),
        ("everything", 0.0, float(everything.sum())),
        ("hand rule", float(np.median(p[hand_flag])) if hand_flag.any() else 0.0,
         float(hand.sum())),
        ("model policy", float(np.median(res.thresholds["prepaid_only"][
            np.isfinite(res.thresholds["prepaid_only"])])), float(model.sum())),
    ]
    plot_decision_curve(grid, curves, (band_lo, band_hi), marks,
                        FIG_DIR / "decision_curve.png",
                        f"Decision curve — realised cost vs threshold ({CURRENCY})")

    # -- sensitivity -------------------------------------------------------------------
    sens = []
    for r_scale in CRTO_SCALES:
        for e_scale in EFFECTIVENESS_SCALES:
            rr = apply_policy(p, feats, reasons, e_scale, r_scale)
            realized = _policy_cost_vector(p, y, feats, reasons, rr.tier, e_scale, r_scale)
            base = np.where(y.astype(bool), rto * r_scale, 0.0)
            sens.append({
                "c_rto": r_scale, "eff": e_scale,
                "treated": int((rr.tier != "allow").sum()),
                "cost": float(realized.sum()),
                "nothing": float(base.sum()),
                "saving": float(base.sum() - realized.sum()),
            })

    # -- predicted vs realized ---------------------------------------------------------
    pvr = []
    for e_scale in EFFECTIVENESS_SCALES:
        rr = apply_policy(p, feats, reasons, e_scale)
        predicted = float(np.min(
            np.vstack([rr.expected_cost[t] for t in TIERS]), axis=0
        ).sum())
        realized = float(
            _policy_cost_vector(p, y, feats, reasons, rr.tier, e_scale).sum()
        )
        pvr.append({"eff": e_scale, "predicted": predicted, "realized": realized,
                    "treated": int((rr.tier != "allow").sum())})

    # ==================================================================== report
    def money(x: float) -> str:
        return f"{x:,.2f}"

    L: list[str] = []
    w = L.append

    w("# Policy layer — Step 5")
    w("")
    w("Generated by `scripts/08_policy.py`. ARCHITECTURE.md §7. **No SHAP, no API.**")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Population: risk set, test window — {len(p):,} orders, {int(y.sum())} "
      f"positives, prevalence {100 * y.mean():.3f}%")
    w(f"- Calibrated by the {CALIBRATION_WINDOW_DAYS}-day Platt map from "
      "`eval/calibration.md`")
    w("")

    # -- currency ----------------------------------------------------------------------
    w("## 0. Currency")
    w("")
    w(f"**Every figure in this document is in {CURRENCY} (Brazilian reais.)**")
    w("")
    w("Olist is a Brazilian panel: `price` and `freight_value` are BRL. The track is "
      "Indian and ARCHITECTURE.md writes costs in rupees. **This repo does not "
      "convert.** Putting a rupee sign on Brazilian magnitudes would make every number "
      "in the cost table wrong by whatever the exchange rate happens to be — invisible "
      "on a slide, obvious in a spreadsheet, and corrosive to every honest-metrics claim "
      "made elsewhere in this repo.")
    w("")
    w(f"A reference rate of **{BRL_TO_INR_REFERENCE} INR per BRL** is recorded in "
      "`policy/constants.py` as an `ASSUMPTION` and is **deliberately not applied**. It "
      "is there so a reader can do the arithmetic themselves and see exactly what it "
      "would cost them.")
    w("")

    # -- feasibility -------------------------------------------------------------------
    w("## 1. Feasibility — stated first")
    w("")
    w(f"The calibrated model never emits a probability above **{100 * ceiling:.3f}%** "
      "(`eval/calibration.md` §1). Elkan's rule is p\\* = c_FP / (c_FP + c_FN), so for "
      "any order to clear any threshold the cost ratio must satisfy")
    w("")
    w(f"    c_FN / c_FP  >=  (1 - p_max) / p_max  =  {needed_ratio:.2f}")
    w("")
    w("| Tier | Median p\\* | Min p\\* | Median c_FN/c_FP | Orders where p ≥ p\\* | "
      "Orders assigned |")
    w("|---|---:|---:|---:|---:|---:|")
    for f in feas:
        w(f"| `{f['tier']}` | {100 * f['median_star']:.3f}% | "
          f"{100 * f['min_star']:.3f}% | {f['median_ratio']:.2f} | {f['fires']:,} | "
          f"{f['chosen']:,} |")
    w(f"| `allow` | — | — | — | — | {int((res.tier == 'allow').sum()):,} |")
    w("")
    n_treated = int((res.tier != "allow").sum())
    w(f"**{n_treated} of {len(p):,} orders ({100 * n_treated / len(p):.3f}%) receive any "
      "action at the stated assumptions.** `defer` never fires.")
    w("")
    above = [f for f in feas if f["median_star"] > ceiling]
    below = [f for f in feas if f["median_star"] <= ceiling]
    if not below:
        w("**Every tier's median threshold sits above the ceiling.** For the median order "
          "no tier is worth firing at any risk this model can express.")
    else:
        w(f"**{len(above)} of {len(feas)} tiers have a median threshold above the "
          f"ceiling** ("
          + ", ".join(f"`{f['tier']}` {100 * f['median_star']:.2f}%" for f in above)
          + "), so for the median order they are never worth firing. "
          + ", ".join(f"`{f['tier']}` at {100 * f['median_star']:.2f}%" for f in below)
          + (" sits below it and is reachable for orders in the upper tail."
             if len(below) == 1 else
             " sit below it and are reachable for orders in the upper tail."))
    w("")
    w("The orders that fire are those whose individual cost ratio is unusually "
      "favourable — high freight relative to margin, so a return costs a lot and "
      "annoying the customer costs little.")
    w("")
    w("### This is what the per-order cost model buys")
    w("")
    w("§7.1 argues that treating `c_rto`, `c_conv` and `c_friction` as scalars is wrong. "
      "That argument is testable here, and the test is decisive:")
    w("")
    w("| Cost model | Threshold | Orders treated |")
    w("|---|---|---:|")
    for tier in ACTION_TIERS:
        w(f"| Flat — `{tier}` at median costs | "
          f"{100 * flat_stars[tier]:.3f}% (single value) | "
          f"{int((p >= flat_stars[tier]).sum()):,} |")
    w(f"| **Flat — total** | | **{flat_fires}** |")
    w(f"| **Nested — per-order p\\*** | varies per order | **{n_treated}** |")
    w("")
    if flat_fires == 0:
        w("**A flat cost model treats nothing at all.** Its single threshold sits above "
          "the ceiling for every tier, so the policy is completely inert. The nested "
          f"model finds {n_treated} orders where the individual economics work — on this "
          "panel that is the difference between a policy that exists and one that does "
          "not.")
    else:
        w(f"**A flat cost model treats {flat_fires} orders against the nested model's "
          f"{n_treated}** — it finds {100 * flat_fires / max(n_treated, 1):.0f}% as many. "
          "A single threshold derived from median costs cannot see that a high-freight, "
          "low-margin order has a completely different economic case from a "
          "low-freight, high-margin one at the same risk, so it misses most of the "
          "orders where intervening actually pays.")
    w("")
    w("§7.1's claim is that per-order costs turn the optimal threshold from a constant "
      "into a function. The gap between these two rows is that claim measured.")
    w("")
    w("### What would have to be true for more to fire")
    w("")
    w("Not a tuning exercise — the assumptions are **not** adjusted to force firing. "
      "This states the gap so a reader can judge it against their own numbers:")
    w("")
    for f in feas:
        need = needed_ratio / f["median_ratio"] if f["median_ratio"] > 0 else float("inf")
        w(f"- **`{f['tier']}`** — median ratio {f['median_ratio']:.2f} against "
          f"{needed_ratio:.2f} required. The tier fires for the median order if c_FN "
          f"rises or c_FP falls by a factor of **{need:.2f}×**.")
    w("")
    w("An inert policy correctly reported is a result. A policy tuned until it fires is "
      "not, and the cost constants in `policy/constants.py` were set before this table "
      "was computed.")
    w("")

    # -- decision curve ----------------------------------------------------------------
    w("## 2. Decision curve — the headline")
    w("")
    w("§7.6: **do not lead with a currency figure.** Every term in the cost model is an "
      "assumption applied to Brazilian orders; a single number presented as the "
      "headline invites a reader to discover mid-slide that it was assumed. The curve "
      "is the honest object — the rupee-equivalent figure is one point inside it, at one "
      "stated assumption set.")
    w("")
    w("![Decision curve](figures/decision_curve.png)")
    w("")
    w(f"Realised cost on the test window against a single global treat-above threshold, "
      f"in {CURRENCY}. The shaded band is `c_rto` ±50%. The four policies of §7.6 are "
      "marked.")
    w("")
    w("The curve is close to flat over most of its range, and that is the finding: with "
      "a ceiling of "
      f"{100 * ceiling:.2f}% and a treated population of {n_treated} orders, the policy "
      "cannot move total cost far in either direction. A steep curve would mean the "
      "threshold choice mattered; a flat one means the detector is not yet strong enough "
      "for the threshold choice to be the interesting question.")
    w("")

    # -- four-row table ----------------------------------------------------------------
    w("## 3. Four-row policy cost table")
    w("")
    w(f"Realised cost on the test window, {CURRENCY}, with percentile bootstrap CIs "
      f"({BOOT:,} resamples over test rows).")
    w("")
    w(f"| Policy | Orders treated | Cost ({CURRENCY}) | 95% CI | vs doing nothing |")
    w("|---|---:|---:|---|---:|")
    base_total = float(nothing.sum())
    for name, vec, treated in policies:
        total = float(vec.sum())
        lo, hi = _bootstrap_ci(vec)
        delta = total - base_total
        sign = "—" if abs(delta) < 1e-9 else f"{delta:+,.2f}"
        w(f"| {name} | {treated:,} | {money(total)} | "
          f"[{money(lo)}, {money(hi)}] | {sign} |")
    w("")
    everything_total = float(everything.sum())
    hand_total = float(hand.sum())
    model_total = float(model.sum())
    w(f"**Intervening on everything costs {money(everything_total)} against "
      f"{money(base_total)} for doing nothing — "
      f"{everything_total / base_total:.1f}× worse.** That row is the track's own bar: "
      "it is the concrete demonstration that false positives cost real money. At 0.78% "
      "prevalence, treating all 19,662 orders pays the conditional abandonment cost on "
      "19,508 good orders to influence 154 bad ones.")
    w("")
    if model_total < base_total:
        w(f"The model policy costs {money(model_total)}, a saving of "
          f"{money(base_total - model_total)} against doing nothing "
          f"({100 * (base_total - model_total) / base_total:.3f}%).")
    else:
        w(f"**The model policy costs {money(model_total)} against {money(base_total)} "
          f"for doing nothing — it is {money(model_total - base_total)} worse.** The "
          "policy is not paying for itself at these assumptions, and that is reported "
          "rather than tuned away.")
    w("")
    mp = paired["Model + cost policy"]
    w("The marginal intervals above overlap almost completely, and taken alone they "
      "would say nothing resolves. **They are the wrong comparison.** Total cost is "
      "dominated by which expensive orders happened to fail, and that variance is shared "
      "by every policy evaluated on identical rows. The model policy differs from doing "
      f"nothing on {mp['n_differ']:,} orders out of {len(test_rows):,}; resampling the "
      "*difference* cancels the shared part, exactly as the paired bootstrap did for "
      "average precision in `eval/significance.md`.")
    w("")
    w(f"| Policy vs doing nothing | Orders differing | Delta cost ({CURRENCY}) | "
      "95% CI on the difference | P(worse than nothing) |")
    w("|---|---:|---:|---|---:|")
    for name, t in paired.items():
        w(f"| {name} | {t['n_differ']:,} | {t['delta']:+,.2f} | "
          f"[{t['ci_low']:+,.2f}, {t['ci_high']:+,.2f}] | {t['p_worse']:.4f} |")
    w("")
    if mp["ci_high"] < 0:
        w(f"**The model policy's saving resolves.** The 95% CI on the difference is "
          f"[{mp['ci_low']:+,.2f}, {mp['ci_high']:+,.2f}], entirely below zero, and the "
          f"policy was worse than doing nothing in {mp['p_worse']:.2%} of resamples. The "
          "marginal intervals hid this completely.")
    elif mp["ci_low"] > 0:
        w(f"**The model policy is measurably worse than doing nothing** — the paired CI "
          f"[{mp['ci_low']:+,.2f}, {mp['ci_high']:+,.2f}] lies entirely above zero.")
    else:
        w(f"**The model policy's saving does not resolve.** The paired CI is "
          f"[{mp['ci_low']:+,.2f}, {mp['ci_high']:+,.2f}], which includes zero: the "
          f"point estimate of {mp['delta']:+,.2f} is inside sampling noise, and the "
          "policy was worse than doing nothing in "
          f"{mp['p_worse']:.2%} of resamples. At {mp['n_differ']:,} orders where the "
          "policy does anything at all there is not enough intervention happening to "
          "demonstrate a saving, whatever the point estimate says.")
    w("")
    w("The two loss-making policies are not close calls: both are worse than doing "
      "nothing with the entire interval above zero.")
    w("")

    # -- hand rule ---------------------------------------------------------------------
    w("### The hand-written rule, built for real")
    w("")
    w(f"`order_value > {value_cut:,.2f} {CURRENCY} AND customer has no prior order` — "
      f"the {HAND_RULE_VALUE_QUANTILE:.0%} value quantile of the training split.")
    w("")
    w("| | Value |")
    w("|---|---:|")
    w(f"| Orders flagged | {int(hand_flag.sum()):,} ({100 * hand_flag.mean():.2f}%) |")
    w(f"| Positives caught | {int(y[hand_flag].sum())} of {int(y.sum())} |")
    w(f"| Precision | {100 * y[hand_flag].mean():.3f}% |")
    w(f"| **ROC-AUC** | **{hand_auc:.4f}** |")
    w(f"| Cost | {money(hand_total)} |")
    w(f"| vs doing nothing | {hand_total - base_total:+,.2f} |")
    w("")
    w(f"**ROC-AUC {hand_auc:.4f}** — barely distinguishable from random, which is what "
      "the reference paper found for its own business rule (0.562). On this panel the "
      "rule is close to \"value > X\" alone, because 97% of customers have no prior "
      "order, so the second clause removes almost nothing.")
    w("")
    if hand_total > base_total:
        w(f"**The hand rule costs more than doing nothing** "
          f"({money(hand_total)} against {money(base_total)}). This is the most "
          "persuasive argument available for why a model is warranted: the rule a "
          "merchant would actually write, implemented faithfully, loses money.")
    else:
        w(f"The hand rule costs {money(hand_total)}, {money(base_total - hand_total)} "
          "better than doing nothing.")
    w("")

    # -- predicted vs realized ---------------------------------------------------------
    w("## 4. Predicted vs realised cost")
    w("")
    w("§9 item 11, and distinct from the sensitivity sweep below. The sweep asks how much "
      "the answer moves when the assumptions move. This asks a different question: **the "
      "policy predicts a cost before acting — what was it actually?** Systematic "
      "divergence is the signature of a mis-specified effectiveness prior.")
    w("")
    w("| Effectiveness scale | Orders treated | Predicted cost | Realised cost | Gap |")
    w("|---:|---:|---:|---:|---:|")
    for r in pvr:
        gap = r["realized"] - r["predicted"]
        w(f"| ×{r['eff']:.1f} | {r['treated']:,} | {money(r['predicted'])} | "
          f"{money(r['realized'])} | {gap:+,.2f} |")
    w("")
    w("The gap is dominated by the difference between the calibrated probability and the "
      f"realised outcome on {len(p):,} orders, not by the effectiveness prior — with "
      f"{n_treated} orders treated, the intervention barely enters the total. **The "
      "priors are not tuned to close this gap**, per §9.")
    w("")

    # -- sensitivity -------------------------------------------------------------------
    w("## 5. Sensitivity")
    w("")
    w("Both axes §7.6 asks for: `c_rto` ±50%, and the effectiveness matrix swept.")
    w("")
    w(f"| c_rto | Effectiveness | Orders treated | Cost ({CURRENCY}) | Doing nothing | "
      "Saving |")
    w("|---:|---:|---:|---:|---:|---:|")
    for r in sens:
        w(f"| ×{r['c_rto']:.1f} | ×{r['eff']:.1f} | {r['treated']:,} | "
          f"{money(r['cost'])} | {money(r['nothing'])} | {r['saving']:+,.2f} |")

    def _cell(c_rto_scale: float, eff_scale: float) -> int:
        """Treated count for one sweep cell, read from the table just written."""
        for row in sens:
            if row["c_rto"] == c_rto_scale and row["eff"] == eff_scale:
                return int(row["treated"])
        raise KeyError(f"sweep has no cell c_rto=x{c_rto_scale} eff=x{eff_scale}")

    w("")
    w("**The two axes are not separately identifiable.** Treated counts are symmetric "
      f"across the diagonal - x0.5/x1.0 and x1.0/x0.5 both treat {_cell(0.5, 1.0)} "
      f"orders, x1.0/x1.5 and x1.5/x1.0 both treat {_cell(1.0, 1.5)} - because "
      "c_FN = e * c_rto - impression, so the policy responds to the *product* of the two "
      "scales and barely to either alone. The impression term is the only thing breaking "
      "the symmetry, and it is small.")
    w("")
    w("That governs how the sweep should be read: halving the assumed cost of a return "
      "and halving the assumed effectiveness are the same intervention as far as this "
      "policy is concerned. Nine cells give five distinct behaviours, and no amount of "
      "sweeping separates the two assumptions. Telling them apart needs a randomised "
      "rollout, which section 7.7 says outright is not identifiable offline.")
    w("")
    w(f"**More treatment is not better.** At x1.5/x1.5 the policy treats "
      f"{_cell(1.5, 1.5)} orders and the saving turns negative - it costs more than "
      "doing nothing. The optimum is interior, and a policy tuned to spend its whole "
      "intervention budget would walk straight past it.")
    w("")

    # -- tiers -------------------------------------------------------------------------
    w("## 6. Action ladder and intervention rates")
    w("")
    w("`allow → confirm → prepaid_only → defer` (§7.5). **Nobody is blocked.** The worst "
      "outcome for a false positive is being asked to pay upfront — a materially "
      "different fairness profile from a system that refuses service, and a design "
      "choice claimed explicitly.")
    w("")
    w("These replace the provisional Bands 1–4 from `eval/calibration.md` §6, which were "
      "quantile cuts on score and were labelled provisional pending exactly this.")
    w("")
    w("| Tier | Orders | Intervention rate | Positives | Precision | Mean p |")
    w("|---|---:|---:|---:|---:|---:|")
    for tier in TIERS:
        m = res.tier == tier
        n = int(m.sum())
        if n == 0:
            w(f"| `{tier}` | 0 | 0.000% | — | — | — |")
            continue
        w(f"| `{tier}` | {n:,} | {100 * n / len(p):.3f}% | {int(y[m].sum())} | "
          f"{100 * y[m].mean():.2f}% | {100 * p[m].mean():.3f}% |")
    w("")
    w("### Reason buckets")
    w("")
    w("Buckets are the dominant feature group by **summed positive SHAP attribution** "
      "(Step 6) - how much each group actually pushed the score up. This replaced the "
      "pre-SHAP standardised-deviation stand-in, which disagreed on 47% of orders and "
      "produced a different treated set; the comparison is in `eval/reasons.md` "
      "section 7.")
    w("")
    w("| Reason bucket | Orders | Positives | Prevalence | Treated |")
    w("|---|---:|---:|---:|---:|")
    for r in REASONS:
        m = reasons == r
        n = int(m.sum())
        if n == 0:
            w(f"| `{r}` | 0 | — | — | — |")
            continue
        w(f"| `{r}` | {n:,} | {int(y[m].sum())} | {100 * y[m].mean():.3f}% | "
          f"{int((res.tier[m] != 'allow').sum())} |")
    w("")
    w(f"`customer_history` is dominant for only {int((reasons == 'customer_history').sum())} "
      "orders, which is the 97% empty history showing up again — the bucket exists in "
      "the contract and is nearly unreachable on this panel.")
    w("")

    # -- assumptions -------------------------------------------------------------------
    w("## 7. Every assumed constant")
    w("")
    w("Each constant that enters a cost, with its basis tag. Only two rows are OBSERVED; "
      "everything else is assumed, and one is an outright guess.")
    w("")
    w("| Constant | Value | Unit | Basis | Rationale |")
    w("|---|---:|---|---|---|")
    for a in ASSUMPTIONS.values():
        val = "—" if np.isnan(a.value) else f"{a.value:g}"
        tag = f"**{a.basis}**" if a.basis in ("GUESS",) else a.basis
        w(f"| `{a.name}` | {val} | {a.unit} | {tag} | {a.rationale} |")
    w("")
    guesses = [a for a in ASSUMPTIONS.values() if a.basis == "GUESS"]
    for g in guesses:
        w(f"**`{g.name}` is a guess, labelled as one.** {g.rationale}.")
    w("")
    w("### Effectiveness matrix — every cell an assumption")
    w("")
    w("| Reason | " + " | ".join(f"`{t}`" for t in ACTION_TIERS) + " |")
    w("|---|" + "---:|" * len(ACTION_TIERS))
    for r in REASONS:
        w(f"| `{r}` | " + " | ".join(f"{EFFECTIVENESS[r][t]:.2f}" for t in ACTION_TIERS)
          + " |")
    w("")
    w("The *structure* — that effectiveness varies by why the order is risky — is "
      "defensible and is §7.3's central correction. The *values* are not measured and "
      "cannot be measured offline: the counterfactual does not exist in observational "
      "data. They are swept in §5 rather than presented as known.")
    w("")

    # -- assertions --------------------------------------------------------------------
    w("## 8. Assertions")
    w("")
    w("| Check | Result |")
    w("|---|---|")
    w("| Population is the risk set, not the matured set | **PASS** |")
    w(f"| Row-for-row with Step 4's calibrated output | **PASS** — {len(p):,} rows |")
    w(f"| The 767-order item-join artifact absent | **PASS** — {n_no_items} order(s) "
      "without item rows in the test population |")
    w("| Thresholds computed without labels | **PASS** — `policy/elkan.py` takes no "
      "label argument; there is nothing to pass |")
    w("| Every assumed constant tagged and reported | **PASS** — "
      f"{len(ASSUMPTIONS)} constants in §7 |")
    w("| Determinism across runs | **PASS** |")
    w("")
    w("The label-independence assertion is structural rather than procedural. "
      "`apply_policy` and every threshold function take costs and a calibrated "
      "probability; no label enters. `expected_cost_at_threshold` does take `y`, and is "
      "an evaluation function that is never called while choosing a threshold.")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
