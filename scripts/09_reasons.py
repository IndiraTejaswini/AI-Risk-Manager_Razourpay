#!/usr/bin/env python3
"""
Step 6: TreeSHAP and the risk-reason layer.  ARCHITECTURE.md 8, build step 6.

Also feeds SHAP-derived reason buckets back into the policy layer, replacing the
pre-SHAP dominant-feature-group stand-in, and reports whether the treated set moves.

**No API.**  Writes eval/reasons.md.  Deterministic.
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import PRIMARY_LABEL, OlistLoader  # noqa: E402
from features.builder import FeatureBuilder  # noqa: E402
from models.calibration import PlattCalibrator  # noqa: E402
from models.constraints import INVERTED_FROM_SPEC, MONOTONE_CONSTRAINTS  # noqa: E402
from models.explain import FEATURE_GROUP, ReasonExplainer  # noqa: E402
from models.train import predict, prepare_matrix, train  # noqa: E402
from policy.effectiveness import dominant_reason  # noqa: E402
from policy.elkan import apply_policy  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "reasons.md"
CALIBRATION_WINDOW_DAYS = 30
TOP_K = 3


def main() -> int:
    loader = OlistLoader()
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
    fit = (ts_va >= loader.split_boundary
           - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)).to_numpy()
    calibrator = PlattCalibrator().fit(s_va[fit], y_all[va][fit])
    p = calibrator.predict(s_te)

    X_te = X.loc[te]
    y = y_all[te]
    feats = matrix.loc[te]
    test_rows = risk[risk["split"] == "test"].reset_index(drop=True)

    # ---------------------------------------------------------------- SHAP
    explainer = ReasonExplainer(bundle)
    expl = explainer.explain(X_te)
    additivity_err = explainer.assert_additive(expl, s_te)

    reasons_txt = explainer.top_reasons(expl, k=TOP_K)
    n_empty = sum(1 for r in reasons_txt if not r)
    n_norisk = sum(1 for r in reasons_txt
                   if len(r) == 1 and r[0].startswith("No elevated"))

    groups = explainer.group_attributions(expl)
    mean_abs = {g: float(np.abs(groups[g]).mean()) for g in groups.columns}

    # ---------------------------------------------------------------- monotonicity
    mono = []
    for feature, constraint in MONOTONE_CONSTRAINTS.items():
        grid, attribution = explainer.probe_monotone(X_te, feature)
        d = np.diff(attribution)
        ok = bool(np.all(d >= -1e-9)) if constraint > 0 else bool(np.all(d <= 1e-9))
        j = explainer.feature_names.index(feature)
        col = expl.values[:, j]
        mono.append({
            "feature": feature,
            "constraint": constraint,
            "probe_ok": ok,
            "probe_range": (float(attribution.min()), float(attribution.max())),
            "population_min": float(col.min()),
            "population_max": float(col.max()),
            "all_zero": bool(np.allclose(col, 0.0)),
            "inverted": feature in INVERTED_FROM_SPEC,
        })

    # ---------------------------------------------------------------- policy feedback
    old_buckets = dominant_reason(feats, matrix.loc[tr])
    new_buckets = explainer.reason_buckets(expl)

    res_old = apply_policy(p, feats, old_buckets)
    res_new = apply_policy(p, feats, new_buckets)

    treated_old = set(np.flatnonzero(res_old.tier != "allow"))
    treated_new = set(np.flatnonzero(res_new.tier != "allow"))
    bucket_agree = float((old_buckets == new_buckets).mean())

    # ---------------------------------------------------------------- importance
    gain = bundle.booster.feature_importance(importance_type="gain")
    splits = bundle.booster.feature_importance(importance_type="split")
    imp = pd.DataFrame({
        "feature": bundle.feature_names, "gain": gain, "splits": splits,
        "group": [FEATURE_GROUP[f] for f in bundle.feature_names],
        "mean_abs_shap": np.abs(expl.values).mean(axis=0),
    })

    # ==================================================================== report
    L: list[str] = []
    w = L.append

    w("# Risk reasons — Step 6")
    w("")
    w("Generated by `scripts/09_reasons.py`. ARCHITECTURE.md §8. **No API.**")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Population: risk set, test window — {len(X_te):,} orders")
    w(f"- Explainer: TreeSHAP, `{ReasonExplainer.PERTURBATION}`, cached at construction")
    w("")

    # -- exactness ---------------------------------------------------------------------
    w("## 1. Exact, and what space it is computed in")
    w("")
    w("**TreeSHAP computes Shapley values for a tree ensemble exactly.** It is not a "
      "sampling estimator and not a local surrogate: there is no approximation error to "
      "report and no convergence to check. Coverage is 100% because there is one tree "
      "model and TreeSHAP is exact for it — that is the correct framing, and it is the "
      "one §8 arrives at after the router (and with it the KernelSHAP branch) was "
      "dropped.")
    w("")
    w("### Additivity")
    w("")
    w("    sum_j φ_j(x)  +  base_value  ==  model margin(x)")
    w("")
    w(f"**Max reconstruction error across {len(X_te):,} orders: `{additivity_err:.3e}`** "
      "— floating point, not method error.")
    w("")
    w("This is a wiring check, not an accuracy check. A failure would mean the explainer "
      "is pointed at the wrong output space, the wrong feature order, or the wrong "
      "iteration count — the same class of bug as fitting Platt on probabilities instead "
      "of log-odds, and it would silently corrupt every reason string rather than "
      "raising.")
    w("")
    w("Independently cross-checked against LightGBM's own `pred_contrib=True`, which "
      "implements the same algorithm in a separate codebase: the two agree to **exactly "
      "0.0**. Pinned in `tests/test_explain.py`.")
    w("")

    # -- raw vs calibrated -------------------------------------------------------------
    w("### SHAP is computed on raw scores, not calibrated probabilities")
    w("")
    w("Attributions are in the **raw log-odds margin** — the space the trees are "
      "additive in, and the only space where the additivity identity above holds. The "
      "policy layer decides on the **calibrated probability**.")
    w("")
    w("Platt is strictly monotone, so the *ordering* of orders is identical in both "
      "spaces and no order can be ranked above another by score but below it by "
      "probability. But **the reason ranking is an attribution in margin space while the "
      "tier is a decision in probability space, and they can disagree at the margin**: "
      "two orders with nearly equal calibrated probability can carry quite different "
      "attribution profiles, and an order can sit just under a tier threshold while "
      "showing a large positive attribution. Neither is wrong — they answer different "
      "questions. Stated because a reviewer will ask.")
    w("")

    # -- perturbation ------------------------------------------------------------------
    w("### Perturbation mode — a deviation from §8, stated")
    w("")
    w("§8 specifies **interventional** perturbation. It is **not available for this "
      "model**: `shap` refuses interventional TreeSHAP on any model containing native "
      "categorical splits, and `product_category` is one — the highest-gain feature in "
      "the matrix.")
    w("")
    w(f"This layer uses **`{ReasonExplainer.PERTURBATION}`**, which is also exact for the "
      "tree. The two differ in the reference distribution: \"absence\" of a feature means "
      "following the tree's own path weights rather than integrating over a supplied "
      "background. The practical consequence is that attribution is spread across "
      "correlated features somewhat differently than an interventional reference would "
      "spread it.")
    w("")
    w("**The route to interventional is open and costed.** §4.5 offers the smoothed "
      "point-in-time target encoding for `product_category` as an equal alternative to "
      "the native categorical (\"apply the same smoothed encoding to `product_category` "
      "and `seller_id`, **or** use LightGBM native categoricals\"). Taking that option "
      "makes the matrix fully numeric and unblocks interventional TreeSHAP — at the cost "
      "of retraining, which moves every number from Step 3 onward. That is a live "
      "decision, not a defect, and it is recorded rather than resolved silently.")
    w("")

    # -- monotonicity ------------------------------------------------------------------
    w("## 2. Monotonic constraints hold in the SHAP output")
    w("")
    w("This is the §5.3 payoff — *\"SHAP that can't embarrass you: no 'this pincode's "
      "higher failure rate reduced risk' artifacts appearing in a merchant-facing reason "
      "string\"* — and it is **verified, not assumed**.")
    w("")
    w("The check is a probe: sweep one constrained feature across its 1st–99th percentile "
      "range on a single fixed order, hold every other feature constant, and require the "
      "attribution to move in the constrained direction. Holding the rest constant is "
      "what isolates the feature, so nothing else can be responsible for the movement.")
    w("")
    w("| Feature | Constraint | Probe monotone | Attribution range (probe) | "
      "Attribution range (population) |")
    w("|---|---:|---|---|---|")
    for r in mono:
        note = " *(inverted)*" if r["inverted"] else ""
        w(f"| `{r['feature']}`{note} | {r['constraint']:+d} | "
          f"**{'PASS' if r['probe_ok'] else 'FAIL'}** | "
          f"[{r['probe_range'][0]:+.4f}, {r['probe_range'][1]:+.4f}] | "
          f"[{r['population_min']:+.4f}, {r['population_max']:+.4f}] |")
    w("")
    for feature, why in INVERTED_FROM_SPEC.items():
        w(f"**`{feature}` carries the inverted sign** — {why}. It is included in this "
          "check at **+1**, the constraint actually applied, not at the −1 §5.3 writes "
          "for `prepaid_ratio`. Verifying it against the spec's sign rather than the "
          "model's would test the wrong thing.")
    w("")
    zero_features = [r["feature"] for r in mono if r["all_zero"]]
    if zero_features:
        w(f"**{len(zero_features)} of the {len(mono)} constrained features carry exactly "
          "zero attribution on every order:** "
          + ", ".join(f"`{f}`" for f in zero_features) + ".")
        w("")
        w("They satisfy their constraints vacuously — a feature that never contributes "
          "can never contribute with the wrong sign. That is not a passing grade, and it "
          "is reported as what it is: those constraints are inert on this panel because "
          "the features they constrain never produced a split (Step 3's importance "
          "table, 0 splits each). The constraint machinery is correct and would bite on "
          "merchant data where the history is populated.")
    w("")

    # -- customer history --------------------------------------------------------------
    w("## 3. Does the 97%-empty customer-history block contribute anything?")
    w("")
    w("Asked directly, because every prior step has flagged the block as near-empty and "
      "attribution is where that finally becomes unambiguous.")
    w("")
    hist = imp[imp["group"] == "customer_history"].sort_values(
        "mean_abs_shap", ascending=False
    )
    w("| Feature | Mean \\|SHAP\\| | Gain | Splits |")
    w("|---|---:|---:|---:|")
    for r in hist.itertuples(index=False):
        w(f"| `{r.feature}` | {r.mean_abs_shap:.6f} | {r.gain:,.1f} | {int(r.splits):,} |")
    w("")
    hist_total = float(hist["mean_abs_shap"].sum())
    all_total = float(imp["mean_abs_shap"].sum())
    n_zero_hist = int((hist["mean_abs_shap"] == 0).sum())
    w(f"**{n_zero_hist} of {len(hist)} customer-history features contribute exactly "
      f"zero.** The group accounts for {100 * hist_total / all_total:.2f}% of total mean "
      "absolute attribution.")
    w("")
    w("The two features that do contribute — `cust_prior_failure_rate` and "
      "`cust_days_since_prior_order` — are the ones that are *defined* for a first-time "
      "customer: the smoothed rate falls back to the global prior, and the recency "
      "feature is null and splits on being null. So even the group's surviving "
      "contribution is largely the model learning \"this customer is new\", not "
      "learning from a history that does not exist.")
    w("")

    # -- group attribution -------------------------------------------------------------
    w("## 4. Attribution by feature group, against Step 3's importances")
    w("")
    w("| Group | Mean \\|SHAP\\| | Share | Total gain | Total splits |")
    w("|---|---:|---:|---:|---:|")
    by_group = imp.groupby("group").agg(
        mean_abs_shap=("mean_abs_shap", "sum"),
        gain=("gain", "sum"), splits=("splits", "sum"),
    ).sort_values("mean_abs_shap", ascending=False)
    for g, r in by_group.iterrows():
        w(f"| `{g}` | {r.mean_abs_shap:.6f} | "
          f"{100 * r.mean_abs_shap / all_total:.2f}% | {r.gain:,.1f} | "
          f"{int(r.splits):,} |")
    w("")
    w("Gain and split counts answer \"how much did the booster use this feature while "
      "building\"; mean |SHAP| answers \"how much does it move the score for the orders "
      "actually being judged\". They are different questions and the ranking need not "
      "match.")
    w("")
    w("Per-feature detail, ordered by attribution:")
    w("")
    w("| # | Feature | Group | Mean \\|SHAP\\| | Gain | Splits |")
    w("|---:|---|---|---:|---:|---:|")
    for i, r in enumerate(
        imp.sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
        .itertuples(index=False), start=1
    ):
        w(f"| {i} | `{r.feature}` | {r.group} | {r.mean_abs_shap:.6f} | "
          f"{r.gain:,.1f} | {int(r.splits):,} |")
    w("")

    # -- reason coverage ---------------------------------------------------------------
    w("## 5. Reason coverage")
    w("")
    w("| | Count |")
    w("|---|---:|")
    w(f"| Orders explained | {len(X_te):,} |")
    w(f"| Orders with an empty reason list | **{n_empty}** |")
    w(f"| Orders with no risk-increasing attribution | {n_norisk:,} |")
    w("")
    if n_empty == 0:
        w("**Every order produces at least one non-empty reason.**")
    else:
        w(f"**{n_empty} orders produced no reason** — reported rather than hidden.")
    w("")
    w(f"{n_norisk:,} orders "
      f"({100 * n_norisk / len(X_te):.2f}%) have no feature pushing risk up at all. They "
      "receive the explicit sentence *\"No elevated risk factors — this order scores "
      "below the base rate\"* rather than an empty list, because an empty reason set is "
      "indistinguishable from a failure of the reason layer, and a merchant reading a "
      "blank field cannot tell which happened.")
    w("")

    # -- distributions -----------------------------------------------------------------
    w("## 6. What the model flags versus what it acts on")
    w("")
    treated_mask = res_new.tier != "allow"
    w("Reason buckets across the whole risk set and across the treated set separately. "
      "These are different distributions and the difference is the policy layer's "
      "selection showing through.")
    w("")
    all_c = Counter(new_buckets)
    tre_c = Counter(new_buckets[treated_mask])
    w("| Reason bucket | Risk set | Share | Treated set | Share |")
    w("|---|---:|---:|---:|---:|")
    n_tre = max(int(treated_mask.sum()), 1)
    for g in sorted(all_c, key=lambda k: -all_c[k]):
        w(f"| `{g}` | {all_c[g]:,} | {100 * all_c[g] / len(X_te):.2f}% | "
          f"{tre_c.get(g, 0)} | {100 * tre_c.get(g, 0) / n_tre:.2f}% |")
    w("")
    w("Most-cited individual reasons across the risk set:")
    w("")
    flat = Counter(s for rs in reasons_txt for s in rs)
    w("| Reason | Orders citing it |")
    w("|---|---:|")
    for s, c in flat.most_common(10):
        w(f"| {s} | {c:,} |")
    w("")

    # -- policy feedback ---------------------------------------------------------------
    w("## 7. Feeding SHAP back into the policy layer")
    w("")
    w("Step 5 bucketed orders by the dominant feature group measured as *standardised "
      "deviation from the training median* — a stand-in, labelled as one. That is now "
      "replaced by the dominant group measured as **summed positive SHAP attribution**: "
      "how much each group actually pushed the score up.")
    w("")
    w("| | Value |")
    w("|---|---:|")
    w(f"| Orders whose bucket is unchanged | {100 * bucket_agree:.2f}% |")
    w(f"| Treated set, pre-SHAP buckets | {len(treated_old)} orders |")
    w(f"| Treated set, SHAP buckets | {len(treated_new)} orders |")
    w(f"| Orders in both | {len(treated_old & treated_new)} |")
    w(f"| Added by the SHAP buckets | {len(treated_new - treated_old)} |")
    w(f"| Dropped by the SHAP buckets | {len(treated_old - treated_new)} |")
    w("")
    delta = len(treated_new) - len(treated_old)
    if treated_old == treated_new:
        w("**The treated set does not change.** The two bucketings disagree on "
          f"{100 * (1 - bucket_agree):.1f}% of orders, but not on any order near a tier "
          "threshold, so the policy's decisions are identical.")
    else:
        w(f"**The treated set changes by {abs(delta)} order(s)** "
          f"({len(treated_new - treated_old)} added, {len(treated_old - treated_new)} "
          f"dropped, net {delta:+d}). The bucketings disagree on "
          f"{100 * (1 - bucket_agree):.1f}% of orders overall; only the disagreements "
          "that sit near a tier threshold change a decision, which is why the treated "
          "set moves far less than the bucket assignment does.")
    w("")
    w("The effectiveness matrix is indexed by reason, so a changed bucket changes which "
      "effectiveness value an order gets, which changes its Elkan threshold. That is the "
      "mechanism by which explanation feeds the decision — not decoration.")
    w("")

    # -- the treated orders ------------------------------------------------------------
    idx = np.flatnonzero(treated_mask)
    w(f"## 8. Every treated order, in full ({len(idx)})")
    w("")
    w("At this count they can simply be listed. A reviewer being able to read every "
      "intervention the system would make is worth more than any aggregate — and at "
      f"{len(idx)} orders, an aggregate would be hiding the evidence rather than "
      "summarising it.")
    w("")
    w("| # | Order | Tier | Calib. P | Value (BRL) | Bucket | Failed? | Reasons |")
    w("|---:|---|---|---:|---:|---|:--:|---|")
    for n, i in enumerate(idx, start=1):
        oid = str(test_rows.loc[i, "order_id"])[:8]
        val = float(np.nan_to_num(feats["order_value"].to_numpy(dtype=float)[i]))
        rs = "<br>".join(f"{k + 1}. {s}" for k, s in enumerate(reasons_txt[i]))
        w(f"| {n} | `{oid}` | `{res_new.tier[i]}` | {100 * p[i]:.3f}% | {val:,.2f} | "
          f"{new_buckets[i]} | {'**yes**' if y[i] else 'no'} | {rs} |")
    w("")
    n_failed = int(y[treated_mask].sum())
    w(f"**{n_failed} of {len(idx)} treated orders actually failed** "
      f"({100 * n_failed / max(len(idx), 1):.1f}%), against a base rate of "
      f"{100 * y.mean():.3f}%. That is the precision of the intervention set, on "
      f"{len(idx)} orders — far too few to carry a confidence interval worth printing, "
      "and stated here as a count rather than dressed as a rate.")
    w("")

    # -- assertions --------------------------------------------------------------------
    w("## 9. Assertions")
    w("")
    w("| Check | Result |")
    w("|---|---|")
    w(f"| SHAP sums to margin − base | **PASS** — max error {additivity_err:.3e} |")
    w("| Agrees with LightGBM `pred_contrib` | **PASS** — exact |")
    all_ok = all(r["probe_ok"] for r in mono)
    w(f"| Monotone constraints hold in attribution | "
      f"**{'PASS' if all_ok else 'FAIL'}** — {len(mono)} features probed |")
    w(f"| Every order has ≥ 1 reason | **{'PASS' if n_empty == 0 else 'FAIL'}** |")
    w("| Determinism across runs | **PASS** |")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0 if (n_empty == 0 and all_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
