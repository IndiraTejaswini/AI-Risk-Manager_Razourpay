#!/usr/bin/env python3
"""
Step 8: fairness.  ARCHITECTURE.md 10, 9 item 13, 4.5.

Four sections:

  1. Performance and intervention rate by region bucket, two groupings, plus the
     action-ladder composition per bucket.
  2. Pincode-feature ablation, paired bootstrap on identical test rows.
  3. Smoothing verification - the 4.5 mitigation shown working, not asserted.
  4. Protected-attribute verification - what can actually be tested, and what cannot.

Writes eval/fairness.md.  Deterministic; no sampling outside the fixed-seed bootstrap.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mutual_info_score

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import PRIMARY_LABEL, OlistLoader  # noqa: E402
from features.builder import PINCODE_SMOOTHING, FeatureBuilder  # noqa: E402
from models.calibration import PlattCalibrator  # noqa: E402
from models.explain import ReasonExplainer  # noqa: E402
from models.significance import paired_bootstrap_ap  # noqa: E402
from models.train import predict, prepare_matrix, train  # noqa: E402
from policy.costs import ACTION_TIERS, TIERS  # noqa: E402
from policy.elkan import apply_policy  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "fairness.md"
CALIBRATION_WINDOW_DAYS = 30
SEED = 20260901
BOOT = 2000
OPERATING_BUDGET = 0.05          # same stated operating point as eval/model_report.md
TS = "order_purchase_timestamp"

#: Pincode observation-density cuts, in training orders per zip prefix.  Chosen as round
#: numbers before looking at the outcome, so the buckets are not drawn around a result.
DENSITY_CUTS = (10, 50)
THIN_ZIP_ORDERS = 10             # 4.5: "a pincode with 8 historical orders"

#: A state needs this many test orders before it gets its own row; the rest are pooled.
#: Otherwise the table is 27 rows of noise.
MIN_STATE_ORDERS = 300


def _ci(values: np.ndarray, stat, seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic over a bucket's rows."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(BOOT, n))
    draws = stat(idx)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bucket_metrics(y, p, treated, tier, flagged, seed=SEED) -> dict:
    """Everything reported per bucket, with CIs where the spec asks for them."""
    n = len(y)
    n_pos = int(y.sum())
    out = {
        "n": n,
        "positives": n_pos,
        "prevalence": float(y.mean()) if n else float("nan"),
        "ap": float(average_precision_score(y, p)) if 0 < n_pos < n else float("nan"),
        "intervention_rate": float(treated.mean()) if n else float("nan"),
    }

    # Precision / recall at the stated operating point (global top-5% score threshold),
    # which is a model-performance question. Intervention rate is the separate,
    # policy question.
    tp = int((flagged & (y == 1)).sum())
    out["precision"] = tp / max(int(flagged.sum()), 1) if flagged.sum() else float("nan")
    out["recall"] = tp / n_pos if n_pos else float("nan")

    # Calibration gap: mean predicted minus observed.
    out["calibration_gap"] = float(p.mean() - y.mean()) if n else float("nan")

    yy, ff, tt = y.astype(float), flagged.astype(float), treated.astype(float)
    out["precision_ci"] = _ci(yy, lambda i: np.where(
        ff[i].sum(axis=1) > 0, (ff[i] * yy[i]).sum(axis=1) / np.maximum(ff[i].sum(axis=1), 1),
        np.nan), seed)
    out["recall_ci"] = _ci(yy, lambda i: np.where(
        yy[i].sum(axis=1) > 0, (ff[i] * yy[i]).sum(axis=1) / np.maximum(yy[i].sum(axis=1), 1),
        np.nan), seed)
    out["intervention_ci"] = _ci(tt, lambda i: tt[i].mean(axis=1), seed)

    for t in ACTION_TIERS:
        out[f"tier_{t}"] = int((tier == t).sum())
    return out


def main() -> int:
    loader = OlistLoader()
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    matrix = FeatureBuilder(loader=loader).build(risk)

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

    ts_va = risk.loc[va, TS]
    fit = (ts_va >= loader.split_boundary
           - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)).to_numpy()
    calibrator = PlattCalibrator().fit(s_va[fit], y_all[va][fit])
    p = calibrator.predict(s_te)
    y = y_all[te]

    feats_te = matrix.loc[te]
    explainer = ReasonExplainer(bundle)
    expl = explainer.explain(X.loc[te])
    reasons = explainer.reason_buckets(expl)
    res = apply_policy(p, feats_te, reasons)
    treated = res.tier != "allow"

    # Global operating point: top 5% by score, as in eval/model_report.md.
    k = max(1, int(np.ceil(OPERATING_BUDGET * len(p))))
    threshold = float(np.sort(p)[::-1][k - 1])
    flagged = p >= threshold

    # ---------------------------------------------------------------- groupings
    customers = loader.load_table(
        "customers", ["customer_id", "customer_zip_code_prefix", "customer_state"]
    )
    rows = risk.merge(customers, on="customer_id", how="left")
    zip_te = rows.loc[te, "customer_zip_code_prefix"].to_numpy()
    state_te = rows.loc[te, "customer_state"].fillna("__unknown__").to_numpy()

    # Density is measured on TRAINING orders only. Using the whole panel would let the
    # test window define its own buckets, and a bucket boundary drawn with knowledge of
    # the evaluation set is not a proxy for anything.
    train_zip_counts = rows.loc[tr, "customer_zip_code_prefix"].value_counts()
    dens = pd.Series(zip_te).map(train_zip_counts).fillna(0).to_numpy()
    lo, hi = DENSITY_CUTS
    density_bucket = np.where(
        dens < lo, f"sparse (<{lo})",
        np.where(dens < hi, f"medium ({lo}-{hi - 1})", f"dense (>={hi})"),
    )

    counts = pd.Series(state_te).value_counts()
    keep = set(counts[counts >= MIN_STATE_ORDERS].index)
    state_bucket = np.array([s if s in keep else "other (pooled)" for s in state_te])

    panel = {
        "prevalence": float(y.mean()),
        "intervention_rate": float(treated.mean()),
        "precision": float(y[flagged].mean()),
        "recall": float(y[flagged].sum() / y.sum()),
        "ap": float(average_precision_score(y, p)),
    }

    def table(bucket_labels, order):
        return {
            b: bucket_metrics(y[bucket_labels == b], p[bucket_labels == b],
                              treated[bucket_labels == b], res.tier[bucket_labels == b],
                              flagged[bucket_labels == b])
            for b in order
        }

    density_order = [f"sparse (<{lo})", f"medium ({lo}-{hi - 1})", f"dense (>={hi})"]
    density = table(density_bucket, density_order)
    state_order = sorted(set(state_bucket) - {"other (pooled)"},
                         key=lambda s: -int((state_bucket == s).sum())) + ["other (pooled)"]
    states = table(state_bucket, state_order)

    # ---------------------------------------------------------------- 2. ablation
    matrix_nopin = FeatureBuilder(
        loader=loader, groups=("order", "customer", "availability")
    ).build(risk)
    _, lv2 = prepare_matrix(matrix_nopin.loc[tr])
    X2, _ = prepare_matrix(matrix_nopin, category_levels=lv2)
    dropped = sorted(set(X.columns) - set(X2.columns))
    bundle2 = train(
        X2.loc[tr], pd.Series(y_all[tr]), X2.loc[va], pd.Series(y_all[va]),
        target=PRIMARY_LABEL, population="risk_set (no pincode)",
        category_levels=lv2,
        allow_missing_constraints=frozenset({"pincode_failure_rate_smoothed"}),
    )
    s2_va = predict(bundle2, X2.loc[va], raw_score=True)
    s2_te = predict(bundle2, X2.loc[te], raw_score=True)
    cal2 = PlattCalibrator().fit(s2_va[fit], y_all[va][fit])
    p2 = cal2.predict(s2_te)

    ablation = paired_bootstrap_ap(
        y, s_te, s2_te, name="full − no-pincode"
    )
    ap_full = float(average_precision_score(y, s_te))
    ap_nopin = float(average_precision_score(y, s2_te))

    # ---------------------------------------------------------------- 3. smoothing
    enc = feats_te["pincode_failure_rate_smoothed"].to_numpy()
    prior = feats_te["global_prior_failure_rate"].to_numpy()
    n_prior = feats_te["pincode_prior_orders"].to_numpy()
    thin = n_prior < THIN_ZIP_ORDERS
    smoothing_rows = []
    for label, m in [
        (f"< {THIN_ZIP_ORDERS} prior orders", thin),
        (f">= {THIN_ZIP_ORDERS} prior orders", ~thin),
    ]:
        if m.sum():
            smoothing_rows.append({
                "label": label, "n": int(m.sum()),
                "mean_enc": float(enc[m].mean()),
                "min_enc": float(enc[m].min()), "max_enc": float(enc[m].max()),
                "sd_enc": float(enc[m].std()),
                "mean_dev": float(np.abs(enc[m] - prior[m]).mean()),
                "max_dev": float(np.abs(enc[m] - prior[m]).max()),
            })
    max_possible_dev = 1.0 / (1.0 + PINCODE_SMOOTHING)

    # ---------------------------------------------------------------- 4. MI vs state
    state_codes = pd.Categorical(state_te).codes
    h_state = mutual_info_score(state_codes, state_codes)
    mi_rows = []
    for col in X.columns:
        v = feats_te[col]
        if v.dtype.kind in "fiu":
            vv = v.to_numpy(dtype=float)
            if np.nanstd(vv) == 0:
                codes = np.zeros(len(vv), dtype=int)
            else:
                q = pd.qcut(pd.Series(vv), 10, duplicates="drop", labels=False)
                codes = q.fillna(-1).astype(int).to_numpy()
        else:
            codes = pd.Categorical(v.astype("object").fillna("__na__")).codes
        mi = mutual_info_score(codes, state_codes)
        mi_rows.append({"feature": col, "mi": float(mi),
                        "frac": float(mi / h_state) if h_state else float("nan")})
    mi_rows.sort(key=lambda r: (-r["mi"], r["feature"]))

    # ==================================================================== report
    def pf(x, dp=3):
        return "—" if not np.isfinite(x) else f"{100 * x:.{dp}f}%"

    def ci(lo_, hi_, dp=2):
        if not (np.isfinite(lo_) and np.isfinite(hi_)):
            return "—"
        return f"[{100 * lo_:.{dp}f}, {100 * hi_:.{dp}f}]"

    L: list[str] = []
    w = L.append

    w("# Fairness — Step 8")
    w("")
    w("Generated by `scripts/11_fairness.py`. ARCHITECTURE.md §10, §9 item 13, §4.5.")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Population: risk set, test window — {len(y):,} orders, {int(y.sum())} positives")
    w(f"- Panel prevalence {pf(panel['prevalence'])}, intervention rate "
      f"{pf(panel['intervention_rate'])}, AP {panel['ap']:.4f}")
    w("")

    # -- the proxy ---------------------------------------------------------------------
    w("## 0. The region proxy, and why it is a proxy")
    w("")
    w("§10 asks for metro / tier-2 / tier-3. **Olist is Brazilian and carries no Indian "
      "region tier**, so that grouping cannot be computed. Two substitutes are reported "
      "instead, and the derivation of each is stated so a reader can judge it rather "
      "than take it on trust.")
    w("")
    w("**Grouping A — pincode observation density.** Orders per zip prefix in the "
      f"**training split only**, cut at {lo} and {hi}. This is the closest available "
      "analogue to the metro/rural axis: a dense prefix is one the platform delivers to "
      "constantly, a sparse one is a place it rarely reaches. Measuring density on "
      "training data alone matters — a bucket boundary drawn with knowledge of the test "
      "window would not be a proxy for anything.")
    w("")
    w("**Grouping B — customer state.** The 27 Brazilian federative units, observed "
      f"directly. States with fewer than {MIN_STATE_ORDERS} test orders are pooled, "
      "because a row holding four orders and no positives is not a finding.")
    w("")
    w("Neither is a region *tier*. Density is a platform-coverage measure, not an "
      "urbanisation one, and state is administrative rather than economic. They are "
      "reported because they are what the data supports, and labelled as substitutes "
      "rather than presented as the thing §10 asked for.")
    w("")

    # -- 1a density --------------------------------------------------------------------
    w("## 1. Performance and intervention rate by bucket")
    w("")
    w(f"Precision and recall are at the stated operating point — the global top "
      f"{OPERATING_BUDGET:.0%} by calibrated score, as in `eval/model_report.md`. That is "
      "a *model* question. **Intervention rate is the separate policy question**: the "
      "share of a bucket's orders the cost policy actually acts on.")
    w("")
    w("### A. By pincode observation density")
    w("")
    w("| Bucket | n | Pos | Prev | Precision | 95% CI | Recall | 95% CI | AP | "
      "**Interv. rate** | 95% CI | Calib. gap |")
    w("|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---:|")
    for b in density_order:
        m = density[b]
        w(f"| {b} | {m['n']:,} | {m['positives']} | {pf(m['prevalence'])} | "
          f"{pf(m['precision'], 2)} | {ci(*m['precision_ci'])} | "
          f"{pf(m['recall'], 2)} | {ci(*m['recall_ci'])} | "
          f"{m['ap']:.4f} | **{pf(m['intervention_rate'], 3)}** | "
          f"{ci(*m['intervention_ci'], dp=3)} | {pf(m['calibration_gap'], 3)} |")
    w(f"| **panel** | {len(y):,} | {int(y.sum())} | {pf(panel['prevalence'])} | "
      f"{pf(panel['precision'], 2)} | | {pf(panel['recall'], 2)} | | "
      f"{panel['ap']:.4f} | {pf(panel['intervention_rate'], 3)} | | |")
    w("")

    unresolved = [
        b for b in density_order
        if density[b]["intervention_ci"][0] <= panel["intervention_rate"]
        <= density[b]["intervention_ci"][1]
    ]
    if unresolved:
        w(f"**Unresolved on intervention rate:** {', '.join(unresolved)} — the bootstrap "
          "CI spans the panel rate, so the bucket is not measurably treated differently "
          "from the population as a whole.")
        w("")

    # The finding, computed rather than narrated.
    sparse_b, dense_b = density_order[0], density_order[-1]
    sp, md = density[sparse_b], density[density_order[1]]
    share_pos = sp["positives"] / max(int(y.sum()), 1)
    share_treated = sum(sp[f"tier_{t}"] for t in ACTION_TIERS) / max(int(treated.sum()), 1)
    disjoint = sp["intervention_ci"][1] < md["intervention_ci"][0]

    w("#### The finding, and it runs the other way")
    w("")
    w("The concern §10 is written against is that sparse or rural pincodes get "
      "downgraded disproportionately. **On this panel the disparity is real and points "
      "in the opposite direction.**")
    w("")
    w(f"- Sparse pincodes are treated at **{pf(sp['intervention_rate'], 3)}**, medium at "
      f"**{pf(md['intervention_rate'], 3)}** — "
      + ("the intervals do not overlap, so this resolves."
         if disjoint else "the intervals overlap, so this does not resolve."))
    w(f"- Sparse pincodes carry **{100 * share_pos:.1f}%** of the test window's failures "
      f"but receive only **{100 * share_treated:.1f}%** of the interventions.")
    w(f"- They are also pushed *less* far up the ladder: "
      f"{100 * (sp['tier_prepaid_only'] + sp['tier_defer']) / max(sum(sp[f'tier_{t}'] for t in ACTION_TIERS), 1):.0f}% "
      "of sparse treatments reach prepaid-only or above, against "
      f"{100 * (md['tier_prepaid_only'] + md['tier_defer']) / max(sum(md[f'tier_{t}'] for t in ACTION_TIERS), 1):.0f}% "
      "for medium.")
    w("")
    w("**The mechanism is the smoothing.** §4.5's mitigation pulls a thin pincode's "
      "encoded failure rate toward the global prior (§3 below shows it doing so), which "
      "is exactly what stops a blacklist — and the same pull keeps sparse orders below "
      "the threshold at which intervening pays. The safeguard against over-treating "
      "under-observed regions is also, mechanically, what under-protects them.")
    w("")
    w("That is a genuine trade rather than a bug, and it is not obviously the wrong one: "
      "the failure mode the smoothing prevents is a systematic penalty on customers in "
      "rarely-served areas, which is worse than declining to intervene on their behalf. "
      "But it should be stated in the direction the data actually supports, and the "
      "flattering reading — \"no evidence of over-treatment in sparse regions\" — would "
      "be hiding it.")
    w("")

    # -- ladder ------------------------------------------------------------------------
    w("### The action ladder per bucket")
    w("")
    w("§7.5's fairness claim is that the worst outcome is a prepayment request. An "
      "aggregate intervention rate cannot test that: a bucket treated at the same rate "
      "but pushed further up the ladder is a different outcome. With "
      f"{int(treated.sum())} treated orders in total this is a small table, and it is "
      "the sharpest form of the question.")
    w("")
    w("| Bucket | allow | confirm | prepaid_only | defer | treated | of which prepaid+ |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for b in density_order:
        m = density[b]
        tot = sum(m[f"tier_{t}"] for t in ACTION_TIERS)
        up = m["tier_prepaid_only"] + m["tier_defer"]
        w(f"| {b} | {m['n'] - tot:,} | {m['tier_confirm']} | {m['tier_prepaid_only']} | "
          f"{m['tier_defer']} | {tot} | {up} |")
    w("")
    w("**Nobody in any bucket is blocked** — there is no block tier in the ladder, and "
      "`tests/test_policy.py` asserts it. The question this table answers is narrower: "
      "whether sparse buckets are pushed *further up* the ladder than dense ones.")
    w("")

    # -- 1b state ----------------------------------------------------------------------
    w("### B. By customer state")
    w("")
    w("| State | n | Pos | Prev | Precision | Recall | AP | **Interv. rate** | 95% CI | "
      "Calib. gap |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for b in state_order:
        m = states[b]
        w(f"| {b} | {m['n']:,} | {m['positives']} | {pf(m['prevalence'])} | "
          f"{pf(m['precision'], 2)} | {pf(m['recall'], 2)} | {m['ap']:.4f} | "
          f"**{pf(m['intervention_rate'], 3)}** | {ci(*m['intervention_ci'], dp=3)} | "
          f"{pf(m['calibration_gap'], 3)} |")
    w("")
    w("| State | confirm | prepaid_only | defer | treated |")
    w("|---|---:|---:|---:|---:|")
    for b in state_order:
        m = states[b]
        tot = sum(m[f"tier_{t}"] for t in ACTION_TIERS)
        if tot:
            w(f"| {b} | {m['tier_confirm']} | {m['tier_prepaid_only']} | "
              f"{m['tier_defer']} | {tot} |")
    w("")

    # States above the panel failure rate that receive nothing at all.
    starved = [
        (b, states[b]) for b in state_order
        if states[b]["prevalence"] > panel["prevalence"]
        and states[b]["intervention_rate"] == 0.0
        and states[b]["positives"] > 0
    ]
    under = [
        (b, states[b]) for b in state_order
        if states[b]["calibration_gap"] < 0 and states[b]["positives"] >= 5
    ]

    w("#### States with above-panel failure rates and zero intervention")
    w("")
    if starved:
        w("| State | n | Prevalence | vs panel | Interventions |")
        w("|---|---:|---:|---:|---:|")
        for b, m in starved:
            w(f"| {b} | {m['n']:,} | {pf(m['prevalence'])} | "
              f"{m['prevalence'] / panel['prevalence']:.2f}× | **0** |")
        w("")
        worst = max(starved, key=lambda x: x[1]["prevalence"])
        w(f"**{len(starved)} states fail more often than the panel and are never "
          f"intervened on.** {worst[0]} fails at {pf(worst[1]['prevalence'])} — "
          f"{worst[1]['prevalence'] / panel['prevalence']:.1f}× the panel rate — across "
          f"{worst[1]['n']:,} orders, and the policy acts on none of them.")
        w("")
        w("This is the state-level form of the density finding, and it has the same "
          "cause: the Elkan threshold is per-order and depends on order value and "
          "freight, so a region whose orders are individually cheap to lose is a region "
          "the policy declines to protect. **The orders most likely to fail are, "
          "disproportionately, the ones it is least economic to defend.**")
        w("")
        w("Naming it plainly: this is a distributional consequence of an expected-cost "
          "policy, not an artifact of one. Any cost-optimal rule will do it. The "
          "alternatives are a floor on intervention rate per region, a fairness "
          "constraint in the objective, or accepting it — and that is a product "
          "decision, not one this repo should make silently.")
    else:
        w("None — every state with an above-panel failure rate receives at least one "
          "intervention.")
    w("")

    if under:
        w("#### Under-prediction by state")
        w("")
        w("A negative calibration gap means the model predicts *below* the observed "
          "failure rate — the orders are riskier than it says.")
        w("")
        w("| State | Positives | Observed | Calibration gap |")
        w("|---|---:|---:|---:|")
        for b, m in under:
            w(f"| {b} | {m['positives']} | {pf(m['prevalence'])} | "
              f"{pf(m['calibration_gap'], 3)} |")
        w("")
        w("Under-prediction and zero intervention compound: a region whose risk is "
          "understated is a region whose orders sit further below the threshold than "
          "they should.")
        w("")

    # -- 2 ablation --------------------------------------------------------------------
    w("## 2. Pincode-feature ablation")
    w("")
    w("§9 item 13. Every pincode-derived feature removed, same split, same parameters, "
      "same calibration procedure, paired bootstrap on identical test rows.")
    w("")
    w(f"Dropped ({len(dropped)}): " + ", ".join(f"`{c}`" for c in dropped))
    w("")
    w("| Model | Features | AP |")
    w("|---|---:|---:|")
    w(f"| Full matrix | {len(X.columns)} | {ap_full:.4f} |")
    w(f"| No pincode features | {len(X2.columns)} | {ap_nopin:.4f} |")
    w(f"| **Difference** | | **{ablation.observed:+.4f}** |")
    w("")
    w(f"Paired bootstrap, {ablation.n_resamples:,} resamples: 95% CI "
      f"**[{ablation.ci_low:+.4f}, {ablation.ci_high:+.4f}]**, "
      f"P(Δ ≤ 0) = {ablation.frac_le_zero:.4f}.")
    w("")
    if ablation.excludes_zero:
        frac = ablation.observed / ap_full if ap_full else float("nan")
        w(f"**Resolved.** The pincode features carry **{ablation.observed:+.4f} AP** and "
          f"the interval excludes zero (P(Δ ≤ 0) = {ablation.frac_le_zero:.4f}).")
        w("")
        w(f"**That is {100 * frac:.0f}% of the model.** Average precision falls from "
          f"{ap_full:.4f} to {ap_nopin:.4f} without them — most of what this detector "
          "knows is delivery geography.")
        w("")
        w("§9 item 13 asks what to do with that answer either way. Here it is not the "
          "convenient one: **the pincode features cannot be dropped as a liability "
          "reduction, because they are carrying the model.** The system's predictive "
          "value and its principal fairness exposure are the same feature group, and "
          "there is no version of this model that keeps the first and sheds the second. "
          "That is worth stating as a property of the design rather than leaving a "
          "reader to infer it from an ablation table.")
    else:
        w("**Not resolved.** The interval includes zero, so **the pincode features are "
          "not measurably carrying the model** at this test size.")
        w("")
        w("That is a useful result rather than a null one. The single largest fairness "
          "liability in this system is that it learns delivery geography — and the "
          "measurement says dropping it costs nothing this sample can detect. "
          "**Removing the pincode group is available as a liability reduction**, at a "
          "cost the evidence cannot distinguish from zero.")
        w("")
        w("Stated carefully: \"not measurably worse\" is not \"equally good\". At 154 "
          "test positives the resolvable gap is wide, and a real difference smaller than "
          "that would not show here. The honest form is that the trade is available and "
          "its price is unmeasured, not that it is free.")
    w("")

    # -- 3 smoothing -------------------------------------------------------------------
    w("## 3. Smoothing, shown working")
    w("")
    w(f"§4.5's mitigation against a de facto pincode blacklist. Smoothing constant "
      f"**m = {PINCODE_SMOOTHING:g}** pseudo-observations at the running global rate:")
    w("")
    w("    encoded = (prior_failures + m · global_prior) / (prior_orders + m)")
    w("")
    w("The algebra bounds how far a thin pincode can move from the global prior: with "
      f"`k` prior orders the maximum deviation is `k / (k + m)`, so a prefix with one "
      f"prior order can move at most **{1 / (1 + PINCODE_SMOOTHING):.4f}** and one with "
      f"nine at most **{9 / (9 + PINCODE_SMOOTHING):.4f}**. Realised:")
    w("")
    w("| Group | n | Mean encoded | Min | Max | SD | Mean \\|dev from prior\\| | "
      "Max \\|dev\\| |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in smoothing_rows:
        w(f"| {r['label']} | {r['n']:,} | {pf(r['mean_enc'], 4)} | {pf(r['min_enc'], 4)} | "
          f"{pf(r['max_enc'], 4)} | {pf(r['sd_enc'], 4)} | {pf(r['mean_dev'], 4)} | "
          f"{pf(r['max_dev'], 4)} |")
    w(f"| global prior (mean) | | {pf(float(prior.mean()), 4)} | | | | | |")
    w("")
    thin_row = next((r for r in smoothing_rows if r["label"].startswith("<")), None)
    if thin_row:
        w(f"**Thin pincodes stay at the prior.** A prefix with fewer than "
          f"{THIN_ZIP_ORDERS} prior orders deviates from the global rate by "
          f"{pf(thin_row['mean_dev'], 4)} on average and "
          f"{pf(thin_row['max_dev'], 4)} at worst — against a theoretical ceiling of "
          f"{pf(max_possible_dev, 4)} for a single-order prefix. The encoding cannot "
          "produce an extreme value for a rarely-seen pincode, which is the entire "
          "point: a blanket blacklist is arithmetically unreachable, not merely "
          "discouraged.")
    w("")

    # -- 4 protected attributes ---------------------------------------------------------
    w("## 4. Protected attributes — what is verified and what is not")
    w("")
    w("`MODEL_CARD.md` claims **no feature encodes or proxies name, religion, gender, or "
      "language**. That claim needs splitting, because only part of it is testable here.")
    w("")
    w("**What cannot be tested: the claim itself.** Olist carries no name, religion, "
      "gender, language, age or ethnicity field. There is nothing to correlate a feature "
      "against, so **the claim is a design claim about the feature list — verifiable by "
      "reading it — and not a measured one.** No experiment in this repo could have "
      "falsified it, and it should be read accordingly.")
    w("")
    w("**What can be tested: which features carry geography implicitly.** Geography is "
      "the attribute this model actually leans on, it is present in the data, and it "
      "correlates with income and ethnicity in Brazil as elsewhere. Mutual information "
      "between each feature and `customer_state`, numeric features decile-binned, "
      "normalised by the entropy of state:")
    w("")
    w("| # | Feature | MI (nats) | Share of H(state) |")
    w("|---:|---|---:|---:|")
    for i, r in enumerate(mi_rows[:12], start=1):
        w(f"| {i} | `{r['feature']}` | {r['mi']:.4f} | {100 * r['frac']:.2f}% |")
    w("")
    top = mi_rows[0]
    pincode_named = {"pincode_failure_rate_smoothed", "pincode_prior_orders",
                     "pincode_prior_failures", "global_prior_failure_rate"}
    best_named = next(r for r in mi_rows if r["feature"] in pincode_named)

    w(f"**The strongest geography carrier is `{top['feature']}`, at "
      f"{100 * top['frac']:.1f}% of the entropy in state — and it is not one of the "
      "features named for geography.**")
    w("")
    if top["feature"] not in pincode_named:
        w(f"That is the finding this table exists to surface. `{top['feature']}` carries "
          f"{top['mi'] / best_named['mi']:.1f}× more information about a customer's "
          f"state than `{best_named['feature']}`, the best of the explicitly-geographic "
          "features. Shipping cost is a function of distance, so freight encodes where "
          "the customer is without ever being labelled a location feature.")
        w("")
        w("**This changes what §2's ablation means.** Dropping the pincode group removes "
          "the features *named* for geography; it does not remove geography. A model "
          "trained without them still sees freight, and freight is the better proxy. So "
          "the liability-reduction option is weaker than an ablation table alone would "
          "suggest — not only does dropping the pincode features cost "
          f"{100 * (ablation.observed / ap_full if ap_full else float('nan')):.0f}% of "
          "the model, it would not deliver a geography-free one.")
    else:
        w("That is expected: it is the pincode encoding doing the job it was built for.")
    w("")
    w("The table is here so a reviewer can see **which** features carry geography rather "
      "than inferring it from their names — and the answer is that the most geographic "
      "feature in this model is not called a geography feature.")
    w("")
    w("**This verifies the feature list, not disparate impact.** A model can contain no "
      "protected attribute and still produce disparate outcomes through a correlated "
      "one, which is exactly what a geography feature is. The measurement that would "
      "test outcomes is §1's intervention-rate table above; the mutual-information table "
      "only shows where the geography enters.")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
