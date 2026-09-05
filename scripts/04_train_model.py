#!/usr/bin/env python3
"""
Step 3: train LightGBM and report.  ARCHITECTURE.md 5, 9.

Three fits, all on the same split and the same parameters:

    primary    label_b on the risk set          P(fails | ships)   - the headline
    baseline   label_b, order-group features    what the encodings bought
    secondary  label_a on all matured orders    P(fails)           - benchmark

Writes eval/model_report.md and eval/figures/*.png.  Models to artifacts/models/.

**No calibration.** The booster's output is a score, not a probability. Step 4.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# The report contains characters outside cp1252 (U+2212 MINUS SIGN); a Windows
# console would raise on print() without this.  The file is written as UTF-8
# regardless.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import PRIMARY_LABEL, SECONDARY_LABEL, OlistLoader  # noqa: E402
from features.builder import FeatureBuilder  # noqa: E402
from models.constraints import (  # noqa: E402
    INVERTED_FROM_SPEC,
    MONOTONE_CONSTRAINTS,
    OMITTED_FROM_SPEC,
)
from models.evaluate import (  # noqa: E402
    OPERATING_BUDGET,
    Metrics,
    evaluate,
    pr_points,
    roc_points,
)
from models.plots import plot_pr, plot_roc  # noqa: E402
from models.train import (  # noqa: E402
    DROPPED_CONSTANT_COLUMNS,
    ORDER_ONLY_ABSENT_CONSTRAINTS,
    NUM_BOOST_ROUND,
    ORDER_ONLY_GROUPS,
    PARAMS,
    ModelBundle,
    predict,
    prepare_matrix,
    train,
)

OUT_PATH = REPO_ROOT / "eval" / "model_report.md"
FIG_DIR = REPO_ROOT / "eval" / "figures"
MODEL_DIR = REPO_ROOT / "artifacts" / "models"

#: Committed in eval/positive_counts.md section 6 / eval/label_targets.md section 3.
MDD_PRIMARY = 0.043
MDD_SECONDARY = 0.027


def _fit(
    population: pd.DataFrame,
    matrix: pd.DataFrame,
    label: str,
    name: str,
    allow_missing_constraints: frozenset[str] = frozenset(),
) -> tuple[ModelBundle, dict[str, Metrics], dict[str, np.ndarray]]:
    split = population["split"].to_numpy()
    y = population[label].astype(int).to_numpy()

    tr, va, te = split == "train", split == "validation", split == "test"

    # Category levels from the training split only, so a level first seen later cannot
    # extend the encoding.
    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)

    bundle = train(
        X.loc[tr], pd.Series(y[tr]),
        X.loc[va], pd.Series(y[va]),
        target=label, population=name, category_levels=levels,
        allow_missing_constraints=allow_missing_constraints,
    )

    scores = {s: predict(bundle, X.loc[m]) for s, m in (("validation", va), ("test", te))}
    metrics = {
        "validation": evaluate(y[va], scores["validation"]),
        "test": evaluate(y[te], scores["test"]),
    }
    return bundle, metrics, scores


def main() -> int:
    loader = OlistLoader()

    # ---------------------------------------------------------------- primary
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    matrix_full = FeatureBuilder().build(risk)
    primary, primary_metrics, primary_scores = _fit(
        risk, matrix_full, PRIMARY_LABEL, "risk_set"
    )

    # determinism: refit and compare artifact bytes and predictions
    primary_again, _, again_scores = _fit(risk, matrix_full, PRIMARY_LABEL, "risk_set")
    model_identical = primary.model_sha256 == primary_again.model_sha256
    preds_identical = bool(
        np.array_equal(primary_scores["test"], again_scores["test"])
    )

    # ---------------------------------------------------------------- baseline
    matrix_order = FeatureBuilder(groups=ORDER_ONLY_GROUPS).build(risk)
    baseline, baseline_metrics, baseline_scores = _fit(
        risk, matrix_order, PRIMARY_LABEL, "risk_set (order features only)",
        allow_missing_constraints=ORDER_ONLY_ABSENT_CONSTRAINTS,
    )

    # ---------------------------------------------------------------- secondary
    matured = loader.labelled().join(loader.split_labelled()["split"], how="left")
    matrix_sec = FeatureBuilder().build(matured)
    secondary, secondary_metrics, secondary_scores = _fit(
        matured, matrix_sec, SECONDARY_LABEL, "all matured"
    )

    # Orders with no order_items rows.  On this panel that absence is a CONSEQUENCE of
    # the outcome - an order that goes `unavailable` has no items in the export - so it
    # is post-outcome information, not a checkout-time state.  Quantified and corrected
    # below rather than left in the benchmark.
    item_ids = set(loader.load_table("items", ["order_id"])["order_id"])
    joins_items = matured["order_id"].isin(item_ids)
    leak_n = int((~joins_items).sum())
    leak_pos = int(matured.loc[~joins_items, SECONDARY_LABEL].sum())
    te_mask = matured["split"] == "test"
    leak_test_n = int((~joins_items & te_mask).sum())
    leak_test_pos = int(matured.loc[~joins_items & te_mask, SECONDARY_LABEL].sum())

    matured_clean = matured[joins_items].copy()
    matrix_sec_clean = FeatureBuilder().build(matured_clean)
    secondary_fixed, secondary_fixed_metrics, _ = _fit(
        matured_clean, matrix_sec_clean, SECONDARY_LABEL, "matured, item-joined only"
    )

    risk_joins = int((~loader.risk_set()["order_id"].isin(item_ids)).sum())

    for b, sub in ((primary, "primary"), (baseline, "baseline"), (secondary, "secondary")):
        b.save(MODEL_DIR / sub)

    # ---------------------------------------------------------------- figures
    y_test_primary = risk.loc[risk["split"] == "test", PRIMARY_LABEL].astype(int).to_numpy()
    y_test_secondary = (
        matured.loc[matured["split"] == "test", SECONDARY_LABEL].astype(int).to_numpy()
    )

    pm, bm = primary_metrics["test"], baseline_metrics["test"]
    r1, p1 = pr_points(y_test_primary, primary_scores["test"])
    r2, p2 = pr_points(y_test_primary, baseline_scores["test"])
    plot_pr(
        {
            "full": (r1, p1, f"Full matrix (AP {pm.average_precision:.4f})"),
            "baseline": (r2, p2, f"Order fields only (AP {bm.average_precision:.4f})"),
        },
        prevalence=pm.prevalence,
        path=FIG_DIR / "pr_primary.png",
        title="Precision-recall, primary target — P(fails | ships), test window",
    )
    f1, t1 = roc_points(y_test_primary, primary_scores["test"])
    f2, t2 = roc_points(y_test_primary, baseline_scores["test"])
    plot_roc(
        {
            "full": (f1, t1, f"Full matrix (AUC {pm.roc_auc:.3f})"),
            "baseline": (f2, t2, f"Order fields only (AUC {bm.roc_auc:.3f})"),
        },
        path=FIG_DIR / "roc_primary.png",
        title="ROC, primary target — shown alongside, not the headline",
    )

    sm = secondary_metrics["test"]
    r3, p3 = pr_points(y_test_secondary, secondary_scores["test"])
    plot_pr(
        {"full": (r3, p3, f"Full matrix (AP {sm.average_precision:.4f})")},
        prevalence=sm.prevalence,
        path=FIG_DIR / "pr_secondary.png",
        title="Precision-recall, secondary target — P(fails), benchmark only",
    )

    # ---------------------------------------------------------------- report
    L: list[str] = []
    w = L.append

    def pctf(x: float, dp: int = 3) -> str:
        return f"{100 * x:.{dp}f}%"

    w("# Model report — Step 3")
    w("")
    w("Generated by `scripts/04_train_model.py`. **No calibration layer.** The booster's "
      "output is a score, not a probability; the Platt map is Step 4 and does not exist "
      "yet. Nothing here should be read as a calibrated risk.")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Model sha256 (primary): `{primary.model_sha256}`")
    w("")

    # -- the number that governs every comparison below --------------------------------
    w("## Sample size and what it can resolve")
    w("")
    w("Printed first, ahead of any metric, per ARCHITECTURE.md §9 item 1.")
    w("")
    w("| | PRIMARY | SECONDARY |")
    w("|---|---:|---:|")
    w(f"| Test rows | {pm.n:,} | {sm.n:,} |")
    w(f"| **Test positives** | **{pm.n_positives}** | **{sm.n_positives}** |")
    w(f"| Test prevalence | **{pctf(pm.prevalence)}** | {pctf(sm.prevalence)} |")
    w(f"| **Minimum detectable AP difference** | **{MDD_PRIMARY:.3f}** | "
      f"{MDD_SECONDARY:.3f} |")
    w("")
    w(f"Any AP difference below **{MDD_PRIMARY:.3f}** on the primary target is reported "
      'as "not resolvable at this sample size" — not as a win, and not rerun on the '
      "secondary target until it resolves.")
    w("")

    # -- headline ----------------------------------------------------------------------
    delta = pm.ap_over_baseline
    resolved = delta > MDD_PRIMARY
    w("## 1. Primary target — P(fails | ships)")
    w("")
    w(f"`{PRIMARY_LABEL}` on the risk set. Natural class distribution, no resampling.")
    w("")
    w("| Metric | Value |")
    w("|---|---:|")
    w(f"| **Average precision** | **{pm.average_precision:.4f}** |")
    w(f"| Prevalence baseline | {pm.prevalence:.4f} |")
    w(f"| AP − baseline | **{delta:+.4f}** |")
    w(f"| Lift over baseline | {pm.lift:.2f}× |")
    w(f"| ROC-AUC *(not the headline)* | {pm.roc_auc:.4f} |")
    w(f"| Boosting rounds (early-stopped) | {primary.best_iteration} of {NUM_BOOST_ROUND} |")
    w("")
    w("![Precision-recall, primary](figures/pr_primary.png)")
    w("")
    if resolved:
        w(f"**The model beats the prevalence baseline by {delta:.4f} AP, which exceeds "
          f"the {MDD_PRIMARY:.3f} minimum detectable difference.** The improvement is "
          "resolvable at this test size.")
    else:
        w("### The gap is below the planning MDD — which is not a test, "
          "and `eval/significance.md` resolves it")
        w("")
        w(f"AP is {pm.average_precision:.4f} against a prevalence baseline of "
          f"{pm.prevalence:.4f} — a gap of **{delta:+.4f}**, against a minimum "
          f"detectable difference of **{MDD_PRIMARY:.3f}** at {pm.n_positives} test "
          "positives.")
        w("")
        w("**Do not read that as \"the model cannot be shown to beat the base rate\".** "
          "This section once did, and it was wrong — wrong in the conservative "
          "direction, which is still wrong. The MDD is a *planning* figure computed "
          "before seeing data under an assumed AP and an assumed correlation; comparing "
          "two realised point estimates against it discards the pairing, assumes rho "
          "rather than measuring it, and treats the prevalence baseline as a fixed "
          "constant when its AP moves with the resample.")
        w("")
        w("**`eval/significance.md` runs the test this comparison is standing in for, "
          "and the model separates.** The paired bootstrap on the AP difference gives a "
          "95% CI that excludes zero, and the permutation null against random ranking "
          "gives p = 0.00060. Section 6 there explains why the planning figure was "
          "roughly 3.5x too pessimistic here: it assumed AP around 0.10 against a "
          f"realised {pm.average_precision:.3f}, and the standard error of average "
          "precision scales with sqrt(AP(1-AP)).")
        w("")
        w(f"What this section does support: the ranking is better than chance by every "
          f"point estimate here — {pm.lift:.2f}× lift, ROC-AUC {pm.roc_auc:.3f} — and "
          "nothing was tuned to make it so. Tuning against the test set until a gap "
          "clears a threshold would manufacture the number rather than measure it. The "
          "MDD is retained above as the planning figure it is, and never as a decision "
          "rule.")
    w("")

    # -- operating point ---------------------------------------------------------------
    w("### Operating point")
    w("")
    w(f"**Provisional.** The per-order Elkan threshold is Step 5 and does not exist, so "
      f"the operating point here is a stated intervention budget: treat the top "
      f"{OPERATING_BUDGET:.0%} of scores.")
    w("")
    w("| | Predicted negative | Predicted positive |")
    w("|---|---:|---:|")
    w(f"| **Actual negative** | {pm.tn:,} | {pm.fp:,} |")
    w(f"| **Actual positive** | {pm.fn:,} | {pm.tp:,} |")
    w("")
    w(f"Precision **{pctf(pm.precision, 2)}**, recall **{pctf(pm.recall, 2)}** at "
      f"score threshold `{pm.threshold:.6f}`.")
    w("")
    w("### Precision@k and recall@k at the intervention budget")
    w("")
    w("| Budget | Orders treated | True positives | Precision | Recall | Lift |")
    w("|---|---:|---:|---:|---:|---:|")
    for k, d in pm.at_k.items():
        w(f"| top {k:.0%} | {d['n_selected']:,} | {d['tp']} | "
          f"{pctf(d['precision'], 2)} | {pctf(d['recall'], 2)} | {d['lift']:.2f}× |")
    w("")

    # -- baseline comparison -----------------------------------------------------------
    enc_delta = pm.average_precision - bm.average_precision
    w("## 2. What the encodings bought — order-fields-only baseline")
    w("")
    w("Same split, same parameters, same early stopping. The only difference is the "
      "feature set: order fields alone, with no customer history, no pincode encoding, "
      "and no availability flags.")
    w("")
    w("Availability is excluded from the baseline deliberately — `n_missing_features` "
      "is computed across the customer-history columns, so it carries history-presence "
      "information and belongs on the encodings side of the comparison.")
    w("")
    w("| Model | Features | AP | ROC-AUC | Rounds |")
    w("|---|---:|---:|---:|---:|")
    w(f"| Full matrix | {len(primary.feature_names)} | "
      f"**{pm.average_precision:.4f}** | {pm.roc_auc:.4f} | {primary.best_iteration} |")
    w(f"| Order fields only | {len(baseline.feature_names)} | "
      f"{bm.average_precision:.4f} | {bm.roc_auc:.4f} | {baseline.best_iteration} |")
    w(f"| **Difference** | | **{enc_delta:+.4f}** | {pm.roc_auc - bm.roc_auc:+.4f} | |")
    w("")
    if abs(enc_delta) > MDD_PRIMARY:
        better = "the full matrix" if enc_delta > 0 else "the order-only baseline"
        w(f"The gap of {abs(enc_delta):.4f} AP exceeds the {MDD_PRIMARY:.3f} threshold, "
          f"so **{better} is measurably better** at this test size.")
    else:
        w(f"**Not resolvable.** The gap of {abs(enc_delta):+.4f} AP is inside the "
          f"{MDD_PRIMARY:.3f} minimum detectable difference, so this comparison does "
          "not establish that the history and pincode encodings help — or that they do "
          "not. Given customer history is empty for 97% of rows, a small or absent "
          "contribution is the expected outcome rather than a surprise; what this test "
          "size cannot do is separate 'small' from 'zero'. This row feeds §9's ablation "
          "section as an unresolved comparison.")
    w("")

    # -- feature importance ------------------------------------------------------------
    w("## 3. Feature importance")
    w("")
    w("Gain and split counts from the primary model. Included so the 97%-empty customer "
      "history block is **visible rather than asserted** — a feature that never splits "
      "is doing nothing, and that shows up here directly.")
    w("")
    gain = primary.booster.feature_importance(importance_type="gain")
    splits = primary.booster.feature_importance(importance_type="split")
    imp = (
        pd.DataFrame(
            {"feature": primary.feature_names, "gain": gain, "splits": splits}
        )
        .sort_values(["gain", "feature"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    total_gain = imp["gain"].sum() or 1.0
    w("| # | Feature | Gain | Gain share | Splits | Monotone |")
    w("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(imp.itertuples(index=False), start=1):
        c = MONOTONE_CONSTRAINTS.get(r.feature, 0)
        mark = {1: "+1", -1: "−1", 0: ""}[c]
        w(f"| {i} | `{r.feature}` | {r.gain:,.1f} | {100 * r.gain / total_gain:.2f}% | "
          f"{int(r.splits):,} | {mark} |")
    w("")
    dead = imp[imp["splits"] == 0]["feature"].tolist()
    cust_dead = [f for f in dead if f.startswith("cust_")]
    w(f"**{len(dead)} of {len(imp)} features never produced a split.**")
    if dead:
        w("")
        for f in dead:
            w(f"- `{f}`")
    if cust_dead:
        w("")
        w(f"{len(cust_dead)} of them are customer-history features. That is the 97% null "
          "rate from `eval/feature_report.md` §4 showing up as model behaviour rather "
          "than as a claim: the booster found nothing to split on because for almost "
          "every row there is no history to split.")
    w("")

    # -- constraints -------------------------------------------------------------------
    w("## 4. Monotonic constraints")
    w("")
    w("Loaded from `models.constraints.MONOTONE_CONSTRAINTS`, a named module constant, "
      "and applied as a vector whose length is asserted equal to the feature count.")
    w("")
    w("| Feature | Constraint | Spec (§5.3) |")
    w("|---|---:|---|")
    for feat, c in MONOTONE_CONSTRAINTS.items():
        note = "**inverted** — see below" if feat in INVERTED_FROM_SPEC else "direct"
        w(f"| `{feat}` | {c:+d} | {note} |")
    w("")
    for feat, why in INVERTED_FROM_SPEC.items():
        w(f"**`{feat}` is inverted relative to §5.3.** {why}. A sign error here would be "
          "worse than no constraint: the model would be forced to learn a backwards "
          "relationship and the SHAP reason strings would confidently tell a merchant "
          "the opposite of the truth — which is the failure mode §5.3 exists to prevent. "
          "Asserted in `tests/test_model.py`.")
    w("")
    for feat, why in OMITTED_FROM_SPEC.items():
        w(f"**`{feat}` omitted.** {why}.")
    w("")
    w(f"Constraint vector length {len(primary.monotone_constraints)} = feature count "
      f"{len(primary.feature_names)}.")
    w("")

    # -- secondary ---------------------------------------------------------------------
    w("## 5. Secondary target — P(fails), benchmark only")
    w("")
    w(f"`{SECONDARY_LABEL}` on all matured orders. Reported as a benchmark and **never "
      "substituted for the primary**. A result here does not transfer: it is a different "
      "estimand on a different population.")
    w("")
    w("| Metric | Value |")
    w("|---|---:|")
    w(f"| Test rows / positives | {sm.n:,} / {sm.n_positives} |")
    w(f"| Prevalence baseline | {sm.prevalence:.4f} |")
    w(f"| **Average precision** | **{sm.average_precision:.4f}** |")
    w(f"| AP − baseline | {sm.ap_over_baseline:+.4f} |")
    w(f"| Lift over baseline | {sm.lift:.2f}× |")
    w(f"| ROC-AUC *(not the headline)* | {sm.roc_auc:.4f} |")
    w(f"| Minimum detectable AP difference | {MDD_SECONDARY:.3f} |")
    w("")
    w("![Precision-recall, secondary](figures/pr_secondary.png)")
    w("")
    w("| Budget | Orders treated | True positives | Precision | Recall | Lift |")
    w("|---|---:|---:|---:|---:|---:|")
    for k, d in sm.at_k.items():
        w(f"| top {k:.0%} | {d['n_selected']:,} | {d['tp']} | "
          f"{pctf(d['precision'], 2)} | {pctf(d['recall'], 2)} | {d['lift']:.2f}× |")
    w("")

    # -- the leak ----------------------------------------------------------------------
    fm = secondary_fixed_metrics["test"]
    w("### That number is inflated by a post-outcome join artifact")
    w("")
    w(f"**Do not read {sm.average_precision:.4f} as the secondary target's performance.** "
      "It is the first number in this repo that looked good, and it looked good for the "
      "wrong reason.")
    w("")
    w(f"{leak_n:,} matured orders have **no rows in `olist_order_items_dataset.csv`**, "
      f"and **{leak_pos:,} of them ({100 * leak_pos / leak_n:.1f}%) are `{SECONDARY_LABEL}` "
      f"positives**. In the test window that is {leak_test_n} orders, all positive — "
      f"**{100 * leak_test_pos / sm.n_positives:.1f}% of the secondary target's "
      f"{sm.n_positives} test positives, identified perfectly by a single artifact.**")
    w("")
    w("The absence is a *consequence* of the outcome, not a checkout-time state. An "
      "order that resolves to `unavailable` has no item rows in the export; at the "
      "moment of checkout the customer had items in the cart and those rows would "
      "exist. So `has_item_rows` — and the null pattern across `n_items`, `order_value`, "
      "`n_sellers`, `n_products`, and `n_missing_features` — encodes the label.")
    w("")
    w("This is leakage the column whitelist could not catch. Every column involved is on "
      "the whitelist and knowable at checkout; what leaks is the **join cardinality**, "
      "which no name-based or perturbation check inspects. Worth stating plainly: the "
      "guardrails in this repo are real and they did not catch this.")
    w("")
    w("**Corrected benchmark**, restricted to orders that join to items:")
    w("")
    w("| | Leaked | Corrected |")
    w("|---|---:|---:|")
    w(f"| Test rows | {sm.n:,} | {fm.n:,} |")
    w(f"| Test positives | {sm.n_positives} | {fm.n_positives} |")
    w(f"| Prevalence | {pctf(sm.prevalence)} | {pctf(fm.prevalence)} |")
    w(f"| **Average precision** | {sm.average_precision:.4f} | "
      f"**{fm.average_precision:.4f}** |")
    w(f"| AP − baseline | {sm.ap_over_baseline:+.4f} | **{fm.ap_over_baseline:+.4f}** |")
    w(f"| Lift | {sm.lift:.2f}× | {fm.lift:.2f}× |")
    w(f"| Precision@1% | {pctf(sm.at_k[0.01]['precision'], 2)} | "
      f"{pctf(fm.at_k[0.01]['precision'], 2)} |")
    w("")
    fixed_resolved = fm.ap_over_baseline > MDD_SECONDARY
    w(("**Corrected, the secondary target still clears its threshold** "
       f"({fm.ap_over_baseline:+.4f} against {MDD_SECONDARY:.3f})."
       if fixed_resolved else
       f"**Corrected, the secondary target no longer clears its threshold** — "
       f"{fm.ap_over_baseline:+.4f} against {MDD_SECONDARY:.3f}. The apparent result was "
       "the artifact."))
    w("")
    w(f"**The primary target is unaffected.** The risk set contains {risk_joins} order "
      "without item rows, because an order that reached a carrier had items. That is "
      f"why the primary's {pm.average_precision:.4f} and the secondary's "
      f"{sm.average_precision:.4f} differ by so much — and the gap was the signal that "
      "something was wrong, not evidence that the broader label is easier.")
    w("")
    w("Carried forward: `has_item_rows` is **not a usable feature on Olist** and the "
      "availability group needs the join-cardinality caveat recorded against it in "
      "`data/COLUMN_WHITELIST.md` before any of this reaches a trained model that ships.")
    w("")

    # -- ROC ---------------------------------------------------------------------------
    w("## 6. ROC — shown alongside, explicitly not the headline")
    w("")
    w("![ROC, primary](figures/roc_primary.png)")
    w("")
    w(f"ROC-AUC is {pm.roc_auc:.4f} on the primary target. At {pctf(pm.prevalence)} "
      "prevalence the curve is dominated by the 19,508 negatives, so it looks "
      "respectable whether or not the model is useful at any budget a merchant would "
      "actually run. It is here because omitting it invites the question, and precision-"
      "recall is the metric §9 reports.")
    w("")

    # -- reproducibility ---------------------------------------------------------------
    w("## 7. Reproducibility")
    w("")
    w("Asserted by refitting the primary model in the same process and comparing:")
    w("")
    w("| Check | Result |")
    w("|---|---|")
    w(f"| Model artifact byte-identical | **{'PASS' if model_identical else 'FAIL'}** |")
    w(f"| Test predictions byte-identical | **{'PASS' if preds_identical else 'FAIL'}** |")
    w("")
    w(f"`deterministic=true`, `force_row_wise=true`, seed `{PARAMS['seed']}`, threads "
      f"pinned to `{PARAMS['num_threads']}`. The thread count is pinned because "
      "LightGBM's determinism guarantee holds for the same data *and the same thread "
      "count*; leaving it to the machine's core count would make the artifact "
      "irreproducible across machines.")
    w("")
    w(f"`{', '.join(DROPPED_CONSTANT_COLUMNS)}` is dropped from the trained matrix: it "
      "is constant on this panel, cannot produce a split, and its presence in a feature "
      "list overstates the contract. It remains in the feature contract labelled a "
      "no-op, per the §4.2 rule.")
    w("")
    w("### Column-order contract")
    w("")
    w("The trained column order is recorded in `artifacts/models/*/contract.json`, and "
      "`predict()` requires an exact match. A frame with the right columns in the wrong "
      "order is **rejected, not reordered** — LightGBM addresses features positionally "
      "and so do the monotone constraints, so scoring a permuted frame would apply the "
      "pincode constraint to whatever feature now sits at that index, silently.")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0 if (model_identical and preds_identical) else 1


if __name__ == "__main__":
    raise SystemExit(main())
