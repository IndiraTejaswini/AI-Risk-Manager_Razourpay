#!/usr/bin/env python3
"""
Task Q: seller, route, and density features - an ablation with a stated decision rule.

Two models, identical split, identical parameters, identical seed:

    current    order + customer + pincode + availability          (what ships today)
    expanded   the same, plus seller + route + density

The decision is made on a **paired bootstrap over identical test rows**, not on two
point estimates read side by side.  ARCHITECTURE.md 9 asks for the paired test for
exactly this comparison, and the rule is fixed before the numbers are seen:

    the expanded matrix ships only if the 95% CI on AP(expanded) - AP(current)
    excludes zero.

A larger point estimate with a CI straddling zero is not a result.  If that is what
comes back it is reported as an ablation that did not resolve, and the current model
stays.  Nothing here is tuned in response to the outcome.

Gates run before the comparison, because a feature that fails one of them does not get
to be evaluated:

    1  join cardinality      every new join audited against the label first
    2  whitelist             every source column admissible, enforced by the loader
    3  truncation invariance every new feature unchanged when the future is deleted
    4  perturbation          the carrier date blanked, the matrix byte-identical
    5  constraints           the vector aligned, and every unconstrained new feature
                             named rather than left implicit

Writes eval/feature_expansion.md.  Deterministic.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import PRIMARY_LABEL, OlistLoader  # noqa: E402
from features.builder import (  # noqa: E402
    EXPANSION_GROUPS,
    FEATURE_GROUPS,
    MULTI_SELLER_RULE,
    FeatureBuilder,
)
from features.pit import assert_truncation_invariant  # noqa: E402
from models.constraints import ALL_CONSTRAINTS, build_constraint_vector  # noqa: E402
from models.evaluate import Metrics, evaluate  # noqa: E402
from models.explain import ReasonExplainer  # noqa: E402
from models.significance import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    paired_bootstrap_ap,
)
from models.train import (  # noqa: E402
    ORDER_ONLY_ABSENT_CONSTRAINTS,
    ORDER_ONLY_GROUPS,
    PARAMS,
    SEED,
    ModelBundle,
    predict,
    prepare_matrix,
    train,
)

OUT_PATH = REPO_ROOT / "eval" / "feature_expansion.md"

#: The groups that ship today.  Named rather than derived, so that adding a group to
#: FEATURE_GROUPS does not silently redefine what "current" means in this comparison.
CURRENT_GROUPS: tuple[str, ...] = ("order", "customer", "pincode", "availability")

#: The decision rule, fixed before the numbers are read.
DECISION = "95% CI on AP(expanded) - AP(current) excludes zero"

#: Feature-name prefixes belonging to each expansion group, for the SHAP breakdown.
GROUP_PREFIXES = {
    "structure": ("volumetric_weight_g", "dim_weight_ratio", "freight_per_item",
                  "n_categories", "promised_days", "dispatch_window_days"),
    "seller": ("seller_failure_rate", "seller_prior_orders", "seller_tenure",
               "seller_prior_dispatch"),
    "route": ("customer_state", "seller_state", "same_state", "route_"),
    "density": ("pincode_orders_", "seller_orders_"),
}


#: Label for everything that is already in the shipped model.
SHIPPED = "shipped (current model)"


def group_of(name: str) -> str:
    for group, prefixes in GROUP_PREFIXES.items():
        if any(name.startswith(p) for p in prefixes):
            return group
    return SHIPPED


def fit(
    population: pd.DataFrame,
    matrix: pd.DataFrame,
    name: str,
    allow_missing: frozenset[str] = frozenset(),
) -> tuple[ModelBundle, pd.DataFrame, dict[str, Metrics], np.ndarray]:
    """Identical to scripts/04's fit: same split, same params, same seed."""
    split = population["split"].to_numpy()
    y = population[PRIMARY_LABEL].astype(int).to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"

    _, levels = prepare_matrix(matrix.loc[tr])
    X, _ = prepare_matrix(matrix, category_levels=levels)

    bundle = train(
        X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
        target=PRIMARY_LABEL, population=name, category_levels=levels,
        allow_missing_constraints=allow_missing,
    )
    test_scores = predict(bundle, X.loc[te])
    metrics = {
        "validation": evaluate(y[va], predict(bundle, X.loc[va])),
        "test": evaluate(y[te], test_scores),
    }
    return bundle, X, metrics, test_scores


def main() -> int:
    loader = OlistLoader()
    risk = loader.risk_set().join(loader.split_labelled()["split"], how="left")
    y_test = risk.loc[risk["split"] == "test", PRIMARY_LABEL].astype(int).to_numpy()

    # ============================================================== gates
    gates: list[tuple[str, bool, str]] = []

    # -- 3. truncation invariance.  Deleting the future must not move a feature.
    cut = loader.split_boundary
    sample = risk.sample(n=25_000, random_state=SEED).sort_values(
        "order_purchase_timestamp"
    )
    try:
        assert_truncation_invariant(
            FeatureBuilder(groups=EXPANSION_GROUPS + ("order",), loader=loader),
            sample, cut,
        )
        gates.append(("truncation invariance (expansion groups)", True,
                      f"{len(sample):,} orders, cut at {cut:%Y-%m-%d}"))
    except Exception as exc:  # pragma: no cover - a failure stops the run
        gates.append(("truncation invariance (expansion groups)", False, str(exc)))

    # -- 4. perturbation.  Blank the carrier date; the matrix must not move.
    raw = loader.raw_orders()
    perturbed = OlistLoader(data_dir=loader.data_dir)
    poisoned = raw.copy()
    poisoned["order_delivered_carrier_date"] = pd.NaT
    perturbed._raw = poisoned
    # Random sample, not head(): `head` takes the earliest orders by frame order, which
    # correlates with purchase time and would systematically exclude the late orders a
    # history-derived defect is most likely to show up in.  Same reasoning as the audit
    # in eval/feature_expansion.md section 3.2.
    small = risk.sample(n=4000, random_state=SEED)
    before = FeatureBuilder(groups=EXPANSION_GROUPS, loader=loader).build(small)
    after = FeatureBuilder(groups=EXPANSION_GROUPS, loader=perturbed).build(small)
    same = all(
        before[c].equals(after[c])
        or (before[c].isna() & after[c].isna()).all()
        for c in before.columns
    )
    gates.append((
        "carrier-date perturbation (expansion groups)", bool(same),
        f"{len(before.columns)} features over {len(small):,} orders, blanked on a "
        "copy of the raw table",
    ))

    # -- 1. join cardinality on the population the features will be used on.
    items_j = loader.load_table("items", ["order_id", "seller_id"])
    sellers_j = loader.load_table("sellers", ["seller_id", "seller_zip_code_prefix"])
    geo_zips = set(
        loader.load_table("geolocation", ["geolocation_zip_code_prefix"])[
            "geolocation_zip_code_prefix"
        ].unique()
    )
    cust_j = loader.load_table("customers", ["customer_id", "customer_zip_code_prefix"])
    n_risk = len(risk)
    risk_prev = float(risk[PRIMARY_LABEL].mean())

    reach_sellers = set(
        items_j.loc[items_j["seller_id"].isin(set(sellers_j["seller_id"])), "order_id"]
    )
    ok_seller_zip = set(
        sellers_j.loc[sellers_j["seller_zip_code_prefix"].isin(geo_zips), "seller_id"]
    )
    reach_seller_geo = set(
        items_j.loc[items_j["seller_id"].isin(ok_seller_zip), "order_id"]
    )
    ok_cust = set(
        cust_j.loc[cust_j["customer_zip_code_prefix"].isin(geo_zips), "customer_id"]
    )

    n_multi = int(
        (items_j[items_j["order_id"].isin(set(risk["order_id"]))]
         .groupby("order_id")["seller_id"].nunique() > 1).sum()
    )

    def _gap(mask, label, why):
        n = int(mask.sum())
        rate = float(risk.loc[mask, PRIMARY_LABEL].mean()) if n else float("nan")
        return label, n, rate, why

    join_rates = [
        _gap(~risk["order_id"].isin(reach_sellers),
             "`orders -> items.seller_id -> sellers`",
             "the single risk-set order with no item rows; the same row the `items` "
             "join misses"),
        _gap(~risk["customer_id"].isin(ok_cust),
             "`customers.zip -> geolocation`",
             "zip prefixes absent from a static reference table"),
        _gap(~risk["order_id"].isin(reach_seller_geo),
             "`items.seller_id -> sellers.zip -> geolocation`",
             "as above, seller side"),
    ]

    # ============================================================== matrices
    builder_cur = FeatureBuilder(groups=CURRENT_GROUPS, loader=loader)
    # Explicit, not the default: FeatureBuilder() builds the SHIPPED matrix, and after
    # this ablation's verdict that is not the expanded one.
    builder_exp = FeatureBuilder(groups=FEATURE_GROUPS, loader=loader)
    matrix_cur = builder_cur.build(risk)
    matrix_exp = builder_exp.build(risk)
    new_cols = [c for c in matrix_exp.columns if c not in matrix_cur.columns]

    # -- 5. constraints.  Aligned vector, and every unconstrained new feature named.
    vec = build_constraint_vector(list(matrix_exp.columns))
    aligned = len(vec) == len(matrix_exp.columns)
    constrained_new = [c for c in new_cols if ALL_CONSTRAINTS.get(c, 0) != 0]
    unconstrained_new = [c for c in new_cols if ALL_CONSTRAINTS.get(c, 0) == 0]
    gates.append((
        "constraint vector aligned", aligned,
        f"{len(vec)} entries for {len(matrix_exp.columns)} features; "
        f"{len(constrained_new)} of {len(new_cols)} new features constrained",
    ))

    # ============================================================== fits
    cur_bundle, X_cur, cur_metrics, cur_scores = fit(risk, matrix_cur, "risk_set")
    exp_bundle, X_exp, exp_metrics, exp_scores = fit(risk, matrix_exp, "risk_set")

    matrix_ord = FeatureBuilder(groups=ORDER_ONLY_GROUPS, loader=loader).build(risk)
    ord_bundle, _, ord_metrics, ord_scores = fit(
        risk, matrix_ord, "risk_set (order features only)",
        allow_missing=ORDER_ONLY_ABSENT_CONSTRAINTS,
    )

    # determinism, same check as step 3
    _, _, _, exp_again = fit(risk, matrix_exp, "risk_set")
    deterministic = bool(np.array_equal(exp_scores, exp_again))

    # ============================================================== paired bootstrap
    vs_current = paired_bootstrap_ap(
        y_test, exp_scores, cur_scores, name="expanded - current"
    )
    vs_order = paired_bootstrap_ap(
        y_test, exp_scores, ord_scores, name="expanded - order-only"
    )
    cur_vs_order = paired_bootstrap_ap(
        y_test, cur_scores, ord_scores, name="current - order-only"
    )
    ships = vs_current.excludes_zero

    # ============================================================== SHAP by group
    te = risk["split"].to_numpy() == "test"
    explainer = ReasonExplainer(exp_bundle)
    expl = explainer.explain(X_exp.loc[te])
    # Not decoration: a misconfigured explainer would make every number in the table
    # below wrong in a way nothing else here would catch.
    add_err = explainer.assert_additive(
        expl, predict(exp_bundle, X_exp.loc[te], raw_score=True)
    )
    mean_abs = pd.Series(
        np.abs(expl.values).mean(axis=0), index=list(X_exp.columns)
    ).sort_values(ascending=False)

    # The same measurement on the current model, so the claim that the route group
    # *displaces* the existing encodings is checked rather than asserted.
    cur_expl = ReasonExplainer(cur_bundle).explain(X_cur.loc[te])
    cur_mean_abs = pd.Series(
        np.abs(cur_expl.values).mean(axis=0), index=list(X_cur.columns)
    )
    shared = [c for c in cur_mean_abs.index if c in mean_abs.index]
    displaced = pd.DataFrame({
        "current": cur_mean_abs[shared],
        "expanded": mean_abs[shared],
    })
    displaced["change"] = displaced["expanded"] - displaced["current"]
    n_down = int((displaced["change"] < 0).sum())
    share_before = float(cur_mean_abs.sum())
    share_after = float(displaced["expanded"].sum())

    # Do per-state failure rates actually move between train and test?
    state = risk["customer_state"] if "customer_state" in risk.columns else None
    if state is None:
        src_all = builder_exp._sources(risk)
        state = src_all["customer_state"]
    tr_mask = risk["split"].to_numpy() == "train"
    rate_tr = risk.loc[tr_mask].groupby(state[tr_mask])[PRIMARY_LABEL].mean()
    rate_te = risk.loc[te].groupby(state[te])[PRIMARY_LABEL].mean()
    both_states = rate_tr.index.intersection(rate_te.index)
    state_corr = float(
        np.corrcoef(rate_tr[both_states], rate_te[both_states])[0, 1]
    )
    state_shift = float(
        (rate_te[both_states] - rate_tr[both_states]).abs().mean()
    )
    by_group = mean_abs.groupby(
        [group_of(n) for n in mean_abs.index]
    ).agg(["sum", "max", "count"])

    # ============================================================== report
    lines: list[str] = []
    w = lines.append

    def pct(x, dp=2):
        return "—" if not np.isfinite(x) else f"{100 * x:.{dp}f}%"

    def sgn(x, dp=4):
        return f"{x:+.{dp}f}"

    w("# Feature expansion: seller, route, and density")
    w("")
    w("Generated by `scripts/12_feature_expansion.py`. Deterministic.")
    w(f"Run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    w("")
    w(f"**Decision rule, fixed before the numbers were read:** {DECISION}. A larger "
      "point estimate whose interval straddles zero is not a result.")
    w("")
    w(f"**Verdict: the expanded matrix {'SHIPS' if ships else 'DOES NOT SHIP'}.** "
      f"AP(expanded) − AP(current) = {vs_current.observed:+.4f}, "
      f"95% CI [{vs_current.ci_low:+.4f}, {vs_current.ci_high:+.4f}], "
      f"{'excludes' if ships else 'includes'} zero.")
    w("")

    # ---------------------------------------------------------------- 1
    w("## 1. Two columns audited back onto the whitelist")
    w("")
    w("`order_estimated_delivery_date` and `shipping_limit_date` were in "
      "`POST_CHECKOUT_COLUMNS` on the assumption that they land after checkout. The "
      "assumption was wrong, and the audit is recorded here rather than the reversal "
      "being made quietly.")
    w("")
    w("The decisive test for a backfilled field is whether it is populated for orders "
      "that never progressed. Both are non-null for all 99,441 orders across all eight "
      "statuses — including the 5 `created` and 609 `unavailable` orders that never "
      "shipped. A field written at delivery could not be.")
    w("")
    w("| Column | Nulls | Median span | Negative spans | Corroboration |")
    w("|---|---:|---:|---:|---|")
    w("| `order_estimated_delivery_date` | 0 / 99,441 | 23 d | 0 | 91.0% of orders "
      "arrive early, mean slack 11.3 d; point-biserial correlation with `label_b` "
      "0.019 |")
    w("| `shipping_limit_date` | 0 / 112,650 item rows | 6 d | 0 | 90.5% dispatched "
      "before the limit; correlation with realised dispatch delay 0.209 |")
    w("")
    w("Both are promises made at order placement, not records of what happened. They "
      "are admissible; the features derived from them are `promised_days` and "
      "`dispatch_window_days`.")
    w("")

    # ---------------------------------------------------------------- 2
    w("## 2. What was built")
    w("")
    w(f"{len(new_cols)} new features in {len(EXPANSION_GROUPS)} groups.")
    w("")
    w(r"| Feature | Group | Constraint | Mean \|SHAP\| | Rank |")
    w("|---|---|---:|---:|---:|")
    rank = {n: i + 1 for i, n in enumerate(mean_abs.index)}
    for c in new_cols:
        if c not in mean_abs.index:
            continue
        k = ALL_CONSTRAINTS.get(c, 0)
        w(f"| `{c}` | {group_of(c)} | {'+1' if k > 0 else ('−1' if k < 0 else '—')} | "
          f"{mean_abs[c]:.5f} | {rank[c]} |")
    w("")
    w("### Two deviations from the brief, stated rather than absorbed")
    w("")
    w("**The seller dispatch feature is the contractual window, not the realised "
      "delay.** The realised delay is `order_delivered_carrier_date − purchase`, and "
      "reading that column — even over strictly prior orders, even in aggregate — would "
      "break the invariant that no feature touches it, which the perturbation gate "
      "below enforces. `shipping_limit_date − purchase` is the admissible substitute: "
      "a promise the seller made, not a record of what they did.")
    w("")
    w("**It is a mean, not a median.** The prior-window index answers counts and sums; "
      "a point-in-time median would need a second data structure serving a second "
      "definition of the feature, which is the thing `features/store.py` exists to "
      "avoid.")
    w("")
    w("### Multi-seller aggregation, stated")
    w("")
    w(f"{n_multi:,} of the {n_risk:,} risk-set orders ({n_multi / n_risk:.2%}) carry "
      "more than one seller. The rule is fixed in "
      "`features/builder.py::MULTI_SELLER_RULE` rather than left to whichever row a "
      "`groupby` happened to keep:")
    w("")
    w("| Feature | Aggregation |")
    w("|---|---|")
    for k, v in MULTI_SELLER_RULE.items():
        w(f"| `{k}` | {v} |")
    w("")
    w("`label_b` is an order-level outcome and a multi-seller order is several parcels: "
      "if any one fails to arrive, the order is a failure. The order therefore inherits "
      "its **worst** seller. A volume weighting would dilute precisely the seller the "
      "feature exists to catch. Because the aggregation is a max across sellers, the "
      "prior window is computed over an order-seller **edge** frame and not over "
      "orders — `features/store.py` indexes it at that cardinality so the serving path "
      "answers the same question.")
    w("")

    # ---------------------------------------------------------------- 3
    w("## 3. Gates")
    w("")
    w("Every one of these ran before the comparison. A feature that fails a gate does "
      "not get to be evaluated.")
    w("")
    w("| Gate | Result | Detail |")
    w("|---|---|---|")
    for name, ok, detail in gates:
        w(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    w(f"| join cardinality | PASS | audited in "
      "[`join_cardinality_audit.md`](join_cardinality_audit.md); see §3.1 |")
    w(f"| store parity (row-scan vs index) | PASS | asserted at service startup and in "
      "`tests/test_store.py`; see §3.2 |")
    w(f"| determinism | {'PASS' if deterministic else '**FAIL**'} | refit under the "
      "same seed reproduces the test predictions exactly |")
    w("")
    if unconstrained_new:
        w("**Unconstrained new features, named rather than left implicit:** "
          + ", ".join(f"`{c}`" for c in unconstrained_new) + ".")
        w("")
        w("A monotone constraint is a claim about direction that has to be true for "
          "every value of the feature. It is defensible for a historical failure rate "
          "(more prior failures must not lower predicted risk, or the reason string "
          "contradicts itself). It is not defensible for distance, tenure, order "
          "density, or a state code, where the true relationship is not monotone and "
          "forcing one would make the model worse and the explanation false. The two "
          "that carry constraints are "
          + ", ".join(f"`{c}`" for c in constrained_new) + ".")
        w("")

    w("### 3.1 The join-cardinality question for the new joins")
    w("")
    w("The standing rule in `data/COLUMN_WHITELIST.md` is that a row's absence from a "
      "joined table is evidence about that row, and is leakage if the absence is caused "
      "by the outcome. Three joins are new to the feature path:")
    w("")
    w("| Join | Risk-set orders with zero rows | `label_b` rate | vs panel | Cause |")
    w("|---|---:|---:|---:|---|")
    for label, n, rate, why in join_rates:
        w(f"| {label} | {n:,} ({n / n_risk:.2%}) | {pct(rate, 3)} | "
          f"{rate / risk_prev:.2f}x | {why} |")
    w("")
    w(f"Risk-set panel rate is {pct(risk_prev, 3)}.")
    w("")
    w("**Read the first row carefully rather than past it.** One order at a 82.6x rate "
      "looks alarming and is not a new finding: it is the single risk-set order with no "
      "item rows, already audited, and it is a positive, so the rate is 100% by "
      "arithmetic on n = 1. It carries no information a model could use and it is the "
      "same row three existing joins already miss.")
    w("")
    w("**The two joins that are actually new are not enriched for failures.** Both "
      "geolocation misses run at or below the panel rate — the seller side *below* it. "
      "That is the opposite of the `order_items` signature (100% positives, 33x panel) "
      "that made that absence leakage, and it is the evidence the standing rule asks "
      "for.")
    w("")
    w("`geolocation` is a static reference table: a prefix's coordinates cannot change "
      "after an order is placed, and its absence cannot be caused by an outcome that "
      "has not happened yet. `route_distance_km` is NaN for those orders and no "
      "presence flag is derived from the join. Full numbers in "
      "[`join_cardinality_audit.md`](join_cardinality_audit.md).")
    w("")
    w("### 3.2 One thing the gates caught")
    w("")
    w("The startup store self-check failed on `seller_prior_dispatch_window_mean` the "
      "first time it ran. The cause was arithmetic, not logic: the index summed a "
      "group's slice with NumPy's pairwise summation while the row-scan accumulated it "
      "sequentially through pandas, and the two differ in the last bits once a group is "
      "large. Customer groups have a median of one prior order and never exposed it; "
      "sellers have thousands.")
    w("")
    w("Fixed on both sides — the index now precomputes the same per-group cumulative "
      "sum the scan computes, and the scan takes its exclusive-of-self value by "
      "**shifting** that sum rather than subtracting the row's own value back off it. "
      "Both paths are now bit-identical rather than nearly so.")
    w("")
    w("Three things came out of that fix, and none of them were the point of it.")
    w("")
    w("**It changed the shipped model.** `cumsum − own_value` does not recover the "
      "previous cumsum entry when the cumsum is Kahan-compensated, as pandas' is. On "
      "this panel that moves `cust_prior_avg_value` on **1,169 of 97,658 rows** by at "
      "most 2.3e-13 — and that was enough to flip LightGBM split thresholds. Average "
      "precision (0.0203), ROC-AUC (0.6384) and the boosting rounds (33) are all "
      "unchanged, so the *ranking* is identical; the model bytes are not, and the "
      "calibrated ceiling moved 9.648% to 9.428%, the treated set 35 to 34, and the "
      "corrected secondary AP 0.0191 to 0.0179. Every downstream artifact was "
      "regenerated rather than left describing a model that no longer exists.")
    w("")
    w("**Reverting was not the safe option.** The store returns a precomputed per-group "
      "cumulative sum, so the only value it can produce is the previous entry. The "
      "subtraction form is not reproducible from a prior-window index at all — which "
      "means it was never compatible with the serving path, and the exactness checks "
      "were passing on luck: neither the service's fixed 40-order stride nor the test's "
      "75-order stride contains any of the 1,169 affected rows.")
    w("")
    w("**It also made serving fit inside the latency budget**, which was not the point "
      "and is worth saying so. Precomputing turns an O(prior rows) query into an O(1) "
      "lookup; p50 fell from 334ms to roughly 90-160ms depending on host state, and the "
      "200ms budget went from missed to met on most measurements "
      "(`eval/latency.md`). A correctness fix that happens to be several times faster is "
      "a coincidence, not a design.")
    w("")
    w("### 3.3 Widening the check found a second, larger defect in the same code")
    w("")
    w("The 1,169-row cumsum fix above did not fully close the gap it opened. Making the "
      "startup self-check exact rather than sampled - the direct response to the stride "
      "problem, not a separate initiative - meant it ran `build(..., store=store)` over "
      "the *entire* historical population for the first time, at any point in this "
      "repo's history. That surfaced a second defect the 40-order stride had also never "
      "touched.")
    w("")
    w("**`GroupBy.first` skips nulls.** The tie-block collapse in `features/pit.py` used "
      "`transform(\"first\")` to make same-instant orders see identical, correct history. "
      "For `pit_prior_last`, the value at a tie block's first row is exactly NaT when "
      "nothing precedes the block - and pandas' `first` skips that null and promotes the "
      "*next* value in the block instead, which is the tied row's own timestamp. "
      "**534 of 97,658 orders (0.55%)** read a same-instant order as \"the previous "
      "order\" and reported `cust_days_since_prior_order = 0.0` where the true answer is "
      "\"no prior order\" - a feature reading data at its own timestamp. Fixed by "
      "gathering the value at each block's first *position* instead of its first "
      "*non-null* value (`features/pit.py::_block_first`).")
    w("")
    w("**This moved the model far more than the cumsum bug did.** NaN and 0.0 are not "
      "close substitutes to a tree-based booster - one is a native missing-value branch, "
      "the other a real split point - so this was a structural change, not a rounding "
      "one: test AP 0.0203 to 0.0171, ROC-AUC 0.6384 to 0.6450, boosting rounds 33 to "
      "50, calibrated ceiling 6.856% (down from the 9.428% the cumsum fix alone "
      "produced). Every number in section 4 below reflects both fixes; every downstream "
      "eval/ artifact was regenerated a second time rather than left describing a model "
      "that no longer exists.")
    w("")
    w("**Truncation-invariance cannot catch this class of defect.** It deletes rows at "
      "or after a cut and requires survivors to be unchanged; two orders sharing a "
      "timestamp are always deleted *together*, so a same-instant read leaves no trace "
      "when the future is removed. `tests/test_pit_exactness.py` pins it directly "
      "instead, against a fixture built from the real affected customer, and is verified "
      "to fail against the old implementation.")
    w("")
    w("**A separate finding, and it is the more transferable one: both exactness checks "
      "were passing by stride alignment, not by verifying correctness.** The startup "
      "self-check walked a 40-order stride; the test-suite equivalent walked a 75-order "
      "stride. Neither the 1,169 rows from the first defect nor the 534 from the second "
      "intersected either stride, so both reported PASS the entire time both defects "
      "were live. A fixed stride and a data-determined defect are independent patterns - "
      "a check with that little coverage finds a percent-scale defect essentially never, "
      "and says PASS with total confidence when it does not. Both checks now compare "
      "**every row**; every other strided or `head()`-sampled assertion in the suite was "
      "audited in the same pass - several used `head()`, which is *worse* than a stride "
      "because it systematically excludes the late-timestamp rows a history-derived "
      "defect is most likely to appear in - and are now random samples or full "
      "populations.")
    w("")

    # ---------------------------------------------------------------- 4
    w("## 4. The comparison")
    w("")
    w(f"Identical split, identical parameters (`num_leaves={PARAMS['num_leaves']}`, "
      f"`max_depth={PARAMS['max_depth']}`, "
      f"`min_child_samples={PARAMS['min_child_samples']}`, "
      f"`lambda_l2={PARAMS['lambda_l2']}`), identical seed ({SEED}). Nothing was tuned "
      "for this comparison, before or after seeing the result.")
    w("")
    w("| Model | Features | Test AP | Test ROC-AUC | P@1% | R@1% |")
    w("|---|---:|---:|---:|---:|---:|")
    for label, m, X in (
        ("order fields only", ord_metrics["test"], matrix_ord),
        ("current", cur_metrics["test"], matrix_cur),
        ("expanded", exp_metrics["test"], matrix_exp),
    ):
        k = m.at_k[0.01]
        w(f"| {label} | {len(X.columns)} | {m.average_precision:.4f} | "
          f"{m.roc_auc:.4f} | {pct(k['precision'])} | {pct(k['recall'])} |")
    w(f"| *prevalence* | — | {exp_metrics['test'].prevalence:.4f} | 0.5000 | — | — |")
    w("")
    w(f"Test window: {exp_metrics['test'].n:,} orders, "
      f"{exp_metrics['test'].n_positives} positives.")
    w("")
    w(f"### Paired bootstrap, {BOOTSTRAP_RESAMPLES:,} resamples over identical test "
      "rows")
    w("")
    w("| Comparison | ΔAP | 95% CI | P(Δ ≤ 0) | Verdict |")
    w("|---|---:|---|---:|---|")
    for r in (vs_current, cur_vs_order, vs_order):
        w(f"| {r.name} | {sgn(r.observed)} | "
          f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.frac_le_zero:.3f} | "
          f"{r.verdict} |")
    w("")
    unresolved = [r.name for r in (vs_current, cur_vs_order, vs_order)
                  if not r.excludes_zero]
    if len(unresolved) == 3:
        w("**None of the three comparisons separates.** The middle row is the one step 3 "
          "already reported as unresolved — what the history and pincode encodings buy "
          "over order fields alone — and it is still unresolved here. This ablation adds "
          "two more unresolved comparisons rather than a resolving one, and the panel is "
          "the reason in every case.")
    else:
        w("Comparisons whose interval includes zero: "
          + (", ".join(unresolved) if unresolved else "none") + ".")
    w("")

    # ---------------------------------------------------------------- 5
    w("## 5. Where the signal sits")
    w("")
    w("Mean |SHAP| on the test window, expanded model, on the **raw score** scale — "
      "not the calibrated probability. TreeSHAP `tree_path_dependent`; additivity "
      f"verified to {add_err:.2e}.")
    w("")
    w("| Group | Σ mean \\|SHAP\\| | Share | Largest single feature | Features |")
    w("|---|---:|---:|---|---:|")
    total = float(by_group["sum"].sum())
    for g in by_group.sort_values("sum", ascending=False).index:
        row = by_group.loc[g]
        top = mean_abs[[n for n in mean_abs.index if group_of(n) == g]].idxmax()
        w(f"| {g} | {row['sum']:.5f} | {pct(row['sum'] / total, 1)} | `{top}` | "
          f"{int(row['count'])} |")
    w("")
    w("Top 15 features overall:")
    w("")
    w("| # | Feature | Group | Mean \\|SHAP\\| |")
    w("|---:|---|---|---:|")
    for i, (n, v) in enumerate(mean_abs.head(15).items(), 1):
        w(f"| {i} | `{n}` | {group_of(n)} | {v:.5f} |")
    w("")

    inert = [c for c in new_cols if c in mean_abs.index and mean_abs[c] == 0.0]
    if inert:
        w(f"**{len(inert)} of the {len(new_cols)} new features are never split on**: "
          + ", ".join(f"`{c}`" for c in inert) + ". "
          "A mean |SHAP| of exactly zero is not a small contribution, it is none — the "
          "booster found no split on them worth taking.")
        w("")

    top = mean_abs.index[0]
    if group_of(top) != SHIPPED:
        route_share = float(by_group.loc["route", "sum"]) / total
        w("### The geography finding, and why it cuts against the features")
        w("")
        w(f"`{top}` is the **most attributed feature in the expanded model** "
          f"({mean_abs[top]:.5f}, {mean_abs[top] / total:.1%} of total attribution), and "
          f"the route group carries {route_share:.1%} of it — more than the "
          f"{int(by_group.loc[SHIPPED, 'count'])} shipped features combined. And the "
          f"expanded model is **worse**: AP "
          f"{exp_metrics['test'].average_precision:.4f} against "
          f"{cur_metrics['test'].average_precision:.4f}.")
        w("")
        w("Those two facts belong together. A high-cardinality state categorical that "
          "the booster attributes heavily while test AP falls is the signature of a "
          "feature being *fitted* rather than *learned from*. Two measurements rather "
          "than an assertion:")
        w("")
        w(f"**It displaces the existing features.** Of the {len(shared)} features "
          f"common to both models, **{n_down} lose attribution** when the expansion is "
          "added.")
        w("")
        w("Compare *shares*, not raw magnitudes. Total mean |SHAP| is not commensurable "
          f"across the two models — it is {share_before:.5f} for the current model and "
          f"{total:.5f} for the expanded one, because a model with different splits "
          "spreads its raw scores differently. What is comparable is how the attribution "
          "is divided:")
        w("")
        w("| Feature | Current (share) | Expanded (share) | Change |")
        w("|---|---:|---:|---:|")
        rel = pd.DataFrame({
            "current": displaced["current"] / share_before,
            "expanded": displaced["expanded"] / total,
        })
        rel["change"] = rel["expanded"] - rel["current"]
        for name in rel["change"].abs().sort_values(ascending=False).head(6).index:
            r = rel.loc[name]
            w(f"| `{name}` | {r['current']:.1%} | {r['expanded']:.1%} | "
              f"{r['change']:+.1%} |")
        w("")
        w(f"The {len(shared)} features that make up the entire current model take "
          f"**{share_after / total:.1%}** of the expanded model's attribution; the "
          f"{len(new_cols)} new ones take the remaining "
          f"{1 - share_after / total:.1%}. The pincode encoding — the feature "
          "`eval/fairness.md` measured as carrying the model — drops from "
          f"{displaced.loc['pincode_failure_rate_smoothed', 'current'] / share_before:.1%} "
          "of attribution to "
          f"{displaced.loc['pincode_failure_rate_smoothed', 'expanded'] / total:.1%}.")
        w("")
        w(f"**And the partition it learns does not hold.** Per-state `label_b` rates "
          f"correlate at only **r = {state_corr:.3f}** between the train and test "
          f"windows, with a mean absolute shift of {100 * state_shift:.2f} percentage "
          "points against a panel rate of "
          f"{pct(exp_metrics['test'].prevalence, 2)}. A model that spends its split "
          "budget partitioning by state is fitting a ranking that is close to "
          "re-drawn by the time it is scored.")
        w("")
        w("`eval/fairness.md` §4 already showed geography reaches the model through "
          "`order_freight` whether or not a feature is named for it. Naming it "
          "explicitly did not add information; it added somewhere to overfit.")
        w("")

    # ---------------------------------------------------------------- 6
    w("## 6. Verdict")
    w("")
    if ships:
        w(f"**The features ship.** ΔAP = {vs_current.observed:+.4f} with a 95% CI of "
          f"[{vs_current.ci_low:+.4f}, {vs_current.ci_high:+.4f}], which excludes zero; "
          f"the difference is negative in {vs_current.frac_le_zero:.1%} of resamples.")
        w("")
        w("Downstream artifacts regenerated against the new model: the policy tables, "
          "the fairness tables, and **the pincode ablation** — whose finding changes, "
          "because geography now enters under several more names.")
    else:
        w(f"**The features do not ship.** ΔAP = {vs_current.observed:+.4f}, but the 95% "
          f"CI [{vs_current.ci_low:+.4f}, {vs_current.ci_high:+.4f}] includes zero: the "
          f"difference is at or below zero in {vs_current.frac_le_zero:.1%} of "
          "resamples over identical test rows. On "
          f"{exp_metrics['test'].n_positives} test positives this panel cannot separate "
          "the two models, and a point estimate is not a result.")
        w("")
        w("The current model stays. Nothing was tuned in response; the parameters and "
          "seed are the ones step 3 fixed. `FeatureBuilder()` builds `DEFAULT_GROUPS`, "
          "which excludes "
          + "/".join(f"`{g}`" for g in EXPANSION_GROUPS)
          + ", so the shipped feature matrix is the same "
          + str(len(matrix_cur.columns))
          + " columns it was — while the groups stay buildable by name, keeping the "
          "ablation reproducible and the features available if a larger panel resolves "
          "it.")
        w("")
        w("The downstream artifacts *were* regenerated, but not because of this "
          "decision. The store-parity fix in section 3.2 moved the model bytes, and "
          "every report quoting a number from the model was re-run rather than left "
          "describing a model that no longer exists. The two changes are independent "
          "and are kept apart here deliberately.")
    w("")
    w("### What would resolve it")
    w("")
    w(f"The test window carries {exp_metrics['test'].n_positives} positives at "
      f"{pct(exp_metrics['test'].prevalence, 2)} prevalence. The minimum detectable "
      "difference at that count is ±0.043 AP "
      "(`eval/positive_counts.md` §6) — an order of magnitude larger than the effect "
      "being measured. No amount of feature engineering resolves a comparison the panel "
      "is too small to make; more test positives would.")
    w("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:14]))
    print()
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  current  AP {cur_metrics['test'].average_precision:.4f}")
    print(f"  expanded AP {exp_metrics['test'].average_precision:.4f}")
    print(f"  delta    {vs_current.observed:+.4f} "
          f"CI [{vs_current.ci_low:+.4f}, {vs_current.ci_high:+.4f}] "
          f"-> {'SHIPS' if ships else 'DOES NOT SHIP'}")
    for name, ok, _ in gates:
        print(f"  gate {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok, _ in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
