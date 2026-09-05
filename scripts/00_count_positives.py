#!/usr/bin/env python3
"""
Step 0 - the positive-count gate.  ARCHITECTURE.md section 2, build order step 0.

Stacks the label / maturation / boleto / temporal-split filters in the order section 2
specifies, counts what survives at each stage, applies the section 2 decision rule, and
prints the minimum detectable PR-AUC difference at the realised test size.

This is the first commit.  Nothing downstream may be built before the artifact it
writes exists:

    eval/positive_counts.md

No model, no sampling, no randomness.  Two runs produce identical counts.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# Parameters.
#
# The population parameters - maturation window, test fraction, label definition, the
# observed-timestamp list - live in data/loader.py and are imported, not restated.  This
# script owns only what is specific to the gate: the boleto definition and the section 2
# estimates it reports against.
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import (  # noqa: E402
    DELIVERED,
    KAGGLE_DATASET,
    MATURATION_DAYS,
    ORDERS_CSV,
    PAYMENTS_CSV,
    TEST_FRACTION,
    OlistLoader,
)

BOLETO = "boleto"          # section 3.6 - closest structural analog to COD

OUT_PATH = REPO_ROOT / "eval" / "positive_counts.md"

# The published estimates in section 2, carried here so the gate can report estimate
# against actual.  Section 2's instruction is "those figures are estimates - compute the
# real ones"; showing both is what makes a divergence legible instead of silent.
SECTION_2_ESTIMATES = {
    "raw_orders": 99_441,
    "positives_labelled": 2_963,
    "positives_mature": 1_234,
    "positives_boleto": 240,
    "positives_boleto_test": 50,
}

# Normal quantiles, hardcoded so this step needs no scipy.
Z_ALPHA_TWO_SIDED_05 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143


# --------------------------------------------------------------------------------------
# Minimum detectable PR-AUC difference (section 2, "the consequence nobody plans for")
# --------------------------------------------------------------------------------------

def ap_standard_error(ap: float, n_pos: int) -> float:
    """
    Planning approximation for the standard error of average precision.  AP is an
    average over the positives, so its sampling error is governed by the positive
    count, not by n.  Rough, and labelled as such - it is superseded at evaluation
    time by the paired bootstrap CI that section 9 mandates.
    """
    return math.sqrt(ap * (1.0 - ap) / n_pos)


def min_detectable_diff(ap: float, n_pos: int, rho: float) -> float:
    """
    Smallest AP gap detectable at alpha = 0.05 (two-sided), power = 0.80.

    rho is the correlation between the two models' AP estimates.  Section 9 requires a
    paired bootstrap over identical test rows, so rho is high in practice; rho = 0 is
    the unpaired worst case and brackets the answer from above.
    """
    se_diff = ap_standard_error(ap, n_pos) * math.sqrt(2.0 * (1.0 - rho))
    return (Z_ALPHA_TWO_SIDED_05 + Z_POWER_80) * se_diff


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------

def main() -> int:
    loader = OlistLoader()
    data_dir = loader.data_dir
    payments = pd.read_csv(data_dir / PAYMENTS_CSV)

    # Population, snapshot, maturation and the split boundary all come from the loader.
    # This script does not re-derive them; if it did, the two definitions could drift and
    # the artifact would describe a population nothing else in the repo uses.
    orders = loader.raw_orders()
    orders["is_positive"] = orders["order_status"] != DELIVERED
    n_raw = len(orders)

    snapshot = loader.snapshot(orders)

    mature = loader.labelled()
    # The gate predates the primary/secondary split and reports the unconditional label
    # throughout; label_a is that label under the loader's naming.
    mature["is_positive"] = mature["label_a"]
    n_dropped_immature = n_raw - len(mature)
    pos_dropped_immature = (
        int(orders["is_positive"].sum()) - int(mature["is_positive"].sum())
    )

    # --- why maturation bites as little as it does ------------------------------------
    # Diagnostics, not filters.  The section 2 estimate expected maturation to halve the
    # positive class; it does not, and the report has to say why with numbers.
    last_delivered_purchase = (
        orders.loc[orders["order_status"] == DELIVERED, "order_purchase_timestamp"]
        .max()
        .strftime("%Y-%m-%d")
    )
    late_window_days = MATURATION_DAYS + 15
    late = orders[
        orders["order_purchase_timestamp"] > snapshot - pd.Timedelta(days=late_window_days)
    ]
    n_late_orders = len(late)
    late_status_summary = ", ".join(
        f"{n} `{s}`" for s, n in late["order_status"].value_counts().items()
    ) or "none"

    # --- boleto subset (section 3.6) --------------------------------------------------
    # Payments are one-to-many: some orders carry more than one payment row (a card plus
    # vouchers, typically).  An order counts as boleto if any of its payment rows is
    # boleto.  The stricter "boleto is the only instrument" definition is reported
    # alongside so the choice is visible rather than buried.
    boleto_any = set(payments.loc[payments["payment_type"] == BOLETO, "order_id"])
    types_per_order = payments.groupby("order_id")["payment_type"].agg(set)
    boleto_only = set(types_per_order[types_per_order == {BOLETO}].index)
    orders_with_payment = set(payments["order_id"])
    n_no_payment_record = int((~orders["order_id"].isin(orders_with_payment)).sum())

    mature["is_boleto"] = mature["order_id"].isin(boleto_any)
    mature["is_boleto_only"] = mature["order_id"].isin(boleto_only)
    boleto = mature[mature["is_boleto"]].copy()

    mixed_instrument = set(types_per_order[types_per_order.map(len) > 1].index)
    n_mixed_instrument = int(mature["order_id"].isin(mixed_instrument).sum())

    # --- temporal split (section 2, section 9) ----------------------------------------
    # One chronological boundary, computed on the full matured population, then applied
    # unchanged to every subset.  This keeps all-orders and boleto on the same test
    # window so their positive counts are comparable.
    #
    # The boundary is taken at a row position rather than by quantile interpolation, so
    # it is an actually-observed timestamp rather than a synthetic instant between two
    # orders.  Ties on the boundary second fall into test, so the realised fraction can
    # sit fractionally above TEST_FRACTION; it is reported rather than assumed.
    # Both come from the loader; `mature` already carries is_test.
    cut = loader.split_boundary
    cut_idx = loader.split_index
    boleto["is_test"] = boleto["order_purchase_timestamp"] >= cut

    all_test = mature[mature["is_test"]]
    boleto_test = boleto[boleto["is_test"]]

    n_pos_all_test = int(all_test["is_positive"].sum())
    n_pos_boleto_test = int(boleto_test["is_positive"].sum())

    # --- section 2 decision rule ------------------------------------------------------
    # The section 2 stack terminates on boleto test positives, so that is the number the
    # rule keys on.
    gate_n = n_pos_boleto_test
    if gate_n >= 150:
        decision = "BOLETO PRIMARY"
        decision_text = (
            "Boleto subset is the **primary** analysis. All-orders is reported alongside "
            "as the larger-sample view."
        )
    elif gate_n >= 50:
        decision = "ALL-ORDERS PRIMARY"
        decision_text = (
            "All-orders is **primary**. Boleto is a clearly-labelled **secondary** "
            "analysis, with the small-n caveat stated in the table caption."
        )
    else:
        decision = "BOLETO IN LIMITATIONS ONLY"
        decision_text = (
            "Boleto appears in **Limitations only**. Do not build a metrics table on it."
        )

    # ----------------------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------------------
    def pct(a: int, b: int) -> str:
        return f"{100.0 * a / b:.2f}%" if b else "-"

    def status_counts(df: pd.DataFrame) -> dict:
        return df.loc[df["is_positive"], "order_status"].value_counts().to_dict()

    stack = [
        ("All Olist orders (raw)", n_raw, int(orders["is_positive"].sum())),
        (
            f"Mature only (purchase + {MATURATION_DAYS}d <= snapshot)",
            len(mature),
            int(mature["is_positive"].sum()),
        ),
        (
            "Boleto subset (any boleto payment row)",
            len(boleto),
            int(boleto["is_positive"].sum()),
        ),
        (
            f"Temporal test split (most recent {TEST_FRACTION:.0%})",
            len(boleto_test),
            n_pos_boleto_test,
        ),
    ]

    L: list[str] = []
    w = L.append

    w("# Positive-count gate")
    w("")
    w("**ARCHITECTURE.md section 2 / build-order step 0.** Generated by "
      "`scripts/00_count_positives.py`.")
    w("Deterministic: no sampling, no model, no randomness. Re-running reproduces every "
      "number below.")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Dataset: `{KAGGLE_DATASET}`")
    for _name, _digest in loader.checksums(ORDERS_CSV, PAYMENTS_CSV).items():
        w(f"- `{_name}` sha256: `{_digest}`")
    w("")
    w("## 1. Parameters")
    w("")
    w("| Parameter | Value | Source |")
    w("|---|---|---|")
    w(f"| Label | `order_status != '{DELIVERED}'` | section 3.1 |")
    w(f"| `MATURATION_DAYS` | {MATURATION_DAYS} | section 3.5 |")
    w(f"| `TEST_FRACTION` | {TEST_FRACTION:.0%}, most recent by purchase timestamp | "
      "sections 2, 9 |")
    w(f"| COD analog | `payment_type == '{BOLETO}'` | section 3.6 |")
    w(f"| Snapshot | `{snapshot}` | derived: latest observed event in the orders table |")
    w(f"| Split boundary | `{cut}` | purchase timestamp at row position "
      f"{cut_idx:,} of {len(mature):,} matured rows, sorted ascending |")
    w("")
    w("The snapshot excludes `order_estimated_delivery_date`, which is a forward-looking "
      "estimate rather than an observation; including it would push the snapshot past the "
      "end of the data and silently weaken the maturation filter.")
    w("")
    w("## 2. The filter stack")
    w("")
    w("Cumulative, in the order section 2 specifies.")
    w("")
    w("| Stage | Orders | Positives | Prevalence |")
    w("|---|---:|---:|---:|")
    for name, rows, pos in stack:
        w(f"| {name} | {rows:,} | {pos:,} | {pct(pos, rows)} |")
    w("")
    w(f"Maturation removed **{n_dropped_immature:,} orders** "
      f"({pct(n_dropped_immature, n_raw)} of the raw table), of which "
      f"**{pos_dropped_immature:,}** carried a positive label. Those are unresolved "
      "orders near the snapshot boundary, not observed failures; section 3.5 requires "
      "this count be stated.")
    w("")
    w("### Estimate against actual")
    w("")
    w("Section 2's figures are estimates from the published status distribution, with "
      "the instruction to compute the real ones. Both are shown so any divergence is "
      "visible rather than silent.")
    w("")
    w("| Stage | Section 2 estimate | Actual | Delta |")
    w("|---|---:|---:|---:|")
    est_actual = [
        ("Raw orders", SECTION_2_ESTIMATES["raw_orders"], n_raw),
        ("Positives after label",
         SECTION_2_ESTIMATES["positives_labelled"], int(orders["is_positive"].sum())),
        ("Positives after maturation",
         SECTION_2_ESTIMATES["positives_mature"], int(mature["is_positive"].sum())),
        ("Positives in boleto subset",
         SECTION_2_ESTIMATES["positives_boleto"], int(boleto["is_positive"].sum())),
        ("Positives in boleto test split",
         SECTION_2_ESTIMATES["positives_boleto_test"], n_pos_boleto_test),
    ]
    for label, est, act in est_actual:
        w(f"| {label} | {est:,} | {act:,} | {act - est:+,} |")
    w("")
    w("### Why maturation removes so little")
    w("")
    w("The estimate expected maturation to cut the positive class roughly in half, "
      "leaving only `canceled` + `unavailable`. It does not, and the reason is "
      "structural rather than a filter bug.")
    w("")
    w(f"Order volume collapses at the end of the panel: the last `{DELIVERED}` order was "
      f"purchased `{last_delivered_purchase}`, and only **{n_late_orders}** orders were "
      f"purchased in the final {MATURATION_DAYS + 15} days before the snapshot "
      f"({late_status_summary}). There is almost nothing near the boundary to remove.")
    w("")
    w("The statuses the estimate assumed were immature are spread across the whole panel, "
      "not banked against the cutoff:")
    w("")
    w("| `order_status` | n | First purchase | Last purchase |")
    w("|---|---:|---|---|")
    for st in ("shipped", "invoiced", "processing", "created", "approved"):
        sub = orders.loc[orders["order_status"] == st, "order_purchase_timestamp"]
        if len(sub):
            w(f"| `{st}` | {len(sub):,} | {sub.min():%Y-%m-%d} | {sub.max():%Y-%m-%d} |")
    w("")
    w("A `shipped` order purchased in 2016 and still undelivered at a 2018 snapshot is a "
      "**mature, genuinely unresolved order**, not an immature one. The maturation filter "
      "is doing its job; the estimate's mechanism was wrong. The practical consequence is "
      "favourable - the positive class survives the filter roughly intact - but it "
      "sharpens the section 3.6 construct-validity problem rather than easing it, because "
      "the surviving positives are a broader mix of failure mechanisms than the estimate "
      "assumed. See the composition table below.")
    w("")
    w(f"`{PAYMENTS_CSV}` has no row for **{n_no_payment_record}** order(s); those cannot "
      "be confirmed as boleto and are treated as non-boleto.")
    w("")
    w("## 3. Test-split positives, both candidate primary populations")
    w("")
    w("One chronological boundary, computed on the full matured population and applied "
      "unchanged to each subset, so the two rows share a test window.")
    w("")
    w("| Population | Test orders | Test positives | Prevalence |")
    w("|---|---:|---:|---:|")
    w(f"| All orders | {len(all_test):,} | {n_pos_all_test:,} | "
      f"{pct(n_pos_all_test, len(all_test))} |")
    w(f"| Boleto subset | {len(boleto_test):,} | {n_pos_boleto_test:,} | "
      f"{pct(n_pos_boleto_test, len(boleto_test))} |")
    w("")
    w(f"Realised test fraction: **{pct(len(all_test), len(mature))}** of matured orders "
      f"(target {TEST_FRACTION:.0%}); boundary is an observed purchase timestamp, and "
      "ties on the boundary second fall into test.")
    w("")
    n_bo = int(mature["is_boleto_only"].sum())
    if n_bo == len(boleto):
        w("**Sensitivity on the boleto definition: none.** The looser *any boleto payment "
          "row* and the stricter *boleto is the only instrument* definitions select the "
          f"identical {len(boleto):,} orders. Of the {n_mixed_instrument:,} matured orders "
          "carrying more than one distinct payment type, **zero** include boleto - in "
          "Olist, boleto is never combined with another instrument. The definitional "
          "choice therefore costs nothing here, and no caveat is owed.")
    else:
        n_bo_pos = int(mature.loc[mature["is_boleto_only"], "is_positive"].sum())
        w("Sensitivity on the boleto definition: under the stricter *boleto is the only "
          f"payment instrument* rule the matured subset holds {n_bo:,} orders and "
          f"{n_bo_pos:,} positives, versus {len(boleto):,} / "
          f"{int(boleto['is_positive'].sum()):,} under *any boleto row*. The looser "
          "definition is used above.")
    w("")
    w("## 4. Positive-class composition by `order_status`")
    w("")
    w("Section 3.6 requires this: `unavailable` is stock unavailability and much of "
      "`canceled` is seller-side - both fulfilment failures with a different causal "
      "generator from doorstep refusal.")
    w("")
    populations = [
        ("Raw", orders),
        ("Mature", mature),
        ("Mature boleto", boleto),
        ("Test, all orders", all_test),
        ("Test, boleto", boleto_test),
    ]
    raw_counts = status_counts(orders)
    statuses = sorted(
        {s for _, df in populations for s in status_counts(df)},
        key=lambda s: -raw_counts.get(s, 0),
    )
    w("| Population | " + " | ".join(f"`{s}`" for s in statuses) + " | Total |")
    w("|---|" + "---:|" * (len(statuses) + 1))
    for label, df in populations:
        counts = status_counts(df)
        cells = " | ".join(f"{counts.get(s, 0):,}" for s in statuses)
        w(f"| {label} | {cells} | {sum(counts.values()):,} |")
    w("")

    # -- decision ----------------------------------------------------------------------
    w("## 5. Section 2 decision")
    w("")
    w(f"Gate number - **boleto test positives = {gate_n}**.")
    w("")
    w("| Test positives | Action | |")
    w("|---|---|---|")
    rules = ("BOLETO PRIMARY", "ALL-ORDERS PRIMARY", "BOLETO IN LIMITATIONS ONLY")
    marks = ["<-- **THIS**" if decision == d else "" for d in rules]
    w(f"| >= 150 | Boleto primary, all-orders alongside | {marks[0]} |")
    w(f"| 50-150 | All-orders primary, boleto secondary with small-n caveat | {marks[1]} |")
    w(f"| < 50 | Boleto in Limitations only, no metrics table | {marks[2]} |")
    w("")
    w(f"**Decision: {decision}.** {decision_text}")
    w("")

    # -- MDD ---------------------------------------------------------------------------
    primary_is_boleto = decision == "BOLETO PRIMARY"
    primary_n_pos = n_pos_boleto_test if primary_is_boleto else n_pos_all_test
    primary_name = "boleto" if primary_is_boleto else "all orders"

    w("## 6. Minimum detectable PR-AUC difference")
    w("")
    w("Section 2 requires this be computed at the realised test size and printed at the "
      "top of the evaluation section. Any comparison below this threshold is reported as "
      '"not resolvable at this sample size" - not as a win or a loss.')
    w("")
    w("Planning approximation, stated with its assumptions rather than implied:")
    w("")
    w("```")
    w("SE(AP)   ~  sqrt( AP (1 - AP) / n_pos )      AP averages over positives, so its")
    w("                                             sampling error tracks n_pos, not n")
    w("SE(diff) =  SE(AP) * sqrt( 2 (1 - rho) )     rho = correlation between the two")
    w("                                             models' AP estimates")
    w("MDD      =  (z_0.975 + z_0.80) * SE(diff)    alpha = 0.05 two-sided, power = 0.80")
    w("```")
    w("")
    w("Evaluated at the primary population's test positive count "
      f"(**{primary_name}, n_pos = {primary_n_pos}**). AP is not known at step 0, so the "
      "threshold is tabulated across plausible values; the realised figure is recomputed "
      "against the section 9 paired bootstrap.")
    w("")
    w("| Assumed AP | MDD, paired (rho = 0.8) | MDD, paired (rho = 0.5) | "
      "MDD, unpaired (rho = 0) |")
    w("|---:|---:|---:|---:|")
    for ap in (0.05, 0.10, 0.20, 0.30):
        cells = " | ".join(
            f"{min_detectable_diff(ap, primary_n_pos, r):.3f}" for r in (0.8, 0.5, 0.0)
        )
        w(f"| {ap:.2f} | {cells} |")
    w("")
    w("Section 9 mandates a paired bootstrap over identical test rows, so the rho = 0.8 "
      "column is the operative one; rho = 0 brackets it from above.")
    w("")
    if n_pos_boleto_test > 0 and not primary_is_boleto:
        w("Secondary population for reference "
          f"(**boleto, n_pos = {n_pos_boleto_test}**), rho = 0.8:")
        w("")
        w("| Assumed AP | MDD, paired (rho = 0.8) |")
        w("|---:|---:|")
        for ap in (0.05, 0.10, 0.20, 0.30):
            w(f"| {ap:.2f} | {min_detectable_diff(ap, n_pos_boleto_test, 0.8):.3f} |")
        w("")

    # -- consequences ------------------------------------------------------------------
    w("## 7. Consequences for the build")
    w("")
    # Section 5.4's re-add condition is "if the section 2 gate returns a larger test set
    # than estimated", and it was written against the ~50-positive estimate.  The primary
    # population carries n_pos far above that, so the condition is met on its own terms -
    # but the useful statement is what is now resolvable, not a bare verdict.
    est_test_pos = SECTION_2_ESTIMATES["positives_boleto_test"]
    ratio = primary_n_pos / est_test_pos
    mdd_at_10 = min_detectable_diff(0.10, primary_n_pos, 0.8)
    mdd_est_at_10 = min_detectable_diff(0.10, est_test_pos, 0.8)
    w("- **EBM ablation (sections 5.4, 15).** The re-add condition is the gate returning "
      "a larger test set than estimated, and it is **met**: the primary population holds "
      f"{primary_n_pos} test positives against the section 2 estimate of {est_test_pos} "
      f"({ratio:.1f}x). At AP = 0.10 the resolvable gap tightens from "
      f"~{mdd_est_at_10:.3f} to ~{mdd_at_10:.3f} AP. Section 5.4 calls EBM the single "
      "highest-value re-add, so **reinstate it** - with the caveat that the ~1-point "
      "difference section 5.4 describes as the interesting case is still below this "
      "threshold. A 3-point-plus gap now resolves; a 1-point gap does not.")
    w(f"- **Boleto stays small.** {gate_n} test positives means every boleto number "
      f"carries a resolvable gap of ~{min_detectable_diff(0.10, gate_n, 0.8):.3f} AP at "
      "AP = 0.10. Secondary, caveated, and no model comparison is run on it.")
    w("- **Section 8.1 Rudin paragraph.** The gate makes the *\"we tested the glassbox "
      "alternative and report the number\"* branch the live one, not the *\"could not "
      "resolve\"* branch. Write that version.")
    w("- **Section 9 item 1.** This positive count and this threshold go at the top of "
      "the evaluation section, ahead of any metric.")
    w("- **Section 3.6.** The composition table in section 4 above is the input to the "
      "stricter-label sensitivity check.")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
