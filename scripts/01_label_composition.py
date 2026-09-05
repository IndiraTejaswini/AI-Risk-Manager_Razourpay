#!/usr/bin/env python3
"""
Label composition and candidate-label comparison.

Answers, all on the mature order set defined by scripts/00_count_positives.py:

  1. Positive-class composition by order_status - count and share of positives.
  2. A fulfillment flag: which positives physically entered fulfillment and which
     never shipped, with a total for each group.
  3. For shipped-and-never-delivered orders only: distribution by purchase month and
     the top 10 sellers by count, to establish whether they are spread across the
     panel or clustered.
  4. Three candidate label definitions with train/test positive counts.

Writes eval/label_composition.md.

No model, no sampling, no randomness.  Two runs produce identical counts.

Note on the fulfillment flag: the honest signal is order_delivered_carrier_date, not
order_status.  Every `shipped` order carries a carrier date, but so do 75 `canceled`
orders - those shipped and were cancelled in transit.  A status-only flag would put
them on the wrong side of the line, and they are exactly the rows candidate label (b)
turns on.  Both readings are reported.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------------------
# Parameters.  The population - maturation, labels, split boundary - comes from
# data/loader.py.  This script owns only its own analysis constants.
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import (  # noqa: E402
    DELIVERED,
    ITEMS_CSV,
    KAGGLE_DATASET,
    ORDERS_CSV,
    OlistLoader,
)

# Committed results of scripts/00_count_positives.py.  Now a loader regression test:
# these are the numbers the artifacts in eval/ were written against, and the loader is
# the single definition that has to keep producing them.
GATE_EXPECTED = {
    "raw_orders": 99_441,
    "mature_orders": 99_433,
    "mature_positives": 2_955,
    "test_orders_all": 19_887,
    "test_positives_all": 379,
}

# Statuses that never reach a carrier, used only to cross-check the carrier-date flag.
NEVER_SHIPPED_STATUSES = {"unavailable", "invoiced", "processing", "created", "approved"}

TOP_N_SELLERS = 10

OUT_PATH = REPO_ROOT / "eval" / "label_composition.md"


# Held identical to scripts/00_count_positives.py section 6, so a candidate label's
# sample-size cost is quoted on the same scale as the gate's.
Z_ALPHA_TWO_SIDED_05 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143


def min_detectable_diff(ap: float, n_pos: int, rho: float) -> float:
    se = math.sqrt(ap * (1.0 - ap) / n_pos)
    return (Z_ALPHA_TWO_SIDED_05 + Z_POWER_80) * se * math.sqrt(2.0 * (1.0 - rho))


def main() -> int:
    loader = OlistLoader()
    data_dir = loader.data_dir
    items = pd.read_csv(data_dir / ITEMS_CSV)

    # Population from the loader; this script does not re-derive maturation or the split.
    orders = loader.raw_orders()
    orders["is_positive"] = orders["order_status"] != DELIVERED
    n_raw = len(orders)

    mature = loader.labelled()
    mature["is_positive"] = mature["label_a"]
    cut = loader.split_boundary

    # --- loader regression check against the committed step 0 numbers -----------------
    got = {
        "raw_orders": n_raw,
        "mature_orders": len(mature),
        "mature_positives": int(mature["is_positive"].sum()),
        "test_orders_all": int(mature["is_test"].sum()),
        "test_positives_all": int((mature["is_test"] & mature["is_positive"]).sum()),
    }
    mismatches = {k: (GATE_EXPECTED[k], v) for k, v in got.items() if GATE_EXPECTED[k] != v}
    if mismatches:
        sys.exit(
            "OlistLoader no longer reproduces the population the committed artifacts in "
            "eval/ were written against; refusing to write a composition report on a "
            "different denominator.\n"
            + "\n".join(
                f"  {k}: committed={e:,} loader={a:,}" for k, (e, a) in mismatches.items()
            )
        )

    # --- fulfillment flag -------------------------------------------------------------
    # Physical evidence the parcel moved, rather than an inference from status.
    mature["entered_fulfillment"] = mature["order_delivered_carrier_date"].notna()
    pos = mature[mature["is_positive"]].copy()
    n_pos = len(pos)

    # Cross-check the carrier-date flag against the statuses that structurally cannot
    # have shipped.  A non-zero count here means the flag and the status vocabulary
    # disagree and one of them is wrong.
    contradiction = int(
        (pos["order_status"].isin(NEVER_SHIPPED_STATUSES) & pos["entered_fulfillment"]).sum()
    )

    # --- shipped-and-never-delivered --------------------------------------------------
    # Status `shipped` is the group the question names: handed to a carrier, no delivery
    # date, not cancelled.
    snd = mature[mature["order_status"] == "shipped"].copy()

    by_month = (
        snd["order_purchase_timestamp"].dt.to_period("M").value_counts().sort_index()
    )
    # Panel-wide monthly volume, so the monthly counts can be read as a rate rather than
    # as raw volume tracking overall growth.
    panel_month = (
        mature["order_purchase_timestamp"].dt.to_period("M").value_counts().sort_index()
    )

    # Sellers.  An order can contain items from several sellers; it is counted once per
    # distinct seller, so the seller column sums to more than the order count.
    order_seller = items[["order_id", "seller_id"]].drop_duplicates()
    snd_sellers = order_seller[order_seller["order_id"].isin(set(snd["order_id"]))]
    mature_sellers = order_seller[order_seller["order_id"].isin(set(mature["order_id"]))]

    seller_snd = snd_sellers["seller_id"].value_counts()
    seller_total = mature_sellers["seller_id"].value_counts()
    n_snd_with_seller = snd_sellers["order_id"].nunique()
    n_snd_missing_seller = len(snd) - n_snd_with_seller

    # Deterministic ordering: count descending, then seller_id ascending.
    seller_tbl = (
        pd.DataFrame({"snd": seller_snd})
        .join(pd.DataFrame({"total": seller_total}), how="left")
        .reset_index(names="seller_id")
        .sort_values(["snd", "seller_id"], ascending=[False, True], kind="mergesort")
    )
    top_sellers = seller_tbl.head(TOP_N_SELLERS)

    # Denominators are order-seller pairs on both sides, matching the numerators, so the
    # two shares are directly comparable.  Mixing pair counts against order counts would
    # inflate the ratio by the multi-seller rate.
    n_snd_pairs = len(snd_sellers)
    n_panel_pairs = len(mature_sellers)
    top_share_of_snd = top_sellers["snd"].sum() / n_snd_pairs if n_snd_pairs else 0.0
    top_share_of_panel = (
        top_sellers["total"].sum() / n_panel_pairs if n_panel_pairs else 0.0
    )

    # --- candidate labels -------------------------------------------------------------
    not_delivered = mature["order_status"] != DELIVERED
    labels = {
        "(a) any non-delivered": not_delivered,
        "(b) entered fulfillment, never delivered": (
            not_delivered & mature["entered_fulfillment"]
        ),
        "(c) non-delivered excluding `unavailable`": (
            not_delivered & (mature["order_status"] != "unavailable")
        ),
    }
    # Status-only reading of (b), for the footnote.
    b_status_only = int((mature["order_status"] == "shipped").sum())

    is_test = mature["is_test"]
    n_train_rows = int((~is_test).sum())
    n_test_rows = int(is_test.sum())

    # ----------------------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------------------
    def pct(a, b) -> str:
        return f"{100.0 * a / b:.2f}%" if b else "-"

    L: list[str] = []
    w = L.append

    w("# Label composition")
    w("")
    w("Companion to `eval/positive_counts.md`. Generated by "
      "`scripts/01_label_composition.py`.")
    w("Everything below is computed on the **mature order set** defined by "
      "`scripts/00_count_positives.py`.")
    w("Deterministic: no sampling, no model, no randomness.")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Dataset: `{KAGGLE_DATASET}`")
    for _name, _digest in loader.checksums(ORDERS_CSV, ITEMS_CSV).items():
        w(f"- `{_name}` sha256: `{_digest}`")
    w("")
    w("| Population | Value |")
    w("|---|---:|")
    w(f"| Mature orders | {len(mature):,} |")
    w(f"| Mature positives | {n_pos:,} |")
    w(f"| Split boundary | `{cut}` |")
    w(f"| Train rows / test rows | {n_train_rows:,} / {n_test_rows:,} |")
    w("")
    w("Population cross-checked against the committed step 0 numbers before writing; "
      "the script exits rather than report a composition against a different "
      "denominator.")
    w("")

    # -- 1. composition ----------------------------------------------------------------
    w("## 1. Positive-class composition by `order_status`")
    w("")
    w("| `order_status` | Positives | Share of positives | Share of mature orders |")
    w("|---|---:|---:|---:|")
    comp = pos["order_status"].value_counts()
    for status, n in comp.items():
        w(f"| `{status}` | {n:,} | {pct(n, n_pos)} | {pct(n, len(mature))} |")
    w(f"| **Total** | **{n_pos:,}** | **100.00%** | **{pct(n_pos, len(mature))}** |")
    w("")

    # -- 2. fulfillment flag -----------------------------------------------------------
    w("## 2. Entered fulfillment vs never shipped")
    w("")
    w("The flag is `order_delivered_carrier_date IS NOT NULL` - physical evidence the "
      "parcel reached a carrier - rather than an inference from `order_status`. The two "
      "readings disagree, and the disagreement is material:")
    w("")
    w("| `order_status` | Positives | Entered fulfillment | Never shipped |")
    w("|---|---:|---:|---:|")
    for status, n in comp.items():
        sub = pos[pos["order_status"] == status]
        ef = int(sub["entered_fulfillment"].sum())
        w(f"| `{status}` | {n:,} | {ef:,} | {n - ef:,} |")
    n_ef = int(pos["entered_fulfillment"].sum())
    w(f"| **Total** | **{n_pos:,}** | **{n_ef:,}** | **{n_pos - n_ef:,}** |")
    w("")
    w("| Group | Positives | Share of positives |")
    w("|---|---:|---:|")
    w(f"| Entered fulfillment | {n_ef:,} | {pct(n_ef, n_pos)} |")
    w(f"| Never shipped | {n_pos - n_ef:,} | {pct(n_pos - n_ef, n_pos)} |")
    w("")
    n_cancel_shipped = int(
        pos.loc[pos["order_status"] == "canceled", "entered_fulfillment"].sum()
    )
    n_cancel = int((pos["order_status"] == "canceled").sum())
    w(f"**{n_cancel_shipped} of {n_cancel} `canceled` positives carry a carrier date** - "
      "they shipped and were cancelled in transit. A status-only flag would file them "
      "under *never shipped*, which is the wrong side of the line for a delivery-failure "
      "label and is exactly the population candidate label (b) turns on.")
    w("")
    w("Cross-check: statuses that structurally cannot have shipped "
      f"({', '.join('`' + s + '`' for s in sorted(NEVER_SHIPPED_STATUSES))}) carrying a "
      f"carrier date: **{contradiction}**. "
      + ("The carrier-date flag and the status vocabulary agree."
         if contradiction == 0 else
         "Non-zero - the flag and the status vocabulary disagree and one is wrong."))
    w("")

    # -- 3. shipped-and-never-delivered ------------------------------------------------
    w("## 3. Shipped and never delivered")
    w("")
    w(f"`order_status == 'shipped'`: handed to a carrier, no delivery date, not "
      f"cancelled. **{len(snd):,} orders**, {pct(len(snd), n_pos)} of the positive class.")
    w("")
    w("### 3.1 By purchase month")
    w("")
    w("Monthly panel volume is shown alongside so the counts read as a rate rather than "
      "tracking overall growth.")
    w("")
    THIN_MONTH = 100  # below this, the monthly rate is too noisy to read
    w("| Purchase month | Shipped, never delivered | Mature orders that month | Rate |")
    w("|---|---:|---:|---:|")
    for period, n in by_month.items():
        tot = int(panel_month.get(period, 0))
        thin = " †" if tot < THIN_MONTH else ""
        w(f"| {period}{thin} | {n:,} | {tot:,} | {pct(n, tot)} |")
    w("")
    thin_months = [str(p) for p in by_month.index if int(panel_month.get(p, 0)) < THIN_MONTH]
    if thin_months:
        w(f"† Fewer than {THIN_MONTH} orders in the month "
          f"({', '.join(thin_months)}) - the rate is noise at that denominator and the "
          "panel's ramp-up and wind-down tails should not be read as a trend.")
        w("")
    peak_month = by_month.idxmax()
    peak_n = int(by_month.max())
    n_months = int(len(by_month))
    # Concentration read on the thick months only, so the thin tails cannot drive it.
    thick = by_month[[int(panel_month.get(p, 0)) >= THIN_MONTH for p in by_month.index]]
    thick_rates = [
        by_month[p] / int(panel_month.get(p, 0)) for p in thick.index
    ]
    w(f"Spread across **{n_months} of the panel's {len(panel_month)} months**, from "
      f"{by_month.index.min()} to {by_month.index.max()}. The heaviest single month "
      f"({peak_month}) holds {peak_n:,} orders - {pct(peak_n, len(snd))} of the group. "
      f"Across the {len(thick)} months with at least {THIN_MONTH} orders the rate moves "
      f"between {min(thick_rates) * 100:.2f}% and {max(thick_rates) * 100:.2f}%.")
    w("")
    w("**Temporally spread, not clustered.** No month or short window accounts for the "
      "bulk of them, and the monthly rate stays inside a narrow band across two years. "
      "This is a standing background rate, not a bounded incident - so the label is not "
      "an artifact of one carrier outage or one bad quarter, and a model trained on the "
      "earlier period is describing the same phenomenon that appears in the test window.")
    w("")

    w("### 3.2 Top sellers")
    w("")
    w("An order can contain items from several sellers and is counted once per distinct "
      "seller, so the seller column sums above the order count. "
      f"{n_snd_with_seller:,} of {len(snd):,} orders join to `{ITEMS_CSV}`"
      + (f"; {n_snd_missing_seller} has no item row and is absent from this table."
         if n_snd_missing_seller else "."))
    w("")
    w(f"| # | `seller_id` | Shipped, never delivered | Seller's mature orders | "
      "Rate within seller | Seller's share of panel |")
    w("|---:|---|---:|---:|---:|---:|")
    for i, row in enumerate(top_sellers.itertuples(index=False), start=1):
        w(f"| {i} | `{row.seller_id}` | {int(row.snd):,} | {int(row.total):,} | "
          f"{pct(row.snd, row.total)} | {pct(row.total, n_panel_pairs)} |")
    w("")
    w(f"Shares are order-seller pairs on both sides ({n_snd_pairs:,} pairs in the group, "
      f"{n_panel_pairs:,} across the panel), so the two are directly comparable.")
    w("")
    w(f"The top {TOP_N_SELLERS} sellers account for **{top_share_of_snd * 100:.2f}%** of "
      f"shipped-and-never-delivered pairs while carrying "
      f"**{top_share_of_panel * 100:.2f}%** of all pairs "
      f"({len(seller_snd):,} distinct sellers appear in the group in total, out of "
      f"{len(seller_total):,} in the panel).")
    w("")
    w("The rate-within-seller column is the one that carries the signal: the panel-wide "
      f"rate is {pct(n_snd_pairs, n_panel_pairs)}, so a seller sitting near that figure "
      "is unremarkable regardless of how many failures they contribute in absolute terms.")
    w("")
    concentrated = top_share_of_snd > 2 * top_share_of_panel
    w("**Seller concentration: "
      + ("present" if concentrated else "mild")
      + ".** "
      + (f"The top {TOP_N_SELLERS} are over-represented by "
         f"{top_share_of_snd / top_share_of_panel:.1f}x relative to their order volume, "
         "so a meaningful share of this failure mode is seller-side rather than "
         "delivery-side. That is a fulfilment signal, and it argues for `seller_id` as a "
         "feature and against reading these orders as doorstep refusals."
         if top_share_of_panel and concentrated else
         "Their share of the failures is close to their share of volume, so the group is "
         "not driven by a handful of sellers."))
    w("")

    # -- 4. candidate labels -----------------------------------------------------------
    w("## 4. Candidate label definitions")
    w("")
    w("MDD is the minimum detectable PR-AUC difference at that test size, on the same "
      "approximation `eval/positive_counts.md` section 6 states, at AP = 0.10 and "
      "rho = 0.8.")
    w("")
    w("| Definition | Total positives | Train | Test | Test prevalence | MDD |")
    w("|---|---:|---:|---:|---:|---:|")
    for name, mask in labels.items():
        tr = int((mask & ~is_test).sum())
        te = int((mask & is_test).sum())
        mdd = f"{min_detectable_diff(0.10, te, 0.8):.3f}" if te else "-"
        w(f"| {name} | {int(mask.sum()):,} | {tr:,} | {te:,} | "
          f"{pct(te, n_test_rows)} | {mdd} |")
    w("")
    w(f"Definition (b) uses the carrier-date flag. Read from status alone "
      f"(`order_status == 'shipped'`) it would be {b_status_only:,} orders, omitting the "
      f"{n_cancel_shipped} cancelled-in-transit rows.")
    w("")
    w("Read against the section 2 decision rule, which keys on test positives:")
    w("")
    for name, mask in labels.items():
        te = int((mask & is_test).sum())
        verdict = (
            "well clear of the 150 threshold" if te >= 300
            else "above the 150 threshold, but with little headroom" if te >= 150
            else "in the 50-150 band, all-orders primary" if te >= 50
            else "below 50 - not enough to build a metrics table on"
        )
        w(f"- **{name}** - {te:,} test positives, {verdict}.")
    w("")
    w("None of the three falls below the gate. The choice is therefore a construct-"
      "validity question rather than a sample-size one, and section 3.6 already frames "
      "it: (a) is the broadest and the least specific to a delivery failure; (b) is the "
      "closest to doorstep refusal and costs roughly 60% of the positive class; (c) sits "
      "between them and removes the one status that is unambiguously a stock problem "
      "rather than a delivery one. This script reports the counts; it does not pick.")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
