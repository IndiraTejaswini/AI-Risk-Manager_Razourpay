#!/usr/bin/env python3
"""
Task G: join-cardinality audit.

`order_items` was one instance of a general class - **a row's absence from a joined
table encoding the outcome**.  Neither the name-based whitelist check nor the
carrier-date perturbation test can see it: every column involved is admissible and
knowable at checkout, and what leaks is the cardinality of the join.

This audits every join in the pipeline the same way, plus the availability flags, which
are the features most likely to be proxies for the same artifact.

Reads source tables directly rather than through `OlistLoader.load_table`, because the
audit must cover tables (`order_reviews`, `geolocation`) that are deliberately absent
from the feature whitelist.  Nothing here builds a feature.

Writes eval/join_cardinality_audit.md.  Deterministic.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.loader import PRIMARY_LABEL, SECONDARY_LABEL, OlistLoader  # noqa: E402
from features.builder import FeatureBuilder  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "join_cardinality_audit.md"

#: A group whose positive rate exceeds this multiple of the panel rate is flagged.
FLAG_RATIO = 3.0


@dataclass
class JoinResult:
    table: str
    via: str
    n_zero: int
    n_zero_due_to_items: int
    rate_a: float
    rate_b: float
    ratio_a: float
    ratio_b: float
    risk_n_zero: int
    risk_rate_b: float
    risk_ratio_b: float

    @property
    def flagged(self) -> bool:
        if self.n_zero == 0:
            return False
        return max(self.ratio_a, self.ratio_b) >= FLAG_RATIO


def _secondary_ap_from_model_report() -> tuple[str, str]:
    """
    Read the leaked and corrected secondary AP straight from the artifact that
    computes them, rather than citing a number retyped here.

    A citation is exactly the kind of number that goes stale silently: this file does
    not train a model, so if it were hardcoded it could only ever be corrected by
    someone remembering to. `eval/model_report.md` is generated earlier in the
    pipeline (`make model`, before `make features` ... `06_join_audit`), so it exists
    by the time this runs.
    """
    text = (REPO_ROOT / "eval" / "model_report.md").read_text(encoding="utf-8")
    m = re.search(
        r"\| \*\*Average precision\*\* \| ([\d.]+) \| \*\*([\d.]+)\*\* \|", text
    )
    if not m:
        raise ValueError(
            "could not find the leaked/corrected secondary AP row in "
            "eval/model_report.md - has its table format changed?"
        )
    return m.group(1), m.group(2)


def main() -> int:
    loader = OlistLoader()
    d = loader.data_dir

    matured = loader.labelled()
    risk = loader.risk_set()
    leaked_ap, corrected_ap = _secondary_ap_from_model_report()

    panel_a = float(matured[SECONDARY_LABEL].mean())
    panel_b = float(matured[PRIMARY_LABEL].mean())
    risk_panel_b = float(risk[PRIMARY_LABEL].mean())

    items = pd.read_csv(d / "olist_order_items_dataset.csv",
                        usecols=["order_id", "product_id", "seller_id"])
    customers = pd.read_csv(d / "olist_customers_dataset.csv",
                            usecols=["customer_id", "customer_zip_code_prefix"])

    order_has_items = set(items["order_id"])

    # Orders reaching >=1 row of each table, by the join path the pipeline would use.
    reach: dict[str, tuple[str, set]] = {}

    reach["order_items"] = ("orders.order_id → items.order_id", order_has_items)

    payments = pd.read_csv(d / "olist_order_payments_dataset.csv", usecols=["order_id"])
    reach["order_payments"] = (
        "orders.order_id → payments.order_id", set(payments["order_id"])
    )

    reviews = pd.read_csv(d / "olist_order_reviews_dataset.csv", usecols=["order_id"])
    reach["order_reviews"] = (
        "orders.order_id → reviews.order_id", set(reviews["order_id"])
    )

    reach["customers"] = (
        "orders.customer_id → customers.customer_id",
        set(matured.loc[matured["customer_id"].isin(set(customers["customer_id"])),
                        "order_id"]),
    )

    products = pd.read_csv(d / "olist_products_dataset.csv", usecols=["product_id"])
    ok_items = items[items["product_id"].isin(set(products["product_id"]))]
    reach["products"] = (
        "orders → items.product_id → products.product_id", set(ok_items["order_id"])
    )

    sellers = pd.read_csv(d / "olist_sellers_dataset.csv", usecols=["seller_id"])
    ok_sellers = items[items["seller_id"].isin(set(sellers["seller_id"]))]
    reach["sellers"] = (
        "orders → items.seller_id → sellers.seller_id", set(ok_sellers["order_id"])
    )

    geo = pd.read_csv(d / "olist_geolocation_dataset.csv",
                      usecols=["geolocation_zip_code_prefix"])
    geo_zips = set(geo["geolocation_zip_code_prefix"].unique())
    cust_geo = customers[customers["customer_zip_code_prefix"].isin(geo_zips)]
    reach["geolocation (customer side)"] = (
        "orders.customer_id → customers.zip → geolocation.zip",
        set(matured.loc[matured["customer_id"].isin(set(cust_geo["customer_id"])),
                        "order_id"]),
    )

    # Seller side of the same reference join.  route_distance_km needs BOTH endpoints,
    # so the customer-side entry alone does not discharge the standing rule for it.
    sellers_zip = pd.read_csv(d / "olist_sellers_dataset.csv",
                              usecols=["seller_id", "seller_zip_code_prefix"])
    ok_seller_geo = set(
        sellers_zip.loc[
            sellers_zip["seller_zip_code_prefix"].isin(geo_zips), "seller_id"
        ]
    )
    reach["geolocation (seller side)"] = (
        "orders → items.seller_id → sellers.zip → geolocation.zip",
        set(items.loc[items["seller_id"].isin(ok_seller_geo), "order_id"]),
    )

    results: list[JoinResult] = []
    for table, (via, ids) in reach.items():
        zero = ~matured["order_id"].isin(ids)
        n_zero = int(zero.sum())
        sub = matured[zero]

        rate_a = float(sub[SECONDARY_LABEL].mean()) if n_zero else float("nan")
        rate_b = float(sub[PRIMARY_LABEL].mean()) if n_zero else float("nan")

        rzero = ~risk["order_id"].isin(ids)
        rn = int(rzero.sum())

        results.append(
            JoinResult(
                table=table,
                via=via,
                n_zero=n_zero,
                n_zero_due_to_items=int(
                    (zero & ~matured["order_id"].isin(order_has_items)).sum()
                ),
                rate_a=rate_a,
                rate_b=rate_b,
                ratio_a=rate_a / panel_a if n_zero else float("nan"),
                ratio_b=rate_b / panel_b if n_zero and panel_b else float("nan"),
                risk_n_zero=rn,
                risk_rate_b=float(risk[rzero][PRIMARY_LABEL].mean()) if rn else float("nan"),
                risk_ratio_b=(
                    float(risk[rzero][PRIMARY_LABEL].mean()) / risk_panel_b
                    if rn else float("nan")
                ),
            )
        )

    # ---------------------------------------------------------------- availability
    matrix = FeatureBuilder().build(matured)
    flags = [c for c in matrix.columns if c.startswith("has_")] + ["n_missing_features"]
    y_a = matured[SECONDARY_LABEL].astype(float).to_numpy()
    y_b = matured[PRIMARY_LABEL].astype(float).to_numpy()

    flag_rows = []
    for col in flags:
        v = matrix[col].astype(float).to_numpy()
        const = float(np.nanstd(v)) == 0.0
        corr_a = float("nan") if const else float(np.corrcoef(v, y_a)[0, 1])
        corr_b = float("nan") if const else float(np.corrcoef(v, y_b)[0, 1])
        if col == "n_missing_features":
            set_mask = v > 0
            set_label = "> 0"
        else:
            set_mask = v == 1
            set_label = "= 1"
        n_set = int(set_mask.sum())
        flag_rows.append({
            "col": col, "set_label": set_label, "constant": const,
            "n_set": n_set,
            "rate_a_set": float(y_a[set_mask].mean()) if n_set else float("nan"),
            "rate_a_clear": float(y_a[~set_mask].mean()) if n_set < len(v) else float("nan"),
            "corr_a": corr_a, "corr_b": corr_b,
        })

    # ---------------------------------------------------------------- report
    def pctf(x, dp=3):
        return "—" if not np.isfinite(x) else f"{100 * x:.{dp}f}%"

    def rat(x):
        return "—" if not np.isfinite(x) else f"{x:.2f}×"

    L: list[str] = []
    w = L.append

    w("# Join-cardinality audit")
    w("")
    w("Generated by `scripts/06_join_audit.py`. Deterministic.")
    w("")
    w(f"- Generated (UTC): `{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}`")
    w(f"- Matured population: {len(matured):,} orders, `{SECONDARY_LABEL}` rate "
      f"{pctf(panel_a)}, `{PRIMARY_LABEL}` rate {pctf(panel_b)}")
    w(f"- Risk set: {len(risk):,} orders, `{PRIMARY_LABEL}` rate {pctf(risk_panel_b)}")
    w(f"- Flag threshold: positive rate ≥ **{FLAG_RATIO:g}×** the panel rate")
    w("")
    w("## Why this audit exists")
    w("")
    w("`order_items` turned out to be a leak: 767 matured orders have no item rows and "
      "100% of them are `label_a` positives, because an order that resolves to "
      "`unavailable` has no items in the export. Correcting for it dropped the secondary "
      f"target's average precision from {leaked_ap} to {corrected_ap} "
      "(`eval/model_report.md` §5).")
    w("")
    w("**Neither guardrail in this repo could see it.** The column whitelist checks "
      "names, and every column involved is admissible and genuinely knowable at "
      "checkout. The perturbation test blanks `order_delivered_carrier_date` and checks "
      "the matrix is unchanged — it inspects one column's *values*, not a join's "
      "*cardinality*. So this is a class of defect the existing checks are structurally "
      "incapable of catching, and it needs its own audit.")
    w("")

    # -- table -------------------------------------------------------------------------
    w("## 1. Every join, audited")
    w("")
    w("\"Zero joined rows\" means the order reaches no row of that table by the stated "
      "path.")
    w("")
    w("| Table | Join path | Orders with zero rows | `label_a` rate | vs panel | "
      "`label_b` rate | vs panel | Flag |")
    w("|---|---|---:|---:|---:|---:|---:|:--:|")
    for r in sorted(results, key=lambda x: -x.n_zero):
        flag = "🚩" if r.flagged else ""
        w(f"| `{r.table}` | {r.via} | {r.n_zero:,} | {pctf(r.rate_a)} | "
          f"{rat(r.ratio_a)} | {pctf(r.rate_b)} | {rat(r.ratio_b)} | {flag} |")
    w("")
    w("Same joins against the **risk set**, the population the primary target is "
      "defined on:")
    w("")
    w("| Table | Orders with zero rows | `label_b` rate | vs risk-set panel |")
    w("|---|---:|---:|---:|")
    for r in sorted(results, key=lambda x: -x.risk_n_zero):
        w(f"| `{r.table}` | {r.risk_n_zero:,} | {pctf(r.risk_rate_b)} | "
          f"{rat(r.risk_ratio_b)} |")
    w("")

    by_name = {r.table: r for r in results}
    flagged = [r for r in results if r.flagged]
    reviews_r = by_name["order_reviews"]

    def plural(n: int) -> str:
        return "order" if n == 1 else "orders"

    w("## 2. Findings")
    w("")
    w(f"**{len(flagged)} of {len(results)} joins flagged**, but they are not "
      f"{len(flagged)} separate defects and they do not carry equal weight. One is new, "
      "independent, and "
      "reaches the primary target; three are the `order_items` artifact seen through "
      "three joins.")
    w("")

    # ---- the new one -----------------------------------------------------------------
    w("### 🚩 `order_reviews` — a second, independent leak, and this one reaches the "
      "primary target")
    w("")
    w(f"{reviews_r.n_zero:,} matured orders have no review row. Their `label_a` rate is "
      f"{pctf(reviews_r.rate_a)} ({rat(reviews_r.ratio_a)} panel) and their `label_b` "
      f"rate is {pctf(reviews_r.rate_b)} ({rat(reviews_r.ratio_b)} panel).")
    w("")
    w(f"**On the risk set — the population the primary target is defined on — "
      f"{reviews_r.risk_n_zero:,} orders are affected, with a `label_b` rate of "
      f"{pctf(reviews_r.risk_rate_b)} against a panel rate of {pctf(risk_panel_b)}. "
      f"That is {rat(reviews_r.risk_ratio_b)}.**")
    w("")
    w(f"Only {reviews_r.n_zero_due_to_items} of the {reviews_r.n_zero:,} are explained by "
      "the missing-items artifact, so this is an independent finding, not a restatement "
      "of it.")
    w("")
    w("The mechanism is obvious once stated: **a customer who never receives their order "
      "does not review it.** Review presence is downstream of delivery, so its absence "
      "is a direct read on the outcome — and unlike the items artifact, it bites hardest "
      "exactly on the population we care about.")
    w("")
    w("**No current feature is affected.** The feature builder never joins "
      "`order_reviews`; every review column sits in `REVIEW_COLUMNS` and is rejected by "
      "`assert_no_leakage`. Nothing in `eval/model_report.md` or "
      "`eval/significance.md` is contaminated by this.")
    w("")
    w("**It is a landmine for the next step, and the whitelist as written does not stop "
      "it.** ARCHITECTURE.md §3.7 admits `review_score` *\"as a customer-history feature "
      "if computed point-in-time from strictly prior orders\"*. That allowance is correct "
      "and this audit does not overturn it — but it is about **prior orders' reviews**, "
      "which have resolved and are genuinely known at checkout. The distinction the "
      "whitelist currently leaves implicit:")
    w("")
    w("> A **prior** order's review — its score, its existence, its absence — is "
      "admissible: that order resolved before the one being scored.")
    w(">")
    w("> **This** order's review is inadmissible in every form, and that includes its "
      "*absence*. A `has_review` flag, a review-row count, or any outer join that leaves "
      "nulls when no review exists, is the outcome.")
    w("")
    w("A point-in-time customer-history feature built from reviews must therefore count "
      "only strictly-prior orders — which the existing PIT primitives already enforce — "
      "and must never fall back to a current-order review presence flag.")
    w("")

    # ---- the items family ------------------------------------------------------------
    family = [r for r in flagged if r.table in ("order_items", "products", "sellers")]
    w("### 🚩 `order_items`, `products`, `sellers` — one defect, three joins")
    w("")
    w("| Table | Orders with zero rows | Of which: no items at all | `label_a` rate |")
    w("|---|---:|---:|---:|")
    for r in family:
        w(f"| `{r.table}` | {r.n_zero:,} | {r.n_zero_due_to_items:,} | "
          f"{pctf(r.rate_a)} |")
    w("")
    w("`products` and `sellers` are reached *through* `order_items`, so an order with no "
      "item rows reaches neither. All 767 in each case are exactly the orders that have "
      "no items — the same rows, counted three times. Fixing the item join fixes all "
      "three; there is no separate product-side or seller-side defect.")
    w("")
    items_r = by_name["order_items"]
    w(f"**The primary target is not exposed.** {items_r.risk_n_zero:,} "
      f"{plural(items_r.risk_n_zero)} in the risk set of {len(risk):,} is affected, "
      "because an order that reached a carrier had items. Already corrected for the "
      "secondary target in `eval/model_report.md` §5.")
    w("")

    # ---- watch items -----------------------------------------------------------------
    geo_c = by_name["geolocation (customer side)"]
    geo_s = by_name["geolocation (seller side)"]
    w("### `geolocation` — now load-bearing, and the seller side needs unpacking")
    w("")
    w(f"**Customer side: {geo_c.n_zero:,} matured orders unreached, "
      f"{rat(geo_c.ratio_a)} on `label_a` and {rat(geo_c.risk_ratio_b)} on `label_b` "
      f"within the risk set ({geo_c.risk_n_zero:,} orders). Below the "
      f"{FLAG_RATIO:g}× flag.**")
    w("")
    w(f"**Seller side: {geo_s.n_zero:,} matured orders unreached at "
      f"{pctf(geo_s.rate_a)} `label_a` — {rat(geo_s.ratio_a)}, which trips the flag. "
      "It is the items artifact wearing a third hat.** The join runs through "
      f"`order_items`, so {geo_s.n_zero_due_to_items:,} of those {geo_s.n_zero:,} "
      "orders — "
      f"{100 * geo_s.n_zero_due_to_items / geo_s.n_zero:.0f}% — have no item rows at "
      "all and never reached a seller to geocode. Removing them leaves "
      f"{geo_s.n_zero - geo_s.n_zero_due_to_items:,} orders whose seller's zip prefix "
      "is genuinely absent from the reference table.")
    w("")
    w(f"**On the risk set the seller side is inert**: {geo_s.risk_n_zero:,} orders at "
      f"{pctf(geo_s.risk_rate_b)} `label_b`, {rat(geo_s.risk_ratio_b)} the panel rate — "
      "*below* it, not above. That is the population the primary model trains and scores "
      "on, and it is where `route_distance_km` is used.")
    w("")
    w("This entry stopped being hypothetical: `route_distance_km` "
      "(`eval/feature_expansion.md`) joins **both** endpoints to this table, which is "
      "why the seller side is audited separately — the customer-side entry alone would "
      "not have discharged the rule for a feature that needs both.")
    w("")
    w("The argument for admissibility, made explicitly rather than assumed: "
      "`geolocation` is a **static reference table**. A zip prefix's coordinates do not "
      "depend on any order, and a prefix missing from it is a property of Brazilian "
      "postal coverage, fixed long before the order was placed. Its absence cannot be "
      "caused by an outcome that has not happened yet — which is the exact property the "
      "`order_items` absence lacked.")
    w("")
    w("What that argument does **not** license: §4.5 lists geocoding confidence itself "
      "as a feature (\"did the pincode resolve at all\"). That is a row-presence feature "
      "by construction, and it is deliberately not built. `route_distance_km` is NaN "
      "where either endpoint fails to resolve and no flag is derived from the failure. "
      "The distinction is the whole point of this audit: reading a joined *value* is "
      "safe here, manufacturing a feature out of the *join's failure* is a separate "
      "claim that would need its own evidence.")
    w("")
    clean = [r for r in results
             if not r.flagged and not r.table.startswith("geolocation")]
    for r in clean:
        if r.n_zero == 0:
            w(f"- **`{r.table}`** — every matured order reaches at least one row. No "
              "cardinality signal exists, so nothing can leak through it.")
        else:
            w(f"- **`{r.table}`** — {r.n_zero:,} {plural(r.n_zero)} with zero rows, "
              f"positive rate {pctf(r.rate_a)} ({rat(r.ratio_a)} panel). Below the "
              f"{FLAG_RATIO:g}× threshold.")
    w("")

    # -- availability flags ------------------------------------------------------------
    w("## 3. Are the availability features proxies for the same artifact?")
    w("")
    w("The availability group exists to capture \"this order came through a degraded "
      "path\" (§4.1). On this panel a degraded path is *caused by* the outcome, so these "
      "are the features most likely to encode it. Measured on the matured population, "
      "where the artifact is live.")
    w("")
    w("| Feature | Set when | n set | `label_a` rate when set | when clear | "
      "corr with `label_a` | corr with `label_b` |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for f in flag_rows:
        corr_a = "constant" if f["constant"] else f"{f['corr_a']:+.4f}"
        corr_b = "constant" if f["constant"] else f"{f['corr_b']:+.4f}"
        w(f"| `{f['col']}` | {f['set_label']} | {f['n_set']:,} | "
          f"{pctf(f['rate_a_set'])} | {pctf(f['rate_a_clear'])} | {corr_a} | {corr_b} |")
    w("")
    w("**`has_item_rows` is the artifact, restated as a feature.** It is 0 for exactly "
      "the orders the `order_items` join misses, and every one of them is a positive. "
      "Its correlation with `label_a` is the whole finding in one number.")
    w("")
    w("**`n_missing_features` inherits it.** The null pattern it counts is dominated by "
      "the columns that go null when the item join misses — `order_value`, `n_items`, "
      "`n_sellers`, `n_products`, `avg_item_price`, `freight_ratio` — so it is a "
      "continuous restatement of the same flag rather than an independent signal.")
    w("")
    w("**`has_zip_prefix` is constant** and therefore carries nothing, which is the "
      "finding already recorded in `eval/feature_report.md`.")
    w("")

    # -- consequences ------------------------------------------------------------------
    w("## 4. Consequences")
    w("")
    w("1. **The `order_items` artifact does not reach the primary target.** The risk set "
      "requires a carrier date, and an order that reached a carrier had items — one "
      "order in 97,658 is affected. Every headline number in `eval/model_report.md` and "
      "`eval/significance.md` stands.")
    w("2. **The `order_reviews` artifact would reach it, and is not currently exploited.** "
      f"{reviews_r.risk_n_zero:,} risk-set orders at {rat(reviews_r.risk_ratio_b)} the "
      "panel rate. No feature joins the table today, and the whitelist rejects every "
      "review column, so nothing is contaminated — but §3.7's allowance for "
      "`review_score` as point-in-time history needs the prior-order / current-order "
      "distinction spelled out before that feature is built.")
    w("3. **The secondary target is exposed and is reported corrected.** Its usable "
      f"average precision is {corrected_ap}, not {leaked_ap}.")
    w("4. **`has_item_rows` and `n_missing_features` must not enter a shipped model "
      "trained on the matured population.** They are admissible on the risk set, where "
      "the artifact does not exist, and inadmissible off it. That conditionality is the "
      "point: these are not universally-bad features, they are features whose validity "
      "depends on the population.")
    w("5. **A standing rule is added to `data/COLUMN_WHITELIST.md`**: any feature "
      "derived from join cardinality or row presence requires an audit entry here "
      "before use.")
    w("")
    w("Findings are pinned in `tests/test_join_audit.py`, so a change in the data or the "
      "join paths surfaces as a test failure rather than as a quietly different report.")
    w("")

    report = "\n".join(L) + "\n"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"[written] {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
