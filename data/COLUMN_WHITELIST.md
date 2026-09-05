# Checkout-time column whitelist

ARCHITECTURE.md §3.7. A column is admissible as a feature only if it is knowable **at
checkout, before the parcel moves**.

**This file is machine-parsed and enforced.** [`data/whitelist.py`](whitelist.py) reads
the "Admissible at checkout" table below, and `OlistLoader.load_table` admits only
columns it finds there — requesting anything else raises `LeakageError`. A test asserts
the parsed document and `CHECKOUT_SAFE_COLUMNS` in [`data/loader.py`](loader.py) are
identical, so the two cannot drift.

Editing the table below changes what the loader will read. It is not documentation of
the rule; it is the rule.

Exclusions live in code — `LABEL_ONLY_COLUMNS`, `POST_CHECKOUT_COLUMNS`,
`REVIEW_COLUMNS` — and are enforced by `OlistLoader.assert_no_leakage`, which every
frame leaving `load_table()` and `checkout_frame()` passes through.

---

## Label-construction columns — never features

These build the target. Reading any of them into a feature matrix is the outcome
leaking into the predictors.

| Column | Role |
|---|---|
| `order_status` | Builds `label_a` (`!= 'delivered'`) and the delivered test in `label_b`. |
| `order_delivered_carrier_date` | **Defines the risk set and `label_b`. Sole legitimate consumer.** See below. |
| `order_delivered_customer_date` | Post-delivery. Not currently used for either label, still inadmissible. |

### `order_delivered_carrier_date` — the load-bearing exclusion

This column has exactly one job: it is the fulfillment flag that defines the risk set,
and therefore the primary target `label_b` and the conditioning in P(fails | ships).

It is a **post-checkout** field. At scoring time the parcel has not moved and the value
does not exist. Any feature derived from it would be reading the future.

The exclusion is enforced two ways:

1. **By name** — it is in `LABEL_ONLY_COLUMNS`, so `assert_no_leakage` rejects any frame
   carrying it.
2. **By behaviour** — `tests/test_leakage.py::test_carrier_date_is_label_only` perturbs
   the column on a copy of the raw data, rebuilds, and asserts that `label_b` changes
   while `checkout_frame()` is byte-identical. A name check cannot catch a feature
   silently *derived* from a forbidden column; this can.

That the column is simultaneously required (for the label) and forbidden (for features)
is the whole reason it gets its own test rather than sitting in a list.

---

## Post-checkout columns — not labels, still inadmissible

| Column | Why |
|---|---|
| `order_approved_at` | Payment authorisation lands after checkout. |

### Two columns audited and admitted

`order_estimated_delivery_date` and `shipping_limit_date` were previously listed here on
the assumption that they land after checkout. **That assumption was wrong and the audit
is recorded rather than the reversal being made quietly** (`eval/feature_expansion.md`
§1).

The decisive test for a backfilled field is whether it is populated for orders that never
progressed. Both are non-null for **all 99,441 orders across all eight statuses**,
including the 5 `created` and 609 `unavailable` orders that never shipped — a field
written at delivery could not be. Neither has a single negative span against the purchase
timestamp.

| Column | Evidence it is set at purchase |
|---|---|
| `order_estimated_delivery_date` | 0 nulls in every status; median promised span 23d; 91% of orders arrive early with 11.3d slack; correlation with the label 0.019 |
| `shipping_limit_date` | 0 nulls in every status; median contractual window 6d; 90.5% dispatched before it; correlation with actual dispatch delay 0.209 |

Both are promises made at order placement, not records of what happened. They are
admissible, and the derived features are `promised_days` and
`seller_median_dispatch_window`.

## Review columns — post-outcome

`review_id`, `review_score`, `review_comment_title`, `review_comment_message`,
`review_creation_date`, `review_answer_timestamp`.

All are written after the order resolves. `review_score` is admissible **only** as a
point-in-time customer-history feature computed from strictly prior orders — a step 2
concern, and not currently built.

---

## Admissible at checkout

Declared in `CHECKOUT_SAFE_COLUMNS`. Only `orders` and `customers` are materialised
today; the rest are declared so that the whitelist stays the single place the rule lives
when step 2 joins them.

| Table | Columns | Materialised |
|---|---|---|
| `orders` | `order_id`, `customer_id`, `order_purchase_timestamp`, `order_estimated_delivery_date` | yes |
| `customers` | `customer_id`, `customer_unique_id`, `customer_zip_code_prefix`, `customer_city`, `customer_state` | yes |
| `payments` | `order_id`, `payment_sequential`, `payment_type`, `payment_installments`, `payment_value` | no |
| `items` | `order_id`, `order_item_id`, `product_id`, `seller_id`, `price`, `freight_value`, `shipping_limit_date` | yes |
| `products` | `product_id`, `product_category_name`, `product_name_lenght`, `product_description_lenght`, `product_photos_qty`, `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm` | no |
| `sellers` | `seller_id`, `seller_zip_code_prefix`, `seller_city`, `seller_state` | yes |
| `geolocation` | `geolocation_zip_code_prefix`, `geolocation_lat`, `geolocation_lng`, `geolocation_city`, `geolocation_state` | yes |

`geolocation` is a static reference table - a zip prefix's coordinates do not depend
on any order and cannot move after one is placed. It carries repeated rows per prefix, so
it is reduced to a per-prefix median centroid before use; it is the one whitelisted table
whose key is **not** unique, and it is never merged, only mapped.

`order_id` and `customer_id` are keys, not features. `order_purchase_timestamp` is
admissible because it *is* checkout time — it is also what the point-in-time feature
construction in step 2 will cut every history window against.

### A note on `payments`

Payment rows are knowable at checkout, but on this panel they are recorded once per
order after the fact. Treating `payment_value` as a checkout-time field assumes the
payment intent is known when the order is placed, which holds for `payment_type` and
`payment_installments` and is weaker for the value on multi-instrument orders. Flagged
here rather than in a comment because step 2 has to decide it.

---

## Join cardinality — a leak the whitelist cannot catch

**`has_item_rows` is not a usable feature on Olist.** Neither is the null pattern it
drives across `n_items`, `order_value`, `n_sellers`, `n_products`, `avg_item_price`,
`freight_ratio`, and `n_missing_features`.

767 matured orders have no rows in `olist_order_items_dataset.csv`, and **100% of them
are `label_a` positives** — 98 of them in the test window, which is 25.9% of that
target's test positives. The absence is a *consequence* of the outcome: an order that
resolves to `unavailable` has no item rows in the export, but at the moment of checkout
the customer had items in the cart and those rows would exist.

Measured effect (`eval/model_report.md` §5): the secondary target's average precision
falls from **0.2938 to 0.0190** once orders that do not join to items are excluded, and
precision@1% falls from 50.75% to 2.53%. The apparent result was entirely the artifact.

**The primary target is unaffected** — the risk set contains exactly one order without
item rows, because an order that reached a carrier had items.

Why no guardrail caught it: every column involved is on the whitelist and genuinely
knowable at checkout. What leaks is the **cardinality of a join**, which neither the
name-based check nor the carrier-date perturbation test inspects. Both of those
guardrails work as specified and neither was capable of seeing this.

The general rule this implies, for any table joined in later: **a row's absence from a
source table is evidence about that row, and if the absence is caused by the outcome it
is leakage regardless of which columns the join produces.** Check the join rate against
the label before trusting any feature derived from an outer join.

### STANDING RULE — join cardinality requires an audit entry

> **Any feature derived from join cardinality or row presence — a `has_*` flag, a row
> count, a null-count, or any outer join whose nulls are informative — requires an
> explicit entry in [`eval/join_cardinality_audit.md`](../eval/join_cardinality_audit.md)
> before use.**
>
> The entry must state the number of orders with zero joined rows and their positive rate
> against the panel rate, **for the population the feature will be used on**. A join that
> is safe on one population is not thereby safe on another: `has_item_rows` is inert on
> the risk set and catastrophic on the matured set.
>
> Neither the name check nor the perturbation test can discharge this. They inspect
> column names and column values; this is a property of a join.

Regenerate the audit with `python scripts/06_join_audit.py`. Findings are pinned in
`tests/test_join_audit.py`.

### `order_reviews` — admissible only from strictly prior orders

The audit found a second, independent leak in `order_reviews`, and unlike the items
artifact **it reaches the primary target**: 731 risk-set orders have no review row and
their `label_b` rate is 11.63% against a panel rate of 1.21% — **9.6×**. A customer who
never receives an order does not review it, so review presence is downstream of delivery.

No current feature is affected — the feature builder never joins the table, and every
review column is in `REVIEW_COLUMNS`. But §3.7 of ARCHITECTURE.md admits `review_score`
*"as a customer-history feature if computed point-in-time from strictly prior orders"*,
and that allowance needs its boundary stated:

> A **prior** order's review — score, existence, or absence — is admissible. That order
> resolved before the one being scored, so all of it is knowable at checkout.
>
> **This** order's review is inadmissible in every form, **including its absence**. A
> `has_review` flag, a review row count, or an outer join that leaves nulls where no
> review exists, is the outcome wearing a feature's name.

## Group leakage

ARCHITECTURE.md §3.7 asks for an assertion that no `customer_unique_id` straddles train
and test, on the reasoning that "temporal splitting mostly handles it."

**On this panel it does not.** 472 `customer_unique_id` values have orders on both sides
of the boundary (461 within the risk set). The assertion as written in §3.7 would fail.

`tests/test_split.py` asserts the measured count instead, so the number cannot drift
unnoticed. The straddle is not itself feature leakage: §4.3 requires every history
feature be computed point-in-time from strictly prior orders, under which a repeat
customer's test-period order sees only their own earlier history and nothing from the
test window. It would be leakage under k-fold out-of-fold target encoding, which §4.3
rejects for exactly this reason. The count is documented so that if PIT construction is
ever relaxed, the exposure is already quantified.
