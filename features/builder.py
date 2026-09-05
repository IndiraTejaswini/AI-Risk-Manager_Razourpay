"""
Point-in-time feature builder, Olist-populated groups only.

Groups built here (ARCHITECTURE.md 4.1).  SHIPPED - what ``FeatureBuilder()`` builds:

    order         value, freight, item/seller counts, payment shape, timing
    customer      expanding windows over strictly-prior orders
    pincode       smoothed target encoding, point-in-time
    availability  source-presence flags and n_missing_features

BUILT AND NOT SHIPPED - reachable only by naming them (see :data:`DEFAULT_GROUPS`):

    structure     parcel shape, and the two promises made at checkout (promised
                  delivery span, contractual dispatch window)
    seller        smoothed target encoding over the order's sellers, point-in-time
    route         origin-destination: states, same-state, pair encoding, distance
    density       rolling 24h/7d volume in the pincode and at the seller

The second block passes every gate the first does - it is excluded because a paired
bootstrap could not distinguish the two models, not because it is unfinished.
eval/feature_expansion.md is the report.

Groups deliberately NOT built (ARCHITECTURE.md 4.2): **address** and **network/ring**.
Olist carries no address strings, device identifiers, or phone numbers, so those
extractors are written against the production schema and exercised by the demo harness
only.  They are excluded from this matrix and produce no reported metric.

--------------------------------------------------------------------------------------
WHY HISTORY FEATURES ENCODE label_a AND NOT THE PRIMARY TARGET
--------------------------------------------------------------------------------------

The customer and pincode encodings aggregate the *outcomes of strictly-prior orders*.
That is what target encoding is, and it is legitimate: at checkout for the order at time
t, prior orders have already resolved.

They encode ``label_a`` (``order_status != 'delivered'``), not the primary target
``label_b``.  label_b is defined by ``order_delivered_carrier_date``, and encoding it
would make a feature depend on that column transitively - through prior rows rather than
through the row being scored, but a dependence nonetheless.

Using label_a keeps the invariant absolute and testable: **no feature reads
order_delivered_carrier_date, directly or transitively, ever.**  That is what
tests/test_feature_builder.py asserts per group by blanking the column and requiring the
matrix to be byte-identical.

The cost is that the encoding targets a slightly broader outcome than the model
predicts - "failed to be delivered" rather than "shipped and failed to be delivered".
As a *predictor* that is fine and arguably preferable: it is a strictly larger and
better-populated signal, and it carries no coupling to the target's defining column.
Stated here because it is a real choice, not an accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.loader import OlistLoader
from features.pit import (
    pit_expanding_mean,
    pit_prior_count,
    pit_prior_count_asof,
    pit_prior_first,
    pit_prior_last,
    pit_prior_sum,
    pit_smoothed_rate,
)
from features.store import HistoryStore

__all__ = [
    "FeatureBuilder",
    "STORE_GROUP_COLS",
    "STORE_VALUE_COLS",
    "FEATURE_GROUPS",
    "DEFAULT_GROUPS",
    "EXPANSION_GROUPS",
    "MULTI_SELLER_RULE",
]

FEATURE_GROUPS = (
    "order", "customer", "pincode", "structure", "seller", "route", "density",
    "availability",
)

#: The groups added by the seller/route expansion, named so the ablation that decided
#: whether they ship stays reproducible (eval/feature_expansion.md).
#:
#: `structure` is separate from `order` rather than folded into it precisely so that
#: this list is the whole of what the ablation adds.  Putting the parcel-shape and
#: promise features into the existing order group would have put them on BOTH sides of
#: the comparison and quietly changed what "current" means.
EXPANSION_GROUPS = ("structure", "seller", "route", "density")

#: What ``FeatureBuilder()`` builds when the caller names no groups - the shipped
#: matrix.
#:
#: The expansion groups are NOT in it.  They were built, gated, and evaluated against
#: the current matrix on a paired bootstrap (eval/feature_expansion.md); the interval
#: on the difference includes zero, so the decision rule fixed before the run says the
#: current model stays.  They remain buildable by name so the ablation is reproducible
#: and so a larger panel can resolve it later.
#:
#: Changing this line changes the shipped model.  It is the one place that decision
#: lives.
DEFAULT_GROUPS: tuple[str, ...] = tuple(
    g for g in FEATURE_GROUPS if g not in EXPANSION_GROUPS
)

TS = "order_purchase_timestamp"

#: Pseudo-observations at the global rate.  Deliberately strong: a pincode with a
#: handful of prior orders must not take an extreme value (ARCHITECTURE.md 4.5).
PINCODE_SMOOTHING = 50.0
CUSTOMER_SMOOTHING = 20.0
#: Same value and the same argument as the pincode encoding: a seller or a state pair
#: seen a handful of times must not take an extreme value.
SELLER_SMOOTHING = 50.0
ROUTE_SMOOTHING = 50.0

#: Rolling-window widths for the density group.
DENSITY_WINDOWS = {"24h": pd.Timedelta(days=1), "7d": pd.Timedelta(days=7)}

#: MULTI-SELLER AGGREGATION - stated, not left to whichever row a groupby happened to
#: keep.  1,277 of the 97,658 risk-set orders (1.3%) carry more than one seller.
#:
#:   risk, volume   MAX across the order's sellers.  label_b is an order-level outcome
#:                  and a multi-seller order is several parcels: if any one of them
#:                  fails to arrive the order is a failure.  The order therefore
#:                  inherits its worst seller, not its average one.  A volume weighting
#:                  would dilute precisely the seller the feature exists to catch.
#:   prior orders   MIN.  The least-established seller is the exposure.
#:   tenure,        MIN / MAX respectively, and **NaN if any seller in the order has no
#:   dispatch       prior history** rather than the aggregate of those that do.  NaN
#:                  here means "unknown for at least one seller", which is a different
#:                  statement from a low value, and the booster branches on it natively.
#:   route pair     the DOMINANT seller - the one on the highest-priced item, the same
#:                  rule already used to pick the order's product category.  A pair of
#:                  states is not meaningfully averaged.
MULTI_SELLER_RULE = {
    "seller_failure_rate_smoothed": "max",
    "seller_orders_7d": "max",
    "seller_prior_orders": "min",
    "seller_tenure_days": "min, NaN if any seller is new",
    "seller_prior_dispatch_window_mean": "max, NaN if any seller is new",
    "route_pair": "dominant seller (highest-priced item)",
}

#: The columns the history store must index for build(..., store=...) to answer the
#: same questions as the row-scan.  Held here, beside the features that consume them,
#: because a store constructed with the wrong spec produces a *silently* different
#: matrix - the failure the startup self-check exists to catch, and one that a spec
#: retyped at each call site invites.
#: Grouping column -> the feature groups that query it.  An index is built only when
#: one of its consumers is active: the shipped matrix has no route or seller group, and
#: building their indexes at startup would cost time and memory nothing then reads.
STORE_GROUP_COLS: dict[str, tuple[str, ...]] = {
    "customer_unique_id": ("customer",),
    "customer_zip_code_prefix": ("pincode", "density"),
    "route_pair": ("route",),
}
STORE_VALUE_COLS = ("label_a", "order_value", "is_boleto")

#: Seller history aggregates across an order's sellers, so its prior window is over
#: order-seller edges rather than over orders.
STORE_EDGE_GROUP_COLS: dict[str, tuple[str, ...]] = {
    "seller_id": ("seller", "density"),
}
STORE_EDGE_VALUE_COLS = ("label_a", "dispatch_window_days")

#: Metres-cubed-to-grams divisor for dimensional weight.  5000 cm3/kg is the standard
#: domestic courier divisor; the constant is named rather than inlined because changing
#: it changes a feature.
VOLUMETRIC_DIVISOR_CM3_PER_KG = 5000.0

#: History encodings aggregate this column over strictly-prior rows.  See module
#: docstring for why it is label_a and not the primary target.
HISTORY_TARGET = "label_a"


def _haversine_km(lat1, lng1, lat2, lng2) -> pd.Series:
    """
    Great-circle distance in kilometres.  NaN where either endpoint fails to resolve.

    Straight-line, not road distance - Brazil is large and the two diverge, but the
    feature is a proxy for "how far is this parcel travelling" and the ordering is what
    the model uses.
    """
    r = 6371.0088
    a1, o1, a2, o2 = (
        np.radians(np.asarray(v, dtype="float64")) for v in (lat1, lng1, lat2, lng2)
    )
    h = (
        np.sin((a2 - a1) / 2.0) ** 2
        + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2.0) ** 2
    )
    return pd.Series(
        2.0 * r * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))), index=lat1.index
    )


class FeatureBuilder:
    """
    Builds the point-in-time feature matrix for a population.

    The population is a frame from :class:`~data.loader.OlistLoader` - ``risk_set()`` for
    the primary target, or ``labelled()`` for the secondary.  Output is row-aligned to
    the input index and carries no label, no key, and no post-checkout column.

    Parameters
    ----------
    groups:
        Which feature groups to build.  Defaults to :data:`DEFAULT_GROUPS` - the shipped
        matrix, which is not all of :data:`FEATURE_GROUPS`: the expansion groups are
        buildable by name but are not in the shipped model.  See
        eval/feature_expansion.md.
    loader:
        Loader used to read source tables.  One is constructed if not supplied.
    leak_for_testing:
        Adds a single deliberately non-PIT aggregate,
        ``leaky_customer_failure_rate``, computed over the whole panel rather than
        strictly-prior rows.  Used only by tests/test_pit.py to prove the truncation
        check is capable of failing.  Never set this in a pipeline.
    """

    def __init__(
        self,
        groups: tuple[str, ...] = DEFAULT_GROUPS,
        loader: OlistLoader | None = None,
        leak_for_testing: bool = False,
    ) -> None:
        unknown = set(groups) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"unknown feature group(s): {sorted(unknown)}")
        self.groups = tuple(g for g in FEATURE_GROUPS if g in set(groups))
        self.loader = loader or OlistLoader()
        self.leak_for_testing = leak_for_testing
        # Indexed copies of the m:1 lookup tables, built on first use. A pandas merge
        # factorises the whole right frame every call - ~47ms to attach three columns
        # to a single row - while reindex on a unique index is a hash lookup. On a
        # unique key the two produce identical output, and the loader asserts that
        # uniqueness when the table is read.
        self._indexed: dict[str, pd.DataFrame] = {}
        self._geo: tuple[pd.Series, pd.Series] | None = None
        self._expansion_active = bool(set(self.groups) & set(EXPANSION_GROUPS))

    def _attach(self, base, table: str, key: str, columns: list[str]):
        """
        Left-attach an m:1 lookup table onto ``base``.

        Two strategies, identical in output on a unique key and chosen on population
        size.  ``merge`` factorises the whole right frame, which is the right trade when
        the left side is large but costs ~47ms to attach three columns to one row.
        ``join`` against a cached key index is a hash lookup per row, which wins for a
        request and loses for a full-panel build.
        """
        if len(base) * 8 < len(self.loader.load_table(table, columns)):
            if table not in self._indexed:
                self._indexed[table] = self.loader.load_table(
                    table, columns
                ).set_index(key)
            return base.join(self._indexed[table], on=key)
        return base.merge(self.loader.load_table(table, columns), on=key, how="left")

    # -- source assembly ---------------------------------------------------------------

    def _sources(self, population: pd.DataFrame) -> pd.DataFrame:
        """Join the checkout-safe source tables onto the population, once."""
        base = population[[c for c in ("order_id", "customer_id", TS) if c in population]]
        base = base.copy()
        if HISTORY_TARGET in population:
            base[HISTORY_TARGET] = population[HISTORY_TARGET].astype("float64")

        base = self._attach(
            base, "customers", "customer_id",
            ["customer_id", "customer_unique_id", "customer_zip_code_prefix",
             "customer_state"],
        )

        # Restrict the one-to-many tables to this population before aggregating.
        # The aggregate is per order_id, so filtering first is an identity - but the
        # filter itself costs a hash lookup per source row, so it only pays when the
        # population is a small slice.  Serving one order was grouping all 112k item
        # rows; batch over the whole panel is faster without the filter.
        order_ids = pd.Index(base["order_id"].unique())
        items = self.loader.load_table("items")
        if len(order_ids) * 4 < len(items):
            items = items[items["order_id"].isin(order_ids)]
        agg = items.groupby("order_id").agg(
            order_value=("price", "sum"),
            order_freight=("freight_value", "sum"),
            n_items=("order_item_id", "count"),
            n_sellers=("seller_id", "nunique"),
            n_products=("product_id", "nunique"),
        )
        # Force float64 regardless of population. A left merge yields int64 when every
        # order in the population happens to join and float64 when any does not, so
        # the dtype would otherwise depend on which rows are being scored - a serving
        # frame of one order could differ in dtype from the training matrix.
        agg = agg.astype("float64")
        base = base.merge(agg, on="order_id", how="left")

        # Dominant product = the most expensive item, deterministic on ties by product_id.
        dominant = (
            items.sort_values(["order_id", "price", "product_id"], kind="mergesort")
            .groupby("order_id")
            .tail(1)[["order_id", "product_id"]]
        )
        dominant = self._attach(
            dominant, "products", "product_id",
            [
                "product_id",
                "product_category_name",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
                "product_photos_qty",
            ],
        )
        base = base.merge(dominant.drop(columns=["product_id"]), on="order_id", how="left")

        # Everything the expansion groups need, and nothing else needs, is built only
        # when one of them is active.  `_sources` runs on every batch build AND on every
        # scoring request, so the shipped matrix must not pay for a products join, a
        # sellers attach and a shipping-limit aggregate it never reads.
        if self._expansion_active:
            base = self._expansion_sources(base, items)

        payments = self.loader.load_table("payments")
        if len(order_ids) * 4 < len(payments):
            payments = payments[payments["order_id"].isin(order_ids)]
        pay = payments.groupby("order_id").agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            n_payment_types=("payment_type", "nunique"),
        )
        pay["is_boleto"] = (
            payments[payments["payment_type"] == "boleto"]
            .groupby("order_id")
            .size()
            .reindex(pay.index)
            .notna()
        )
        pay = pay.astype({"payment_value": "float64",
                          "payment_installments": "float64",
                          "n_payment_types": "float64"})
        base = base.merge(pay, on="order_id", how="left")

        if self._expansion_active:
            if "order_estimated_delivery_date" in population.columns:
                promised = pd.to_datetime(population["order_estimated_delivery_date"])
                base["promised_days"] = (
                    (promised.to_numpy() - population[TS].to_numpy())
                    / np.timedelta64(1, "D")
                )

            # Origin-destination pair, on the dominant seller.  Held as a source column
            # so the history store can index it like any other grouping key.
            base["route_pair"] = (
                base["customer_state"].astype("object").fillna("__na__")
                + "->"
                + base["seller_state"].astype("object").fillna("__na__")
            )

        base.index = population.index
        return base

    def _expansion_sources(
        self, base: pd.DataFrame, items: pd.DataFrame
    ) -> pd.DataFrame:
        """Source columns read only by the structure, seller and route groups."""
        # Distinct categories in the order.  Item-level, so it is not the dominant
        # product's category repeated - it counts how mixed the basket is.
        cats = (
            items[["order_id", "product_id"]]
            .merge(
                self.loader.load_table(
                    "products", ["product_id", "product_category_name"]
                ),
                on="product_id", how="left",
            )
            .groupby("order_id")["product_category_name"]
            .nunique()
            .rename("n_categories")
            .astype("float64")
        )
        base = base.merge(cats, on="order_id", how="left")

        # Dominant seller, same rule as the dominant product: the seller on the
        # highest-priced item, tie-broken deterministically by seller_id.
        dom_seller = (
            items.sort_values(["order_id", "price", "seller_id"], kind="mergesort")
            .groupby("order_id")
            .tail(1)[["order_id", "seller_id"]]
        )
        dom_seller = self._attach(
            dom_seller, "sellers", "seller_id",
            ["seller_id", "seller_state", "seller_zip_code_prefix"],
        ).rename(columns={"seller_id": "dominant_seller_id"})
        base = base.merge(dom_seller, on="order_id", how="left")

        # Contractual dispatch window: the seller's deadline to hand the parcel over,
        # minus checkout.  The LATEST limit across the order's items, because the order
        # is not dispatched until its slowest line is.  This is the promise, not the
        # outcome - see the note in _seller_features.
        limits = items.groupby("order_id")["shipping_limit_date"].max()
        base = base.merge(limits.rename("_limit"), on="order_id", how="left")
        base["dispatch_window_days"] = (
            (pd.to_datetime(base["_limit"]).to_numpy() - base[TS].to_numpy())
            / np.timedelta64(1, "D")
        )
        base = base.drop(columns=["_limit"])
        return base

    def edge_frame(self, src: pd.DataFrame) -> pd.DataFrame:
        """
        One row per (order, seller) - the frame the seller history is aggregated over.

        Seller risk is a max across an order's sellers (:data:`MULTI_SELLER_RULE`), so
        its prior window is over edges rather than over orders: indexing it on the order
        row would count a two-seller order once and attribute it to whichever seller
        happened to be dominant.

        Exposed rather than private because the serving path builds the store's seller
        index from it, and that index must be the same shape as the frame the batch path
        scans or the two answer different questions.
        """
        items = self.loader.load_table("items")
        ids = pd.Index(src["order_id"].unique())
        if len(ids) * 4 < len(items):
            items = items[items["order_id"].isin(ids)]
        carry = [c for c in ("order_id", TS, HISTORY_TARGET, "dispatch_window_days")
                 if c in src.columns]
        return (
            items[["order_id", "seller_id"]]
            .drop_duplicates()
            .merge(src[carry], on="order_id", how="inner")
            .reset_index(drop=True)
        )

    def history_store(self, src: pd.DataFrame) -> HistoryStore:
        """
        The prior-window index this builder's ``store=`` path expects.

        One constructor, so the serving path and the tests cannot index a different set
        of columns from the one ``build()`` queries - and scoped to the groups this
        builder actually has, so a shipped service does not pay to index the seller and
        route history it will never read.
        """
        active = set(self.groups)
        group_cols = tuple(
            col for col, consumers in STORE_GROUP_COLS.items()
            if active.intersection(consumers)
        )
        edge_cols = tuple(
            col for col, consumers in STORE_EDGE_GROUP_COLS.items()
            if active.intersection(consumers)
        )
        return HistoryStore(
            src,
            group_cols=group_cols,
            value_cols=STORE_VALUE_COLS,
            edges=self.edge_frame(src) if edge_cols else None,
            edge_group_cols=edge_cols,
            edge_value_cols=STORE_EDGE_VALUE_COLS,
        )

    def _centroids(self) -> tuple[pd.Series, pd.Series]:
        """
        Median lat/lng per zip prefix.

        `geolocation` is the one whitelisted table whose key is not unique - it holds
        one row per geocoded address, thousands per prefix.  The median is used rather
        than the mean because the table carries a handful of coordinates that are not in
        Brazil at all, and a mean would drag the whole prefix toward them.
        """
        if self._geo is None:
            geo = self.loader.load_table(
                "geolocation",
                ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"],
            )
            g = geo.groupby("geolocation_zip_code_prefix").agg(
                lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median")
            )
            self._geo = (g["lat"], g["lng"])
        return self._geo

    # -- groups ------------------------------------------------------------------------

    def _order_features(self, src: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=src.index)
        value = src["order_value"]
        freight = src["order_freight"]

        f["order_value"] = value
        f["order_freight"] = freight
        f["freight_ratio"] = freight / (value + freight).replace(0, np.nan)
        f["log_order_value"] = np.log1p(value.clip(lower=0))
        f["n_items"] = src["n_items"]
        f["n_sellers"] = src["n_sellers"]
        f["n_products"] = src["n_products"]
        f["avg_item_price"] = value / src["n_items"].replace(0, np.nan)

        f["payment_value"] = src["payment_value"]
        f["payment_installments"] = src["payment_installments"]
        f["n_payment_types"] = src["n_payment_types"]
        f["is_boleto"] = src["is_boleto"].astype("float64")

        f["product_weight_g"] = src["product_weight_g"]
        f["product_volume_cm3"] = (
            src["product_length_cm"] * src["product_height_cm"] * src["product_width_cm"]
        )
        f["product_photos_qty"] = src["product_photos_qty"]
        f["product_category"] = src["product_category_name"].astype("object")

        ts = src[TS]
        f["purchase_hour"] = ts.dt.hour.astype("int64")
        f["purchase_dow"] = ts.dt.dayofweek.astype("int64")
        f["purchase_month"] = ts.dt.month.astype("int64")
        f["purchase_is_weekend"] = (ts.dt.dayofweek >= 5).astype("int64")
        return f

    def _structure_features(self, src: pd.DataFrame) -> pd.DataFrame:
        """Parcel shape and the two promises made at checkout."""
        f = pd.DataFrame(index=src.index)
        volume = (
            src["product_length_cm"] * src["product_height_cm"] * src["product_width_cm"]
        )
        # Couriers bill on whichever of actual and dimensional weight is larger, so a
        # high ratio is a bulky-but-light parcel - the shape that gets mishandled.
        f["volumetric_weight_g"] = volume / VOLUMETRIC_DIVISOR_CM3_PER_KG * 1000.0
        f["dim_weight_ratio"] = f["volumetric_weight_g"] / src[
            "product_weight_g"
        ].replace(0, np.nan)
        f["freight_per_item"] = src["order_freight"] / src["n_items"].replace(0, np.nan)
        f["n_categories"] = src["n_categories"]

        # Audited into the whitelist rather than assumed admissible:
        # data/COLUMN_WHITELIST.md, "Two columns audited and admitted".
        f["promised_days"] = (
            src["promised_days"] if "promised_days" in src.columns else np.nan
        )
        f["dispatch_window_days"] = src["dispatch_window_days"]
        return f

    def _customer_features(self, src: pd.DataFrame, store=None) -> pd.DataFrame:
        f = pd.DataFrame(index=src.index)
        key = "customer_unique_id"

        n_prior = pit_prior_count(src, key, TS, store)
        f["cust_prior_orders"] = n_prior
        f["cust_prior_failures"] = pit_prior_sum(src, key, TS, HISTORY_TARGET, store)
        f["cust_prior_failure_rate"] = pit_smoothed_rate(
            src, key, TS, HISTORY_TARGET, CUSTOMER_SMOOTHING, store
        )

        prior_value_sum = pit_prior_sum(src, key, TS, "order_value", store)
        f["cust_prior_avg_value"] = prior_value_sum / n_prior.where(n_prior > 0)

        last_ts = pit_prior_last(src, key, TS, TS, store)
        f["cust_days_since_prior_order"] = (src[TS] - last_ts).dt.total_seconds() / 86400.0

        prior_boleto = pit_prior_sum(src, key, TS, "is_boleto", store)
        f["cust_prior_boleto_ratio"] = prior_boleto / n_prior.where(n_prior > 0)

        if self.leak_for_testing:
            # NOT point-in-time: whole-panel mean per customer, no prior restriction.
            # Exists solely so tests can prove the truncation check catches a leak.
            f["leaky_customer_failure_rate"] = src.groupby(key)[
                HISTORY_TARGET
            ].transform("mean")
        return f

    def _pincode_features(self, src: pd.DataFrame, store=None) -> pd.DataFrame:
        f = pd.DataFrame(index=src.index)
        key = "customer_zip_code_prefix"

        f["pincode_prior_orders"] = pit_prior_count(src, key, TS, store)
        f["pincode_prior_failures"] = pit_prior_sum(src, key, TS, HISTORY_TARGET, store)
        f["pincode_failure_rate_smoothed"] = pit_smoothed_rate(
            src, key, TS, HISTORY_TARGET, PINCODE_SMOOTHING, store
        )
        f["global_prior_failure_rate"] = pit_expanding_mean(
            src, TS, HISTORY_TARGET, store
        )
        return f

    def _seller_features(
        self, src: pd.DataFrame, edges: pd.DataFrame, store=None
    ) -> pd.DataFrame:
        """
        Seller history, point-in-time, aggregated across the order's sellers.

        ``seller_prior_dispatch_window_mean`` is built from the **contractual** window
        (``shipping_limit_date - purchase``), not from the seller's realised dispatch
        delay.  The realised delay is ``order_delivered_carrier_date - purchase``, and
        reading that column - even over strictly prior orders, even in aggregate - would
        break the invariant that no feature touches it, which
        tests/test_feature_builder.py enforces by perturbation on every group including
        this one.  The contractual window is the admissible substitute
        and is named as one; it is a promise the seller made, not a record of what they
        did.  It is also a mean and not a median: the prior-window index answers counts
        and sums, and a point-in-time median would need a second data structure serving
        a second definition.

        Both deviations are stated in eval/feature_expansion.md rather than absorbed
        into the feature name.
        """
        f = pd.DataFrame(index=src.index)
        if edges.empty:
            for name in ("seller_failure_rate_smoothed", "seller_prior_orders",
                         "seller_tenure_days", "seller_prior_dispatch_window_mean"):
                f[name] = np.nan
            return f

        e = edges.copy()
        # The global prior the encoding smooths toward is the global ORDER failure rate.
        # Recomputing it on the edge frame would double-count multi-seller orders and
        # would not match the store, whose global index is built from orders.
        gp = pit_expanding_mean(src, TS, HISTORY_TARGET, store)
        e["_gp"] = e["order_id"].map(
            pd.Series(gp.to_numpy(), index=src["order_id"].to_numpy())
        )

        n_prior = pit_prior_count(e, "seller_id", TS, store).astype("float64")
        rate = pit_smoothed_rate(
            e, "seller_id", TS, HISTORY_TARGET, SELLER_SMOOTHING, store,
            global_prior=e["_gp"],
        )
        first = pit_prior_first(e, "seller_id", TS, store)
        tenure = (e[TS] - first).dt.total_seconds() / 86400.0
        win = pit_prior_sum(
            e, "seller_id", TS, "dispatch_window_days", store
        ) / n_prior.where(n_prior > 0)

        agg = pd.DataFrame({
            "order_id": e["order_id"].to_numpy(),
            "rate": rate.to_numpy(),
            "n": n_prior.to_numpy(),
            "ten": tenure.to_numpy(),
            "win": win.to_numpy(),
        })
        agg["_ten_na"] = agg["ten"].isna().astype("float64")
        agg["_win_na"] = agg["win"].isna().astype("float64")
        g = agg.groupby("order_id", sort=False)
        out = pd.DataFrame({
            "seller_failure_rate_smoothed": g["rate"].max(),
            "seller_prior_orders": g["n"].min(),
            # NaN if ANY seller on the order is new - see MULTI_SELLER_RULE.
            "seller_tenure_days": g["ten"].min().where(g["_ten_na"].max() == 0),
            "seller_prior_dispatch_window_mean": (
                g["win"].max().where(g["_win_na"].max() == 0)
            ),
        })
        return self._align(f, src, out)

    def _route_features(self, src: pd.DataFrame, store=None) -> pd.DataFrame:
        """
        Origin-destination features on the dominant seller.

        `customer_state` and `seller_state` enter as categoricals - geography under its
        own name, which nothing in the shipped matrix carries.  eval/fairness.md 4
        measures how much geography the shipped model already sees *implicitly* (the
        strongest carrier is order_freight, not any feature named for location); this
        group is the other half of that question, and eval/feature_expansion.md answers
        it: naming geography explicitly made `customer_state` the most-attributed
        feature in the model and made the model worse.

        Because these do not ship, the fairness tables and the pincode ablation are
        unchanged - they describe the model that is deployed, which is still the one
        without them.
        """
        f = pd.DataFrame(index=src.index)
        cs = src["customer_state"].astype("object")
        ss = src["seller_state"].astype("object")
        f["customer_state"] = cs
        f["seller_state"] = ss
        f["same_state"] = (cs == ss).astype("float64").where(cs.notna() & ss.notna())

        f["route_pair_prior_orders"] = pit_prior_count(src, "route_pair", TS, store)
        f["route_pair_failure_rate_smoothed"] = pit_smoothed_rate(
            src, "route_pair", TS, HISTORY_TARGET, ROUTE_SMOOTHING, store
        )

        lat, lng = self._centroids()
        f["route_distance_km"] = _haversine_km(
            src["customer_zip_code_prefix"].map(lat),
            src["customer_zip_code_prefix"].map(lng),
            src["seller_zip_code_prefix"].map(lat),
            src["seller_zip_code_prefix"].map(lng),
        )
        return f

    def _density_features(
        self, src: pd.DataFrame, edges: pd.DataFrame, store=None
    ) -> pd.DataFrame:
        """
        Rolling prior volume, in the pincode and at the order's busiest seller.

        Each window is the difference of two prior-counts - orders before *t*, minus
        orders before *t - w* - so the tie rule is inherited from the primitive rather
        than restated here.  Nothing in the window is observed at or after *t*.
        """
        f = pd.DataFrame(index=src.index)
        pin_total = pit_prior_count(src, "customer_zip_code_prefix", TS, store)
        for tag, width in DENSITY_WINDOWS.items():
            before = pit_prior_count_asof(
                src, "customer_zip_code_prefix", TS, width, store
            )
            f[f"pincode_orders_{tag}"] = (pin_total - before).astype("float64")

        if edges.empty:
            f["seller_orders_7d"] = np.nan
            return f

        e_total = pit_prior_count(edges, "seller_id", TS, store)
        e_before = pit_prior_count_asof(
            edges, "seller_id", TS, DENSITY_WINDOWS["7d"], store
        )
        agg = pd.DataFrame({
            "order_id": edges["order_id"].to_numpy(),
            "v": (e_total - e_before).to_numpy().astype("float64"),
        })
        out = agg.groupby("order_id", sort=False)[["v"]].max()
        out.columns = ["seller_orders_7d"]
        return self._align(f, src, out)

    @staticmethod
    def _align(f: pd.DataFrame, src: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        """Attach an order-indexed aggregate onto the population's row order."""
        joined = out.reindex(src["order_id"].to_numpy())
        for col in out.columns:
            f[col] = joined[col].to_numpy()
        return f

    def _availability_features(
        self, src: pd.DataFrame, built: pd.DataFrame
    ) -> pd.DataFrame:
        f = pd.DataFrame(index=src.index)
        f["has_item_rows"] = src["n_items"].notna().astype("int64")
        f["has_payment_row"] = src["payment_value"].notna().astype("int64")
        f["has_product_metadata"] = src["product_weight_g"].notna().astype("int64")
        f["has_zip_prefix"] = src["customer_zip_code_prefix"].notna().astype("int64")

        # Captures "this order came through a degraded path" (ARCHITECTURE.md 4.1).
        # Counted over every other feature column, so it runs last.
        others = pd.concat([built, f], axis=1) if len(built.columns) else f
        f["n_missing_features"] = others.isna().sum(axis=1).astype("int64")
        return f

    # -- entry point -------------------------------------------------------------------

    def build(self, population: pd.DataFrame, store=None) -> pd.DataFrame:
        """
        Return the feature matrix for ``population``, row-aligned to its index.

        Contains no label, no identifier, and no post-checkout column.

        Parameters
        ----------
        store:
            Optional :class:`~features.store.HistoryStore`.  When supplied, the
            point-in-time windows are read from the index rather than scanned out of
            ``population``, so ``population`` need only contain the rows being
            scored.  The feature formulas are unchanged - only the prior-window
            lookup differs.  Serving uses this; batch does not.
        """
        src = self._sources(population)
        parts: list[pd.DataFrame] = []

        needs_edges = bool({"seller", "density"} & set(self.groups))
        edges = self.edge_frame(src) if needs_edges else pd.DataFrame()

        if "order" in self.groups:
            parts.append(self._order_features(src))
        if "customer" in self.groups:
            parts.append(self._customer_features(src, store))
        if "pincode" in self.groups:
            parts.append(self._pincode_features(src, store))
        if "structure" in self.groups:
            parts.append(self._structure_features(src))
        if "seller" in self.groups:
            parts.append(self._seller_features(src, edges, store))
        if "route" in self.groups:
            parts.append(self._route_features(src, store))
        if "density" in self.groups:
            parts.append(self._density_features(src, edges, store))

        built = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=src.index)

        if "availability" in self.groups:
            built = pd.concat([built, self._availability_features(src, built)], axis=1)

        return built
