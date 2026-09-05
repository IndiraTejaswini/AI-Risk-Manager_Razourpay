"""
Olist dataset loader, maturation filter, label construction, and the checkout-time
column whitelist.

This module owns three things that the rest of the pipeline is not permitted to
re-derive: the matured population, the temporal split boundary, and the labels.

--------------------------------------------------------------------------------------
WHAT THE MODEL PREDICTS
--------------------------------------------------------------------------------------

The estimand is a CONDITIONAL probability:

        P(order fails to reach `delivered`  |  order was handed to a carrier)

written throughout this repo as **P(fails | ships)**.  It is NOT P(fails).

The risk set is therefore `order_delivered_carrier_date IS NOT NULL` - orders with
physical evidence that the parcel moved.  Orders that never shipped are removed from the
population entirely; they are not scored as negatives.

Why conditional:

  * The loss being modelled is return-to-origin.  RTO is a post-shipment event by
    definition.  An order that is cancelled while still in the warehouse, or whose stock
    turns out to be `unavailable`, cannot return to origin.  It did not survive to the
    point where the risk exists.
  * Scoring those orders as negatives puts rows in the denominator that were never at
    risk.  That understates prevalence and rewards a model for "correctly" assigning low
    risk to orders whose outcome was determined by something the model does not observe
    and cannot act on.
  * On this panel the restriction removes 1,775 of 99,433 matured orders (1.8%), of
    which 225 fall in the test window.

The cost of conditioning, stated plainly:

  * `order_delivered_carrier_date` is a POST-CHECKOUT variable.  Conditioning the
    training population on it means selecting on something not knowable at scoring time.
    The model is fit on the subpopulation that shipped, and at checkout we do not yet
    know whether a given order belongs to it.
  * The deployed score is consequently valid *conditional on shipment*.  Unconditional
    risk is P(ships) x P(fails | ships).  Any consumer that needs the unconditional
    quantity - a cost policy comparing against the value of the whole order, for
    instance - must multiply by a shipment probability this repo does not model.
  * Selection into the risk set is not random.  Orders are removed for reasons
    (seller-side cancellation, stock unavailability) that plausibly correlate with the
    covariates.  This is conditioning on a post-treatment variable and it can bias
    coefficient interpretation; it does not invalidate the predictive claim on the
    subpopulation, which is what the score is used for.

Every reported probability, calibration curve, and policy threshold in this repo is on
the conditional scale unless it is explicitly labelled otherwise.

--------------------------------------------------------------------------------------
LABELS
--------------------------------------------------------------------------------------

Two labels are constructed on every load.

  label_b  PRIMARY.    Entered fulfillment and never delivered.
                       Evaluated on the risk set, where it *is* P(fails | ships).
                       154 test positives.

  label_a  SECONDARY.  Any non-delivered order.  Evaluated on the full matured
                       population, where it is an unconditional P(fails).
                       379 test positives.

label_a is retained as a benchmark, not as a fallback.  It buys 2.5x the test positives
and a correspondingly tighter minimum detectable difference, at the cost of a target that
mixes stock and seller-side fulfilment failures in with delivery failures.  Reporting
both is what makes that trade-off legible instead of assumed; see
`eval/label_targets.md`.

--------------------------------------------------------------------------------------
LEAKAGE
--------------------------------------------------------------------------------------

`order_delivered_carrier_date` is a label-construction input and nothing else.  It is
listed in LABEL_ONLY_COLUMNS, excluded from every checkout frame, and the exclusion is
enforced behaviourally by tests/test_leakage.py, which perturbs the column and asserts
that label_b moves while the feature matrix does not.

See data/COLUMN_WHITELIST.md for the full admissibility rules.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pandas as pd

from data.whitelist import parse_column_whitelist

__all__ = [
    "OlistLoader",
    "ORDERS_CSV",
    "CUSTOMERS_CSV",
    "PAYMENTS_CSV",
    "ITEMS_CSV",
    "DELIVERED",
    "OBSERVED_TS_COLS",
    "LeakageError",
    "MATURATION_DAYS",
    "TEST_FRACTION",
    "VALIDATION_DAYS",
    "PRIMARY_LABEL",
    "SECONDARY_LABEL",
    "LABEL_ONLY_COLUMNS",
    "POST_CHECKOUT_COLUMNS",
    "FORBIDDEN_FEATURE_COLUMNS",
    "CHECKOUT_SAFE_COLUMNS",
    "AUDITED_CHECKOUT_KNOWABLE",
]

# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------

MATURATION_DAYS = 30       # ARCHITECTURE.md 3.5
TEST_FRACTION = 0.20       # ARCHITECTURE.md 2, 9

#: Length of the temporal validation window, in days before the test boundary.
#: Set to the longest calibration-window candidate in ARCHITECTURE.md 6 ({30,60,90}
#: days), so every candidate is nested inside validation and none can reach past the
#: boundary into test.  Validation carries both the early-stopping signal (5.2) and
#: the Platt fit (6); that they share rows is what 6 specifies, and it means the
#: calibration map is fit on data the model early-stopped against.  Stated, not hidden.
VALIDATION_DAYS = 90
DELIVERED = "delivered"

PRIMARY_LABEL = "label_b"    # P(fails | ships), evaluated on the risk set
SECONDARY_LABEL = "label_a"  # P(fails), evaluated on the full matured population

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

ORDERS_CSV = "olist_orders_dataset.csv"
CUSTOMERS_CSV = "olist_customers_dataset.csv"
PAYMENTS_CSV = "olist_order_payments_dataset.csv"
ITEMS_CSV = "olist_order_items_dataset.csv"

#: Whitelist table name -> CSV filename.
_TABLE_FILES = {
    "orders": ORDERS_CSV,
    "customers": CUSTOMERS_CSV,
    "payments": PAYMENTS_CSV,
    "items": ITEMS_CSV,
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

OBSERVED_TS_COLS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
)

# --------------------------------------------------------------------------------------
# Column admissibility.  See data/COLUMN_WHITELIST.md.
# --------------------------------------------------------------------------------------

#: Columns that construct labels.  Reading any of these into a feature matrix is the
#: outcome leaking into the predictors.  `order_delivered_carrier_date` is here because
#: it defines the risk set and label_b; it has no other legitimate consumer.
LABEL_ONLY_COLUMNS = frozenset({
    "order_status",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
})

#: Knowable only after checkout, so inadmissible as order features even though they are
#: not labels.  `order_estimated_delivery_date` is a post-purchase carrier estimate;
#: `shipping_limit_date` is set on seller assignment.
POST_CHECKOUT_COLUMNS = frozenset({
    "order_approved_at",
})

#: Audited and admitted (eval/feature_expansion.md section 1).  Both were previously
#: assumed post-checkout; both are set at order placement, and the audit is recorded
#: rather than the assumption being quietly reversed:
#:
#:   order_estimated_delivery_date  zero nulls across all eight order statuses,
#:                                  including `created` and `unavailable` which never
#:                                  shipped; a backfilled field would be null there.
#:   shipping_limit_date            same, and it is a contractual dispatch deadline,
#:                                  not a record of dispatch.
AUDITED_CHECKOUT_KNOWABLE = frozenset({
    "order_estimated_delivery_date",
    "shipping_limit_date",
})

#: Post-outcome by construction.  Admissible only as a point-in-time customer-history
#: feature computed from strictly prior orders, which is a step 2 concern.
REVIEW_COLUMNS = frozenset({
    "review_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
})

FORBIDDEN_FEATURE_COLUMNS = LABEL_ONLY_COLUMNS | POST_CHECKOUT_COLUMNS | REVIEW_COLUMNS

#: Checkout-time admissible columns, per source table.  Only the orders and customers
#: entries are materialised today; the rest are declared so the whitelist is the single
#: place the rule lives when step 2 joins them.
CHECKOUT_SAFE_COLUMNS: dict[str, tuple[str, ...]] = {
    "orders": ("order_id", "customer_id", "order_purchase_timestamp",
               "order_estimated_delivery_date"),
    "customers": (
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ),
    "payments": (
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ),
    "items": ("order_id", "order_item_id", "product_id", "seller_id", "price",
              "freight_value", "shipping_limit_date"),
    "products": (
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ),
    "sellers": ("seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"),
    # Static reference table.  A zip prefix's coordinates do not depend on any order and
    # cannot change once one is placed.  Its key is NOT unique - one row per geocoded
    # address - so it is reduced to a per-prefix centroid and never merged.
    "geolocation": (
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ),
}

#: Bookkeeping columns the loader attaches.  Not features; carried so callers can split
#: and group without re-deriving the boundary.
LOADER_COLUMNS = frozenset({
    "label_a", "label_b", "entered_fulfillment", "is_test", "is_mature",
})


#: Process-level memo of the kagglehub download path.  See OlistLoader.data_dir.
_DATASET_DIR: Path | None = None


def _cached_dataset_dir() -> Path | None:
    """
    The kagglehub cache directory, if it already holds the dataset.

    Used only as a fallback when the network call fails.  It looks for the CSV rather
    than trusting the directory to exist, so a half-finished download is not mistaken
    for a usable copy.
    """
    root = Path(
        os.environ.get("KAGGLEHUB_CACHE") or (Path.home() / ".cache" / "kagglehub")
    ) / "datasets" / Path(KAGGLE_DATASET) / "versions"
    if not root.is_dir():
        return None
    usable = [p for p in root.iterdir() if p.is_dir() and (p / ORDERS_CSV).is_file()]
    if not usable:
        return None
    # Highest version number, falling back to name order for non-numeric directories.
    usable.sort(key=lambda p: (p.name.isdigit(), int(p.name) if p.name.isdigit() else 0,
                              p.name))
    return usable[-1].resolve()


def _resolve_dataset_dir() -> Path:
    """
    Resolve the dataset directory once per process, and do not let a network blip end
    the run when the data is already on disk.

    ``kagglehub.dataset_download`` reaches out to check the dataset version even when
    every byte is cached.  A transient ``RemoteDisconnected`` from Kaggle therefore
    fails a step that needs no network at all - which it did, twice, in the middle of
    `make all` and again inside the test suite.  Neither failure was about the data.

    So: try the download, and on any failure fall back to a cache that actually
    contains the CSVs.  If there is no such cache the error names OLIST_DATA_DIR,
    because at that point the run genuinely cannot proceed and the caller needs to know
    what to set.
    """
    global _DATASET_DIR
    if _DATASET_DIR is not None:
        return _DATASET_DIR

    try:
        import kagglehub

        _DATASET_DIR = Path(kagglehub.dataset_download(KAGGLE_DATASET)).resolve()
    except Exception as exc:  # network, auth, SDK - the fallback is the same
        cached = _cached_dataset_dir()
        if cached is None:
            raise RuntimeError(
                f"could not reach Kaggle to resolve {KAGGLE_DATASET!r} ({exc}), and no "
                f"local copy containing {ORDERS_CSV} was found. Set OLIST_DATA_DIR to a "
                "directory holding the Olist CSVs to run without network access."
            ) from exc
        _DATASET_DIR = cached
    return _DATASET_DIR


class LeakageError(AssertionError):
    """Raised when a frame intended for modelling carries an inadmissible column."""


class OlistLoader:
    """
    Loads Olist, applies the maturation filter, builds both labels, and hands out
    checkout-time-admissible frames.

    The model this loader feeds predicts **P(fails | ships)**, not P(fails).  See the
    module docstring for what that means and what it costs.  In short: the primary
    target `label_b` is evaluated on `risk_set()`, the orders that reached a carrier.
    Orders that never shipped are excluded from the population rather than scored as
    negatives, because return-to-origin is a post-shipment event and an order that never
    moved was never at risk.

    Nothing in this class computes a feature.  It establishes the population, the
    labels, and the split, and it refuses to emit a frame containing a column that is
    not knowable at checkout.

    Parameters
    ----------
    data_dir:
        Directory holding the Olist CSVs.  Defaults to ``$OLIST_DATA_DIR`` if set,
        otherwise the dataset is fetched via ``kagglehub`` into its own cache.
    maturation_days:
        Orders must have had this long to resolve before the snapshot, else they are
        dropped as immature (ARCHITECTURE.md 3.5).
    test_fraction:
        Fraction of matured rows, most recent by purchase timestamp, held out as test.

    Examples
    --------
    >>> loader = OlistLoader()                        # doctest: +SKIP
    >>> risk = loader.risk_set()                      # doctest: +SKIP
    >>> y = risk[PRIMARY_LABEL]                       # P(fails | ships) target
    >>> X = loader.checkout_frame(risk)               # doctest: +SKIP
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str] | None = None,
        maturation_days: int = MATURATION_DAYS,
        test_fraction: float = TEST_FRACTION,
    ) -> None:
        self.maturation_days = maturation_days
        self.test_fraction = test_fraction
        self._data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
        self._raw: pd.DataFrame | None = None
        self._labelled: pd.DataFrame | None = None
        # Source tables are immutable for the life of a loader.  Serving calls the
        # feature builder per request, and re-reading four CSVs each time dominated
        # the latency budget; the cache makes the read a startup cost.
        self._tables: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}

    # -- data location -----------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """
        Where the CSVs live.  Resolved once per process, not once per loader.

        Without the process-level memo every ``OlistLoader()`` re-enters
        ``kagglehub.dataset_download``, which reaches the network to check the dataset
        version even when the files are already cached.  The test suite constructs
        dozens of loaders, so a single transient ``RemoteDisconnected`` from Kaggle
        fails an unrelated assertion - which is exactly what it did, in the middle of a
        feature-perturbation test that touches no network at all.

        The dataset is immutable for the life of a process, so the memo is not an
        optimisation traded against correctness; it removes a dependency the test never
        meant to take.  An explicit ``data_dir=`` argument bypasses it entirely.
        """
        if self._data_dir is None:
            env = os.environ.get("OLIST_DATA_DIR")
            if env:
                self._data_dir = Path(env).expanduser().resolve()
            else:
                self._data_dir = _resolve_dataset_dir()
        return self._data_dir

    def checksums(self, *names: str) -> dict[str, str]:
        """SHA-256 of each named CSV, for recording in an artifact."""
        out = {}
        for name in names or (ORDERS_CSV, CUSTOMERS_CSV):
            h = hashlib.sha256()
            with (self.data_dir / name).open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[name] = h.hexdigest()
        return out

    # -- population --------------------------------------------------------------------

    def raw_orders(self) -> pd.DataFrame:
        """The orders table, unfiltered, with timestamp columns parsed."""
        if self._raw is None:
            self._raw = pd.read_csv(
                self.data_dir / ORDERS_CSV, parse_dates=list(OBSERVED_TS_COLS)
            )
        return self._raw.copy()

    def snapshot(self, orders: pd.DataFrame | None = None) -> pd.Timestamp:
        """
        Extract date, derived rather than assumed: the latest observed event in the
        orders table.  ``order_estimated_delivery_date`` is excluded because it is a
        forward-looking estimate and would push the snapshot past the end of the data.
        """
        orders = self.raw_orders() if orders is None else orders
        return max(orders[c].max() for c in OBSERVED_TS_COLS)

    def labelled(self) -> pd.DataFrame:
        """
        The matured population with both labels, the fulfillment flag, and the split.

        Columns added
        -------------
        entered_fulfillment : bool
            ``order_delivered_carrier_date`` is present.  Physical evidence the parcel
            reached a carrier.  Defines the risk set.
        label_a : bool
            SECONDARY.  Any non-delivered order.  Unconditional P(fails).
        label_b : bool
            PRIMARY.  Entered fulfillment and never delivered.  On ``risk_set()`` this
            is P(fails | ships).
        is_test : bool
            In the most recent ``test_fraction`` of matured rows by purchase timestamp.
        """
        if self._labelled is not None:
            return self._labelled.copy()

        orders = self.raw_orders()
        snapshot = self.snapshot(orders)

        matured_by = orders["order_purchase_timestamp"] + pd.Timedelta(
            days=self.maturation_days
        )
        df = orders[matured_by <= snapshot].copy()

        # The fulfillment flag keys on the carrier date rather than on order_status,
        # because 75 `canceled` orders on this panel carry a carrier date - they shipped
        # and were cancelled in transit.  A status-only flag would put them outside the
        # risk set, which is the wrong side of the line for a delivery-failure target.
        df["entered_fulfillment"] = df["order_delivered_carrier_date"].notna()

        not_delivered = df["order_status"] != DELIVERED
        df["label_a"] = not_delivered
        df["label_b"] = not_delivered & df["entered_fulfillment"]

        # Boundary taken at a row position, so it is an observed timestamp rather than an
        # interpolated instant.  Computed once on the full matured population and applied
        # unchanged to every subset, so all populations share a test window.
        ordered = df["order_purchase_timestamp"].sort_values(kind="mergesort")
        self._cut_idx = int(math.floor(len(ordered) * (1.0 - self.test_fraction)))
        self._cut = ordered.iloc[self._cut_idx]
        df["is_test"] = df["order_purchase_timestamp"] >= self._cut

        self._labelled = df
        return df.copy()

    @property
    def split_boundary(self) -> pd.Timestamp:
        """Purchase timestamp at which the test window opens."""
        self.labelled()
        return self._cut

    @property
    def split_index(self) -> int:
        """
        Row position the boundary was taken at, over matured rows sorted ascending.

        Exposed so a report can state where the cut fell without re-deriving it.
        """
        self.labelled()
        return self._cut_idx

    def risk_set(self) -> pd.DataFrame:
        """
        The population the PRIMARY target is defined on: matured orders that reached a
        carrier.

        This is the conditioning that turns ``label_b`` into P(fails | ships).  Orders
        that never shipped are absent, not zero-labelled - they were never at risk of a
        return, and scoring them as negatives would put rows in the denominator that no
        model could have acted on.
        """
        df = self.labelled()
        return df[df["entered_fulfillment"]].copy()

    def split_labelled(self) -> pd.DataFrame:
        """
        The matured population with a three-way chronological ``split`` column.

        ``train`` | ``validation`` | ``test`` - exhaustive, disjoint, and strictly
        ordered in time.  Validation is the last ``VALIDATION_DAYS`` before the test
        boundary; train is everything earlier.  Validation never intersects test by
        construction, which matters because validation is what selects the calibration
        window length and the early-stopping round.
        """
        df = self.labelled()
        val_start = self.split_boundary - pd.Timedelta(days=VALIDATION_DAYS)
        ts = df["order_purchase_timestamp"]

        split = pd.Series("train", index=df.index, dtype=object)
        split[(ts >= val_start) & (ts < self.split_boundary)] = "validation"
        split[ts >= self.split_boundary] = "test"
        df["split"] = split
        return df

    @property
    def validation_start(self) -> pd.Timestamp:
        return self.split_boundary - pd.Timedelta(days=VALIDATION_DAYS)

    # -- source tables -----------------------------------------------------------------

    def load_table(self, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Read a source CSV, admitting only columns on the whitelist.

        The whitelist is parsed from data/COLUMN_WHITELIST.md, so the document is the
        enforced rule rather than a description of one.  Requesting a column absent from
        it - a label field, a post-checkout field, or a typo - raises rather than
        silently returning it.

        Parameters
        ----------
        table:
            Whitelist table name (``orders``, ``customers``, ``payments``, ``items``,
            ``products``, ``sellers``).
        columns:
            Subset to read.  Defaults to every whitelisted column for the table.
        """
        allowed = parse_column_whitelist()
        if table not in allowed:
            raise KeyError(
                f"{table!r} is not a whitelisted source table; "
                f"known tables: {sorted(allowed)}"
            )

        wanted = list(columns) if columns is not None else sorted(allowed[table])
        offenders = sorted(set(wanted) - allowed[table])
        if offenders:
            raise LeakageError(
                f"column(s) not on the whitelist for table {table!r}: "
                + ", ".join(offenders)
                + ". See data/COLUMN_WHITELIST.md."
            )

        key = (table, tuple(wanted))
        if key not in self._tables:
            frame = pd.read_csv(self.data_dir / _TABLE_FILES[table], usecols=wanted)
            self.assert_no_leakage(frame)
            # Uniqueness of the join key, checked once here rather than on every
            # merge.  Feature construction joins customers per request, and pandas'
            # validate="m:1" re-factorises the right frame each time.
            if table == "customers" and "customer_id" in frame.columns:
                if frame["customer_id"].duplicated().any():
                    raise LeakageError(
                        "customer_id is not unique in the customers table; a merge on "
                        "it would fan out rows and silently change every aggregate"
                    )
            self._tables[key] = frame
        # Returned without copying: the cache holds ~350k rows across four tables and
        # copying them per request cost more than the joins did.  Callers treat the
        # result as read-only - every consumer filters or aggregates into a new
        # frame, and the one path that substitutes rows replaces the cache entry
        # wholesale rather than mutating it in place.
        return self._tables[key]

    # -- feature admissibility ---------------------------------------------------------

    @staticmethod
    def assert_no_leakage(frame: pd.DataFrame) -> None:
        """
        Raise ``LeakageError`` if ``frame`` carries a column that is not knowable at
        checkout.

        This is a name-level check.  It catches the column being carried through; it
        cannot detect a feature silently *derived* from a forbidden column upstream.
        tests/test_leakage.py covers that case behaviourally instead, by perturbing
        ``order_delivered_carrier_date`` and asserting the feature matrix is unchanged.
        """
        offenders = sorted(set(frame.columns) & FORBIDDEN_FEATURE_COLUMNS)
        if offenders:
            raise LeakageError(
                "inadmissible column(s) in a frame intended for modelling: "
                + ", ".join(offenders)
                + ". These are label-construction or post-checkout fields; see "
                "data/COLUMN_WHITELIST.md."
            )

    def checkout_frame(
        self,
        population: pd.DataFrame | None = None,
        with_customer: bool = True,
    ) -> pd.DataFrame:
        """
        Checkout-time-admissible columns only, for the given population.

        Contains no features yet - feature construction is step 2.  What it guarantees
        today is the boundary: every column here is knowable when the order is placed,
        and ``assert_no_leakage`` is run on the way out.

        Parameters
        ----------
        population:
            Frame to project.  Defaults to ``risk_set()``, the primary target's
            population.
        with_customer:
            Join the customers table for ``customer_unique_id`` and the zip prefix.
        """
        df = self.risk_set() if population is None else population

        cols = [c for c in CHECKOUT_SAFE_COLUMNS["orders"] if c in df.columns]
        out = df[cols].copy()

        if with_customer:
            customers = pd.read_csv(
                self.data_dir / CUSTOMERS_CSV,
                usecols=list(CHECKOUT_SAFE_COLUMNS["customers"]),
            )
            out = out.merge(customers, on="customer_id", how="left", validate="m:1")

        self.assert_no_leakage(out)
        return out
