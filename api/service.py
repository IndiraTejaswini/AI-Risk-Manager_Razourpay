"""
Scoring service.  ARCHITECTURE.md 11.

--------------------------------------------------------------------------------------
THE SAME CODE PATH AS TRAINING - NO REIMPLEMENTATION
--------------------------------------------------------------------------------------

Every feature the endpoint scores is produced by :class:`features.builder.FeatureBuilder`
- the class that built the training matrix.  Nothing here recomputes a feature, and there
is no second definition of ``freight_ratio`` or ``pincode_failure_rate_smoothed`` to
drift from the first.

That is not merely tidy.  Train/serve skew is the failure mode where a model works
offline and quietly does something else in production, and a hand-written serving copy of
the feature code is how it usually happens.  ``tests/test_api.py`` asserts that a fixed
set of orders scored through the service reproduces the batch predictions **exactly**.

The cost is latency, and it is real.  The point-in-time features are defined against the
order's history, so the builder is handed the historical population plus the request row:
prior-order counts and the smoothed pincode encoding cannot be computed from the payload
alone.  On this panel that is a ~97k-row frame per request.  ``eval/latency.md`` reports
what that costs against the 200ms budget and names the production fix (a materialised
feature store serving the aggregates incrementally) rather than pre-empting it with a
second implementation here.

--------------------------------------------------------------------------------------
WHAT IS CACHED AT STARTUP
--------------------------------------------------------------------------------------

Model, calibrator, SHAP explainer, source tables, and the historical population.  Section
8 asks for the explainer to be cached at startup; in practice everything except the
request row is.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from api.contract import FORBIDDEN_PAYLOAD_FIELDS, ScoreRequest, ScoreResponse
from data.loader import PRIMARY_LABEL, OlistLoader
from features.builder import FeatureBuilder
from models.calibration import PlattCalibrator
from models.explain import ReasonExplainer
from models.train import ModelBundle, predict, prepare_matrix, train
from policy.costs import ACTION_TIERS
from policy.elkan import apply_policy
from policy.explain_cost import explain_cost
from models.counterfactual import counterfactual_score, top_modifiable_feature

__all__ = ["ScoringService", "LeakyPayloadError", "Timings"]

CALIBRATION_WINDOW_DAYS = 30
TS = "order_purchase_timestamp"


class LeakyPayloadError(ValueError):
    """The request carried a field that is not knowable at checkout."""


@dataclass
class Timings:
    """Per-stage wall time in milliseconds for one request."""

    stages: dict[str, float] = field(default_factory=dict)

    def __setitem__(self, k: str, v: float) -> None:
        self.stages[k] = v

    @property
    def total(self) -> float:
        return float(sum(self.stages.values()))


class ScoringService:
    """
    Loads everything once, scores a Razorpay-shaped payload.

    Parameters
    ----------
    loader:
        Shared loader; its source-table cache is what keeps per-request feature
        construction off the disk.
    """

    def __init__(self, loader: OlistLoader | None = None) -> None:
        self.loader = loader or OlistLoader()

        risk = self.loader.risk_set().join(
            self.loader.split_labelled()["split"], how="left"
        )
        self.builder = FeatureBuilder(loader=self.loader)
        matrix = self.builder.build(risk)

        split = risk["split"].to_numpy()
        y = risk[PRIMARY_LABEL].astype(int).to_numpy()
        tr, va = split == "train", split == "validation"

        _, self.category_levels = prepare_matrix(matrix.loc[tr])
        X, _ = prepare_matrix(matrix, category_levels=self.category_levels)

        self.bundle: ModelBundle = train(
            X.loc[tr], pd.Series(y[tr]), X.loc[va], pd.Series(y[va]),
            target=PRIMARY_LABEL, population="risk_set",
            category_levels=self.category_levels,
        )

        s_va = predict(self.bundle, X.loc[va], raw_score=True)
        ts_va = risk.loc[va, TS]
        fit = (ts_va >= self.loader.split_boundary
               - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)).to_numpy()
        self.calibrator = PlattCalibrator().fit(s_va[fit], y[va][fit])

        # Section 8: explainer cached at startup.
        self.explainer = ReasonExplainer(self.bundle)

        # The historical population the point-in-time features are computed against.
        self.history = risk.drop(columns=["split"])

        # Prior-window index over that history.  Without it every request re-derived the
        # point-in-time aggregates from ~97k rows, which was 99% of a 3.2s response
        # (eval/latency.md, step 7).  The index answers the same question by binary
        # search; features/store.py explains why that is one definition and not two, and
        # `_assert_store_matches_rowscan` below checks it on real orders at startup.
        self.store = self.builder.history_store(self.builder._sources(self.history))
        self._assert_store_matches_rowscan(matrix, risk)
        self.customers = self.loader.load_table(
            "customers",
            ["customer_id", "customer_unique_id", "customer_zip_code_prefix"],
        )
        self._zip_to_customer = (
            self.customers.drop_duplicates("customer_zip_code_prefix")
            .set_index("customer_zip_code_prefix")["customer_id"]
        )
        self._ref_to_customer = (
            self.customers.drop_duplicates("customer_unique_id")
            .set_index("customer_unique_id")["customer_id"]
        )

        self.model_version = self._version()

    def _assert_store_matches_rowscan(
        self, batch_matrix: pd.DataFrame, population: pd.DataFrame
    ) -> None:
        """
        The store must reproduce the row-scan exactly, on **every** order, before the
        service accepts traffic.

        Same class of check as the API-vs-batch parity test, one level down: it is what
        stops the optimisation silently changing a feature value.  A drift here would
        not raise anywhere else - the model would happily score the wrong number - so it
        is checked at startup rather than left to the test suite.

        ----------------------------------------------------------------------------
        WHY EVERY ROW, AND NOT A STRIDE
        ----------------------------------------------------------------------------

        This check used to walk a 40-order deterministic stride.  It passed, for months,
        against two genuinely defective implementations:

          * an exclusive prior sum computed as ``cumsum - own_value``, which disagreed
            with the serving index on 1,169 of 97,658 rows (1.2%);
          * a tie-block collapse that skipped nulls, which made
            ``cust_days_since_prior_order`` read a same-instant order as the previous
            one on 534 rows (0.55%).

        Neither the 40-order stride here nor the 75-order stride in tests/test_store.py
        contained a single affected row.  That is not luck that ran out - it is what a
        fixed stride does to a defect whose rows are determined by the data: the two
        patterns are independent, so a check with 0.04% coverage finds a 1% defect
        essentially never, and reports PASS with total confidence when it does not.

        **A sample that can systematically miss a systematic defect is not a check.**
        Both defects were found the first time this ran over the whole population.

        The cost is stated rather than assumed: see ``store_check_seconds`` and
        eval/latency.md.  It is paid once, at startup, by a service that already trains
        a model before accepting traffic.
        """
        started = time.perf_counter()
        via_store = self.builder.build(population, store=self.store)

        offenders: list[str] = []
        for col in batch_matrix.columns:
            a = via_store[col].to_numpy()
            b = batch_matrix[col].to_numpy()
            if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
                bad = ~(
                    (a.astype(float) == b.astype(float))
                    | (pd.isna(a) & pd.isna(b))
                )
            else:
                bad = np.array([
                    not ((x == y) or (pd.isna(x) and pd.isna(y)))
                    for x, y in zip(a, b)
                ])
            n_bad = int(bad.sum())
            if n_bad:
                first = int(np.flatnonzero(bad)[0])
                offenders.append(
                    f"{col} ({n_bad:,} of {len(a):,} rows; first at row {first}: "
                    f"store={a[first]!r} batch={b[first]!r})"
                )

        self.store_check_seconds = time.perf_counter() - started
        self.store_check_rows = len(population)

        if offenders:
            raise AssertionError(
                "the history store disagrees with the row-scan on "
                + "; ".join(offenders)
                + f" over all {len(population):,} orders. The serving path would score "
                "different features from the ones the model was trained and evaluated "
                "on."
            )

    # -- version ------------------------------------------------------------------------

    def _version(self) -> str:
        """
        Model version string, on every response, non-negotiable (11.1).

        Derived from the artifacts rather than hand-maintained: a retrain that changes
        the booster or the calibrator changes the string automatically, so a response can
        always be traced to the exact model that produced it.
        """
        h = hashlib.sha256()
        # Fingerprint the model by what it DOES, not by how it serialises.
        #
        # Hashing booster.model_to_string() looked right and was not: the Windows host
        # and the Linux container produced different digests for a model that scores
        # every order identically (verified to 0.0 difference). The serialised text
        # carries platform-dependent formatting, so the version changed while the model
        # did not - which is precisely backwards for a traceability key.
        #
        # A fixed probe through the full scoring path is stable across platforms and
        # changes exactly when the scores change, which is what the string is for.
        h.update(np.asarray(self._probe_scores(), dtype="<f8").tobytes())
        h.update(f"{self.calibrator.a_:.12g}:{self.calibrator.b_:.12g}".encode())
        h.update(",".join(self.bundle.feature_names).encode())
        return f"rto-{self.bundle.target}-{h.hexdigest()[:12]}"

    #: Rows in the model-version fingerprint.  Not a coverage figure - the string has
    #: to be cheap and stable across platforms - but the same sampling argument applies:
    #: a model change that moved only a fraction of a percent of rows could leave a
    #: small probe unchanged and hand back a stale version string.  2,048 rows costs
    #: milliseconds and is ~8x less likely to miss one than the 256 it replaced.
    PROBE_ROWS = 2_048

    def _probe_scores(self) -> np.ndarray:
        """Raw margins on a fixed, deterministic slice - the model's fingerprint."""
        step = max(len(self.history) // self.PROBE_ROWS, 1)
        probe = self.history.iloc[::step].head(self.PROBE_ROWS)
        feats = self.builder.build(probe, store=self.store)
        X, _ = prepare_matrix(feats, category_levels=self.category_levels)
        return predict(self.bundle, X[list(self.bundle.feature_names)], raw_score=True)

    # -- payload -> population row ------------------------------------------------------

    @staticmethod
    def assert_payload_clean(raw: dict) -> None:
        """
        The leakage gate at the request boundary.

        A caller replaying a historical order can include outcome fields without meaning
        to. The request is **rejected**, not silently stripped: a client sending
        post-outcome data should find out immediately rather than receive a plausible
        score computed from data the model must never see.
        """
        found: list[str] = []

        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in FORBIDDEN_PAYLOAD_FIELDS:
                        found.append(f"{path}{k}")
                    walk(v, f"{path}{k}.")
            elif isinstance(node, list):
                for item in node:
                    walk(item, path)

        walk(raw)
        if found:
            raise LeakyPayloadError(
                "payload carries field(s) that are not knowable at checkout: "
                + ", ".join(sorted(set(found)))
                + ". These are label-construction or post-outcome fields; see "
                "data/COLUMN_WHITELIST.md. The request is rejected rather than having "
                "the fields ignored."
            )

    def _resolve_customer_id(self, req: ScoreRequest) -> str:
        """
        Map the payload's customer identity onto the history's key.

        A merchant customer reference that matches known history uses it; otherwise the
        order is a cold start and is bound to the pincode only. 97% of this panel is the
        latter, so it is the ordinary path rather than the exception.
        """
        cust = req.payload.customer
        ref = cust["entity"].customer_reference if cust else None
        if ref and ref in self._ref_to_customer.index:
            return str(self._ref_to_customer.loc[ref])

        zipcode = req.payload.shipping_address["entity"].zipcode
        try:
            prefix = int(str(zipcode)[:5])
        except ValueError:
            prefix = -1
        if prefix in self._zip_to_customer.index:
            return str(self._zip_to_customer.loc[prefix])
        return f"__unknown__{zipcode}"

    def _order_row(self, req: ScoreRequest) -> tuple[pd.DataFrame, pd.DataFrame,
                                                     pd.DataFrame]:
        """Build the population row plus the item and payment rows the builder joins."""
        order = req.payload.order["entity"]
        oid = order.id
        ts = pd.to_datetime(order.created_at, unit="s")

        row = pd.DataFrame({
            "order_id": [oid],
            "customer_id": [self._resolve_customer_id(req)],
            TS: [ts],
            "label_a": [False],          # unknown at checkout; never read for this row
            "label_b": [False],
            "entered_fulfillment": [True],
            "is_test": [True],
        })

        items = pd.DataFrame([
            {
                "order_id": oid,
                "order_item_id": i + 1,
                "product_id": li.product_id or "__unknown__",
                "seller_id": li.seller_id or "__unknown__",
                "price": li.amount / 100.0 * li.quantity,
                "freight_value": li.shipping_amount / 100.0,
            }
            for i, li in enumerate(req.payload.line_items)
        ])

        pay = req.payload.payment["entity"] if req.payload.payment else None
        method = pay.method if pay else "card"
        payments = pd.DataFrame([{
            "order_id": oid,
            "payment_sequential": 1,
            "payment_type": "boleto" if method in ("boleto", "cod") else method,
            "payment_installments": (pay.emi_installments if pay else None) or 1,
            "payment_value": (pay.amount if pay else order.amount) / 100.0,
        }])
        return row, items, payments

    # -- scoring ------------------------------------------------------------------------

    def score(self, req: ScoreRequest, raw: dict | None = None) -> tuple[ScoreResponse,
                                                                        Timings]:
        t = Timings()
        if raw is not None:
            t0 = time.perf_counter()
            self.assert_payload_clean(raw)
            t["payload_validation"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        row, items, payments = self._order_row(req)
        # Only the request row is built.  The point-in-time windows come from the
        # startup index, which is queried by timestamp and so cannot see the order's own
        # row or anything at or after it - the same strict inequality the row-scan
        # applies.  The request's line items and payment are injected into the cached
        # source tables for the duration of the call and restored afterwards.
        with self._injected(items, payments, row["order_id"]):
            matrix = self.builder.build(row, store=self.store)
        feats = matrix.iloc[[-1]]
        t["feature_construction"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        X, _ = prepare_matrix(feats, category_levels=self.category_levels)
        X = X[list(self.bundle.feature_names)]
        margin = predict(self.bundle, X, raw_score=True)
        t["model_inference"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        p = self.calibrator.predict(margin)
        t["calibration"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        expl = self.explainer.explain(X)
        reasons = self.explainer.top_reasons(expl, k=3)[0]
        buckets = self.explainer.reason_buckets(expl)
        t["shap_and_reasons"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        res = apply_policy(p, feats, buckets)
        tier = str(res.tier[0])
        if tier == "allow":
            finite = [res.thresholds[k][0] for k in ACTION_TIERS
                      if np.isfinite(res.thresholds[k][0])]
            threshold = float(min(finite)) if finite else float("inf")
        else:
            threshold = float(res.thresholds[tier][0])
        t["policy"] = (time.perf_counter() - t0) * 1000

        response = ScoreResponse(
            risk=float(p[0]),
            tier=tier,
            reasons=reasons,
            model_version=self.model_version,
            threshold_used=threshold,
            features_missing=int(feats["n_missing_features"].iloc[0]),
            cost_breakdown=explain_cost(res, feats),
        )
        return response, t

    def explain_order(self, order_id: str) -> dict:
        rows = self.history[self.history["order_id"] == order_id]
        if len(rows) != 1:
            raise KeyError(order_id)
        features = self.builder.build(rows, store=self.store)
        X, _ = prepare_matrix(features, category_levels=self.category_levels)
        margin = predict(self.bundle, X, raw_score=True)
        probability = float(self.calibrator.predict(margin)[0])
        explanation = self.explainer.explain(X)
        reasons = self.explainer.reason_buckets(explanation)
        original_policy = apply_policy(
            np.array([probability]), features, reasons
        )
        feature, improved_value = top_modifiable_feature(explanation, X)
        improved = counterfactual_score(
            self.bundle, self.calibrator, X, feature, improved_value
        )
        changed_policy = apply_policy(
            np.array([improved]), features, reasons
        )
        return {
            "order_id": order_id,
            "feature": feature,
            "current_value": float(X.iloc[0][feature]),
            "improved_value": improved_value,
            "risk": probability,
            "counterfactual_risk": improved,
            "delta": improved - probability,
            "tier": str(original_policy.tier[0]),
            "counterfactual_tier": str(changed_policy.tier[0]),
        }

    # -- source injection ---------------------------------------------------------------

    class _Injection:
        def __init__(self, service, items, payments, ids):
            self.s, self.items, self.payments = service, items, payments
            self.ids = pd.Index(ids)
            self.saved: dict = {}

        def __enter__(self):
            cache = self.s.loader._tables
            for key in list(cache):
                table = key[0]
                extra = {"items": self.items, "payments": self.payments}.get(table)
                if extra is None:
                    continue
                self.saved[key] = cache[key]
                cols = list(cache[key].columns)
                # The cache entry is replaced with ONLY this order's rows, not with the
                # full table plus them.  `_sources` filters these tables to the
                # population's order_ids and the population is one order, so every other
                # row would be discarded a moment later - and filtering 112k item rows
                # per request was costing more than the feature maths.
                #
                # Replacing rather than appending also keeps a re-score idempotent: an
                # order already in history must not have its line items counted twice,
                # which is exactly what a webhook redelivery would do.
                cache[key] = (
                    extra.reindex(columns=cols)
                    if not extra.empty
                    else cache[key].iloc[:0]
                )
            return self

        def __exit__(self, *exc):
            self.s.loader._tables.update(self.saved)
            return False

    def _injected(self, items, payments, order_ids):
        return self._Injection(self, items, payments, order_ids)
