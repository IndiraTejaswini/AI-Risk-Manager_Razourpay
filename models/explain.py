"""
TreeSHAP and the risk-reason layer.  ARCHITECTURE.md 8, build step 6.

--------------------------------------------------------------------------------------
EXACT, NOT APPROXIMATE
--------------------------------------------------------------------------------------

TreeSHAP computes Shapley values for a tree ensemble **exactly**.  It is not a sampling
estimator and it is not a local surrogate: there is no approximation error to report and
no convergence to check.  Section 8 says coverage is 100% because there is one tree model
and TreeSHAP is exact for it, and that is the correct framing - the earlier
"KernelSHAP for the other branch" claim died with the router.

The additivity identity below is the check that the explainer is *wired* correctly, not a
check on the algorithm's accuracy:

    sum_j phi_j(x)  +  base_value  ==  model margin(x)

to floating point.  A failure means the explainer is pointed at the wrong output space or
the wrong feature order - the same class of bug as fitting Platt on probabilities instead
of log-odds, and it would silently corrupt every reason string.

--------------------------------------------------------------------------------------
RAW SCORES, NOT CALIBRATED PROBABILITIES
--------------------------------------------------------------------------------------

SHAP is computed on the **raw log-odds margin**, which is the space the trees are
additive in.  The policy layer decides on the **calibrated probability**.

The two are related by a strictly monotone map (Platt), so the *ranking* of orders is
identical in both.  But the reason ranking is an attribution in margin space and the tier
is a decision in probability space, and they can disagree at the margin: two orders with
nearly equal calibrated probability can have quite different attribution profiles, and an
order can sit just below a tier threshold while carrying a large positive attribution.
Neither is wrong; they answer different questions.  Stated because a reviewer will ask.

--------------------------------------------------------------------------------------
PERTURBATION MODE
--------------------------------------------------------------------------------------

Section 8 specifies **interventional** perturbation.  It is not available for this model:
``shap`` refuses interventional TreeSHAP on any model containing native categorical
splits, and ``product_category`` is one - the highest-gain feature in the matrix.

This module therefore uses ``tree_path_dependent``, which is also exact for the tree and
differs only in the reference distribution ("absence" of a feature means following the
tree's own path weights rather than integrating over a supplied background).  The
consequence is that attribution can be spread across correlated features somewhat
differently than an interventional reference would spread it.

The route to interventional is to replace the native categorical with the smoothed
point-in-time target encoding that section 4.5 offers as an equal alternative
("apply the same smoothed encoding to product_category and seller_id, **or** use LightGBM
native categoricals").  That makes the matrix fully numeric and unblocks interventional
TreeSHAP - at the cost of retraining, which moves every number from step 3 onward.  That
is a live decision, not a defect, and it is recorded here rather than resolved silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from models.constraints import MONOTONE_CONSTRAINTS

__all__ = [
    "FEATURE_GROUP",
    "REASON_TEMPLATES",
    "ReasonExplainer",
    "AdditivityError",
]


class AdditivityError(AssertionError):
    """SHAP values do not reconstruct the model output."""


#: Feature -> the group it belongs to, matching the policy layer's reason buckets.
FEATURE_GROUP: dict[str, str] = {
    # order composition
    "order_value": "order_composition",
    "order_freight": "order_composition",
    "freight_ratio": "order_composition",
    "log_order_value": "order_composition",
    "n_items": "order_composition",
    "n_sellers": "order_composition",
    "n_products": "order_composition",
    "avg_item_price": "order_composition",
    "payment_value": "order_composition",
    "payment_installments": "order_composition",
    "n_payment_types": "order_composition",
    "is_boleto": "order_composition",
    "product_weight_g": "order_composition",
    "product_volume_cm3": "order_composition",
    "product_photos_qty": "order_composition",
    "product_category": "order_composition",
    "purchase_hour": "order_composition",
    "purchase_dow": "order_composition",
    "purchase_month": "order_composition",
    "purchase_is_weekend": "order_composition",
    # pincode
    "pincode_failure_rate_smoothed": "pincode",
    "pincode_prior_orders": "pincode",
    "pincode_prior_failures": "pincode",
    "global_prior_failure_rate": "pincode",
    # customer history
    "cust_prior_orders": "customer_history",
    "cust_prior_failures": "customer_history",
    "cust_prior_failure_rate": "customer_history",
    "cust_prior_avg_value": "customer_history",
    "cust_days_since_prior_order": "customer_history",
    "cust_prior_boleto_ratio": "customer_history",
    # availability
    "has_item_rows": "availability",
    "has_payment_row": "availability",
    "has_product_metadata": "availability",
    "n_missing_features": "availability",
    # parcel structure and the checkout promises
    "volumetric_weight_g": "structure",
    "dim_weight_ratio": "structure",
    "freight_per_item": "structure",
    "n_categories": "structure",
    "promised_days": "structure",
    "dispatch_window_days": "structure",
    # seller history
    "seller_failure_rate_smoothed": "seller_history",
    "seller_prior_orders": "seller_history",
    "seller_tenure_days": "seller_history",
    "seller_prior_dispatch_window_mean": "seller_history",
    # route
    "customer_state": "route",
    "seller_state": "route",
    "same_state": "route",
    "route_pair_prior_orders": "route",
    "route_pair_failure_rate_smoothed": "route",
    "route_distance_km": "route",
    # density
    "pincode_orders_24h": "density",
    "pincode_orders_7d": "density",
    "seller_orders_7d": "density",
}

#: Feature -> merchant-facing sentence for a **risk-increasing** attribution.
#:
#: These are what a merchant reads.  They name the thing that can be acted on, not the
#: feature that carries it: "no delivery history for this pincode" is actionable,
#: "pincode_prior_orders = 3" is not.
REASON_TEMPLATES: dict[str, str] = {
    "pincode_failure_rate_smoothed":
        "This delivery pincode has an elevated historical failure rate.",
    "pincode_prior_orders":
        "We have little delivery history for this pincode.",
    "pincode_prior_failures":
        "Previous orders to this pincode have failed to be delivered.",
    "global_prior_failure_rate":
        "Delivery failures were elevated across the platform around this time.",

    "cust_prior_failures":
        "This customer has previous orders that were not delivered.",
    "cust_prior_failure_rate":
        "This customer's delivery history includes failures.",
    "cust_prior_orders":
        "First order from this customer - no delivery history to draw on.",
    "cust_prior_boleto_ratio":
        "This customer's previous orders used deferred payment.",
    "cust_prior_avg_value":
        "This order is priced unusually against the customer's history.",
    "cust_days_since_prior_order":
        "It has been an unusually long time since this customer's last order.",

    "freight_ratio":
        "Shipping cost is high relative to the order value.",
    "order_freight":
        "Shipping cost on this order is high.",
    "order_value":
        "Order value is unusual for this seller mix.",
    "log_order_value":
        "Order value is unusual for this seller mix.",
    "avg_item_price":
        "Item prices on this order are unusual.",
    "payment_value":
        "The amount paid is unusual for this order.",
    "payment_installments":
        "Paid across an unusually high number of instalments.",
    "n_payment_types":
        "Several payment instruments were combined on this order.",
    "is_boleto":
        "Paid by boleto - a deferred instrument with weaker commitment at checkout.",
    "n_items":
        "This order contains an unusual number of items.",
    "n_sellers":
        "This order spans several sellers, so it ships in multiple parcels.",
    "n_products":
        "This order contains several distinct products.",
    "product_weight_g":
        "This is a heavy parcel, which is handled more and fails more often.",
    "product_volume_cm3":
        "This is a bulky parcel, which is harder to deliver.",
    "product_photos_qty":
        "The product listing has unusually few images.",
    "product_category":
        "This product category has an elevated failure rate.",
    "purchase_hour":
        "Ordered at an hour associated with a higher failure rate.",
    "purchase_dow":
        "Ordered on a day associated with a higher failure rate.",
    "purchase_month":
        "Ordered in a period with an elevated failure rate.",
    "purchase_is_weekend":
        "Weekend order - dispatch is slower and failure rates are higher.",

    "n_missing_features":
        "Some order details were incomplete at checkout.",
    "has_item_rows":
        "Order line detail was missing at checkout.",
    "has_payment_row":
        "Payment detail was missing at checkout.",
    "has_product_metadata":
        "Product detail was missing at checkout.",

    "seller_failure_rate_smoothed":
        "A seller on this order has an elevated historical failure rate.",
    "seller_prior_orders":
        "A seller on this order has little delivery history.",
    "seller_tenure_days":
        "A seller on this order is new to the platform.",
    "seller_prior_dispatch_window_mean":
        "This seller's usual dispatch deadline is unusually long.",

    "route_pair_failure_rate_smoothed":
        "This origin-destination route has an elevated historical failure rate.",
    "route_pair_prior_orders":
        "We have little delivery history for this origin-destination route.",
    "route_distance_km":
        "This parcel is travelling an unusually long distance.",
    "same_state":
        "The parcel crosses a state boundary, which lengthens the route.",
    "customer_state":
        "Deliveries to this state fail more often than the platform average.",
    "seller_state":
        "Dispatches from this state fail more often than the platform average.",

    "pincode_orders_24h":
        "Unusual recent order volume to this pincode.",
    "pincode_orders_7d":
        "Unusual recent order volume to this pincode.",
    "seller_orders_7d":
        "A seller on this order is handling unusual recent volume.",

    "volumetric_weight_g":
        "This is a bulky parcel, which is harder to deliver.",
    "dim_weight_ratio":
        "This parcel is bulky for its weight - the shape that gets mishandled.",
    "freight_per_item":
        "Shipping cost per item on this order is high.",
    "n_categories":
        "This order mixes several product categories, so it ships in more parcels.",
    "promised_days":
        "The promised delivery window on this order is unusually long.",
    "dispatch_window_days":
        "The seller's contractual dispatch deadline on this order is unusually long.",
}

NO_RISK_REASON = "No elevated risk factors - this order scores below the base rate."


@dataclass(frozen=True)
class Explanation:
    values: np.ndarray            # (n, n_features) SHAP in margin space
    base_value: float
    feature_names: tuple[str, ...]


class ReasonExplainer:
    """
    TreeSHAP explainer plus the reason-mapping layer.

    The explainer is built once and held - section 8's "explainer cached at startup".
    Building it is the expensive part; scoring is a tree walk.
    """

    #: shap refuses interventional perturbation on models with categorical splits.
    #: See the module docstring for what that costs and how to remove the constraint.
    PERTURBATION = "tree_path_dependent"

    def __init__(self, bundle) -> None:
        self.bundle = bundle
        self.feature_names = tuple(bundle.feature_names)
        self._explainer = shap.TreeExplainer(
            bundle.booster,
            feature_perturbation=self.PERTURBATION,
            model_output="raw",
        )

    # -- core ---------------------------------------------------------------------------

    def explain(self, X: pd.DataFrame) -> Explanation:
        if list(X.columns) != list(self.feature_names):
            raise ValueError(
                "feature frame does not match the trained contract; refusing to explain "
                "a frame the model would not score the same way"
            )
        values = self._explainer.shap_values(X)
        if isinstance(values, list):          # older/other shap return shapes
            values = values[-1]
        return Explanation(
            values=np.asarray(values, dtype=float),
            base_value=float(np.ravel(self._explainer.expected_value)[-1]),
            feature_names=self.feature_names,
        )

    def assert_additive(
        self, expl: Explanation, margin: np.ndarray, tol: float = 1e-6
    ) -> float:
        """
        sum_j phi_j + base == model margin, to floating point.

        A failure means the explainer is misconfigured - wrong output space, wrong
        feature order, wrong iteration count - and every reason string downstream would
        be quietly wrong.
        """
        recon = expl.values.sum(axis=1) + expl.base_value
        err = float(np.abs(recon - np.asarray(margin, dtype=float)).max())
        if err > tol:
            raise AdditivityError(
                f"SHAP values do not reconstruct the model output: max error {err:.3e} "
                f"exceeds {tol:.0e}. The explainer is misconfigured."
            )
        return err

    # -- grouping -----------------------------------------------------------------------

    def group_attributions(self, expl: Explanation) -> pd.DataFrame:
        """Per-order SHAP summed within each feature group."""
        groups = sorted({FEATURE_GROUP[f] for f in self.feature_names})
        out = {}
        for g in groups:
            cols = [i for i, f in enumerate(self.feature_names) if FEATURE_GROUP[f] == g]
            out[g] = expl.values[:, cols].sum(axis=1)
        return pd.DataFrame(out, index=range(expl.values.shape[0]))

    def reason_buckets(self, expl: Explanation) -> np.ndarray:
        """
        The dominant reason group per order, by summed **positive** attribution.

        This replaces the pre-SHAP stand-in in policy/effectiveness.py: it ranks groups
        by how much each actually pushed the score up, rather than by how unusual the
        order looks on a representative feature.
        """
        groups = sorted({FEATURE_GROUP[f] for f in self.feature_names})
        pos = np.zeros((expl.values.shape[0], len(groups)))
        for j, g in enumerate(groups):
            cols = [i for i, f in enumerate(self.feature_names) if FEATURE_GROUP[f] == g]
            pos[:, j] = np.clip(expl.values[:, cols], 0.0, None).sum(axis=1)
        return np.array(groups)[np.argmax(pos, axis=1)]

    # -- reasons ------------------------------------------------------------------------

    def top_reasons(self, expl: Explanation, k: int = 3) -> list[list[str]]:
        """
        Top ``k`` risk-increasing reasons per order, as merchant-facing sentences.

        Only positive attributions are returned: the question a merchant is asking is
        "why is this order risky", and a feature that *reduced* risk is not an answer to
        it.  An order with no positive attribution gets the explicit no-risk sentence
        rather than an empty list - an empty reason set would be indistinguishable from a
        failure of the reason layer.
        """
        out: list[list[str]] = []
        order = np.argsort(-expl.values, axis=1, kind="stable")
        for i in range(expl.values.shape[0]):
            reasons: list[str] = []
            for j in order[i, :]:
                if expl.values[i, j] <= 0.0 or len(reasons) >= k:
                    break
                template = REASON_TEMPLATES.get(self.feature_names[j])
                if template and template not in reasons:
                    reasons.append(template)
            out.append(reasons or [NO_RISK_REASON])
        return out

    # -- monotonicity -------------------------------------------------------------------

    def probe_monotone(
        self, X: pd.DataFrame, feature: str, n_steps: int = 25, row: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sweep one feature across its observed range on a single fixed order and return
        (values, attributions).

        This is the rigorous form of the section 5.3 check.  A monotone constraint says
        the model's response to that feature is a fixed direction; holding every other
        feature constant isolates it, so the attribution must move in the constrained
        direction and nothing else can be responsible.
        """
        if feature not in MONOTONE_CONSTRAINTS:
            raise KeyError(f"{feature!r} carries no monotone constraint")

        col = X[feature].astype(float)
        lo, hi = float(np.nanquantile(col, 0.01)), float(np.nanquantile(col, 0.99))
        grid = np.linspace(lo, hi, n_steps)

        probe = pd.concat([X.iloc[[row]]] * n_steps, ignore_index=True)
        probe[feature] = grid
        probe = probe[list(self.feature_names)]

        expl = self.explain(probe)
        j = self.feature_names.index(feature)
        return grid, expl.values[:, j]
