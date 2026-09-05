"""
LightGBM training.  ARCHITECTURE.md 5.1, 5.2, 5.3, build step 3.

Natural class distribution, no resampling.  The policy layer in section 7 runs on
probabilities, so reweighting would buy recall and wreck calibration; section 5.5 keeps
that as an ablation rather than a default.

Nothing here calibrates.  The booster's raw output is a score, not a calibrated
probability - the Platt map is step 4 and does not exist yet.

--------------------------------------------------------------------------------------
THE COLUMN-ORDER CONTRACT
--------------------------------------------------------------------------------------

LightGBM addresses features positionally.  A frame with the right columns in the wrong
order produces predictions that are silently wrong - every value is a real number in a
plausible range, and nothing raises.  Monotone constraints make it worse: they are also
positional, so a reordered frame gets the pincode constraint applied to whichever
feature now sits at that index.

So the trained column order is recorded in the bundle and
:func:`predict` requires an exact match.  It does **not** reorder to fit.  Reordering
would paper over a caller that has lost track of its own schema, and the next such bug
would be silent again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data.loader import OlistLoader
from models.constraints import build_constraint_vector

__all__ = [
    "PARAMS",
    "NUM_BOOST_ROUND",
    "EARLY_STOPPING_ROUNDS",
    "DROPPED_CONSTANT_COLUMNS",
    "CATEGORICAL_FEATURES",
    "ORDER_ONLY_GROUPS",
    "ORDER_ONLY_ABSENT_CONSTRAINTS",
    "FeatureContractError",
    "ModelBundle",
    "train",
    "predict",
]

SEED = 42

#: ARCHITECTURE.md 5.2 - regularised for ~1% prevalence.  Defaults overfit roughly a
#: thousand positives; every value below sits inside the stated band.
PARAMS: dict = {
    "objective": "binary",
    # Not AUC, not logloss.  Section 5.2 is explicit: early stopping is scored on
    # average precision, because that is the metric section 9 reports.
    "metric": "average_precision",
    "num_leaves": 31,            # band 15-31
    "max_depth": 5,              # band 4-6
    "min_child_samples": 75,     # band 50-100
    "lambda_l2": 1.0,            # nonzero
    "feature_fraction": 0.7,     # ~0.7
    "bagging_fraction": 1.0,     # no resampling of rows
    "learning_rate": 0.05,
    # Reproducibility.  deterministic=true is only honoured alongside a forced
    # tree-building direction, and it guarantees identical results for the same data and
    # the same thread count - so the thread count is pinned too, or the artifact stops
    # being reproducible across machines.
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "seed": SEED,
    "bagging_seed": SEED,
    "feature_fraction_seed": SEED,
    "data_random_seed": SEED,
    "verbose": -1,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100

#: Constant on this panel, so it cannot produce a split.  Dropped from the trained
#: matrix because its presence overstates the feature contract - a column the model
#: cannot use should not appear in a feature list a reader takes as capability.  It stays
#: in the contract, labelled a no-op (ARCHITECTURE.md 4.2 rule).
DROPPED_CONSTANT_COLUMNS: tuple[str, ...] = ("has_zip_prefix",)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "product_category",
    "customer_state",
    "seller_state",
)

#: The order-only baseline: what the model scores with no history and no pincode
#: encodings.  Given customer history is 97% empty, this is the comparison that says
#: what the encodings actually bought.
ORDER_ONLY_GROUPS: tuple[str, ...] = ("order",)

#: The order-only baseline carries none of the required constrained features by
#: construction: all three live in the customer and pincode groups.  Named explicitly
#: rather than derived from MONOTONE_CONSTRAINTS, so that adding a fourth constraint to a
#: feature the baseline *does* include fails loudly instead of being waved through.
#: (UNSHIPPED_CONSTRAINTS is not listed here - it is not required of any matrix.)
ORDER_ONLY_ABSENT_CONSTRAINTS: frozenset[str] = frozenset({
    "pincode_failure_rate_smoothed",
    "cust_prior_failures",
    "cust_prior_boleto_ratio",
})


class FeatureContractError(ValueError):
    """Raised when a frame does not match the trained feature contract exactly."""


@dataclass(frozen=True)
class ModelBundle:
    """A booster plus everything needed to score with it correctly."""

    booster: lgb.Booster
    feature_names: tuple[str, ...]
    monotone_constraints: tuple[int, ...]
    category_levels: dict[str, tuple[str, ...]]
    target: str
    population: str
    best_iteration: int
    params: dict = field(default_factory=dict)

    @property
    def model_text(self) -> str:
        return self.booster.model_to_string()

    @property
    def model_sha256(self) -> str:
        return hashlib.sha256(self.model_text.encode("utf-8")).hexdigest()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"))
        (directory / "contract.json").write_text(
            json.dumps(
                {
                    "feature_names": list(self.feature_names),
                    "monotone_constraints": list(self.monotone_constraints),
                    "category_levels": {
                        k: list(v) for k, v in self.category_levels.items()
                    },
                    "target": self.target,
                    "population": self.population,
                    "best_iteration": self.best_iteration,
                    "model_sha256": self.model_sha256,
                    "params": self.params,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------------------
# Matrix preparation
# --------------------------------------------------------------------------------------

def prepare_matrix(
    matrix: pd.DataFrame,
    category_levels: dict[str, tuple[str, ...]] | None = None,
    drop: tuple[str, ...] = DROPPED_CONSTANT_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """
    Drop no-op columns and encode categoricals against a fixed level set.

    ``category_levels`` is derived from the training split only when not supplied, so a
    level first seen in validation or test maps to NaN rather than silently extending
    the encoding.
    """
    out = matrix.drop(columns=[c for c in drop if c in matrix.columns]).copy()

    levels: dict[str, tuple[str, ...]] = {}
    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            continue
        if category_levels is not None and col in category_levels:
            cats = list(category_levels[col])
        else:
            cats = sorted(out[col].dropna().unique().tolist())
        # Map levels absent from the training vocabulary to NaN explicitly.  Letting
        # astype() coerce them is deprecated and will raise; more to the point, the
        # intent should be visible - an unseen category is missing information, not an
        # error, and the booster handles it as such.
        known = out[col].where(out[col].isin(cats))
        out[col] = pd.Categorical(known, categories=cats)
        levels[col] = tuple(cats)

    return out, levels


def _assert_training_frame_is_clean(X: pd.DataFrame) -> None:
    """
    The same leakage gate applied to checkout_frame(), applied again at the training
    boundary.

    checkout_frame() guards the raw-column boundary; this guards the boundary that
    actually matters, because a feature matrix reaches the model through here and not
    through there.  Cheap, and it means no future feature group can route around the
    check by constructing its own frame.
    """
    OlistLoader.assert_no_leakage(X)

    labels = {"label_a", "label_b", "entered_fulfillment", "order_status", "split"}
    offenders = sorted(set(X.columns) & labels)
    if offenders:
        raise FeatureContractError(
            "label or bookkeeping column(s) reached train(): " + ", ".join(offenders)
        )

    identifiers = {"order_id", "customer_id", "customer_unique_id"}
    offenders = sorted(set(X.columns) & identifiers)
    if offenders:
        raise FeatureContractError(
            "identifier column(s) reached train(): " + ", ".join(offenders)
        )


# --------------------------------------------------------------------------------------
# Train / predict
# --------------------------------------------------------------------------------------

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    target: str,
    population: str,
    category_levels: dict[str, tuple[str, ...]],
    params: dict | None = None,
    allow_missing_constraints: frozenset[str] = frozenset(),
) -> ModelBundle:
    """
    Fit one booster.  Natural class distribution; no ``scale_pos_weight``.

    Early stopping is scored on average precision over ``X_valid`` (ARCHITECTURE.md
    5.2).  The validation frame must carry the same columns in the same order as the
    training frame.
    """
    _assert_training_frame_is_clean(X_train)
    _assert_training_frame_is_clean(X_valid)

    if list(X_train.columns) != list(X_valid.columns):
        raise FeatureContractError(
            "train and validation frames disagree on column order or membership"
        )

    feature_names = tuple(X_train.columns)
    constraints = build_constraint_vector(feature_names, allow_missing_constraints)
    if len(constraints) != len(feature_names):  # pragma: no cover
        raise FeatureContractError(
            f"constraint vector length {len(constraints)} != "
            f"feature count {len(feature_names)}"
        )

    p = dict(params or PARAMS)
    p["monotone_constraints"] = constraints

    categorical = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    dtrain = lgb.Dataset(
        X_train, label=y_train.astype(int),
        categorical_feature=categorical, free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid, label=y_valid.astype(int), reference=dtrain,
        categorical_feature=categorical, free_raw_data=False,
    )

    booster = lgb.train(
        p,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    return ModelBundle(
        booster=booster,
        feature_names=feature_names,
        monotone_constraints=tuple(constraints),
        category_levels={
            k: v for k, v in category_levels.items() if k in feature_names
        },
        target=target,
        population=population,
        best_iteration=int(booster.best_iteration or booster.num_trees()),
        params=p,
    )


def predict(
    bundle: ModelBundle, X: pd.DataFrame, raw_score: bool = False
) -> np.ndarray:
    """
    Score ``X``, requiring an exact match against the trained feature contract.

    ``raw_score=True`` returns the **log-odds margin** rather than the sigmoid of it.
    That is what the calibration layer consumes: Platt scaling fits a linear map in
    logit space, so feeding it LightGBM's already-squashed probability stacks a
    second sigmoid on the first and produces an extreme slope with a long artificial
    tail. Ranking metrics are unaffected either way - the sigmoid is monotone.

    Raises
    ------
    FeatureContractError
        If columns are missing, extra, or **present in a different order**.  A
        permutation is rejected rather than repaired: LightGBM addresses features
        positionally, and so do the monotone constraints, so a silently reordered frame
        scores every row against the wrong feature and applies the pincode constraint to
        whatever now sits at that index.
    """
    got = list(X.columns)
    want = list(bundle.feature_names)
    if got == want:
        return bundle.booster.predict(
            X, num_iteration=bundle.best_iteration, raw_score=raw_score
        )

    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing or extra:
        raise FeatureContractError(
            "feature frame does not match the trained contract. "
            + (f"missing: {missing}. " if missing else "")
            + (f"unexpected: {extra}. " if extra else "")
        )

    first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
    raise FeatureContractError(
        "feature frame has the correct columns in the wrong ORDER; refusing to reorder "
        f"silently. First mismatch at position {first}: got {got[first]!r}, "
        f"expected {want[first]!r}. LightGBM addresses features positionally and the "
        "monotone constraints are positional too, so scoring this frame would apply "
        "each constraint to the wrong feature."
    )
